import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, random_split
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')

# 【新增】验证逻辑函数
def validate(
    model,
    loader,
    loss_fct,
    autocast_ctx,
    device
):
    model.eval()

    # 本地统计（Python float，避免 GPU 同步）
    local_loss_sum = 0.0
    local_batch_count = 0

    with torch.no_grad():
        for X, Y, loss_mask in loader:
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)
            loss_mask = loss_mask.to(device, non_blocking=True)

            with autocast_ctx:
                out = model(X)
                loss = loss_fct(
                    out.logits.view(-1, out.logits.size(-1)),
                    Y.view(-1)
                ).view(Y.size())

                loss = (loss * loss_mask).sum() / loss_mask.sum()
                if hasattr(out, "aux_loss"):
                    loss = loss + out.aux_loss

            local_loss_sum += loss.item()
            local_batch_count += 1

    # ===============================
    # 🔒 single all-reduce (关键优化点)
    # ===============================
    if dist.is_initialized():
        stats = torch.tensor(
            [local_loss_sum, local_batch_count],
            device=device,
            dtype=torch.float64  # 精度更稳
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        global_loss = stats[0] / stats[1]
    else:
        global_loss = local_loss_sum / max(1, local_batch_count)

    model.train()
    return global_loss.item()



def train_epoch(
    epoch,
    model,
    train_loader,
    val_loader,
    optimizer,
    scaler,
    autocast_ctx,
    start_step,
    iters,
    args,
    wandb=None
):
    global best_val_loss

    model.train()
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    val_total_time = 0.0

    for step, (X, Y, loss_mask) in enumerate(train_loader, start=start_step + 1):
        global_step = epoch * iters + step

        X = X.to(args.device, non_blocking=True)
        Y = Y.to(args.device, non_blocking=True)
        loss_mask = loss_mask.to(args.device, non_blocking=True)

        # ---- LR ----
        lr = get_lr(global_step, args.epochs * iters, args.learning_rate)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ---- forward ----
        with autocast_ctx:
            out = model(X)
            loss = loss_fct(
                out.logits.view(-1, out.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())

            loss = (loss * loss_mask).sum() / loss_mask.sum()
            if hasattr(out, "aux_loss"):
                loss = loss + out.aux_loss

            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        # ---- step ----
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # ============================================================
        # ✅ DDP-safe VALIDATION（最关键）
        # ============================================================
        if (step + 1) % args.eval_interval == 0:
            val_start = time.time()

            val_loss = validate(
                model.module if isinstance(model, DistributedDataParallel) else model,
                val_loader,
                loss_fct,
                autocast_ctx,
                args.device
            )

            val_time = time.time() - val_start
            val_total_time += val_time

            # ---- rank0 only side-effects ----
            if is_main_process():
                Logger(
                    f">>> Step {step} | "
                    f"val_loss={val_loss:.6f} | "
                    f"val_time={val_time:.1f}s"
                )

                if wandb:
                    wandb.log(
                        {
                            "val_loss": val_loss,
                            "val_time": val_time,
                        },
                        step=epoch * iters + step
                    )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_path = f"{args.save_dir}/{args.save_weight}_best.pth"
                    state_dict = (
                        model.module.state_dict()
                        if isinstance(model, DistributedDataParallel)
                        else model.state_dict()
                    )
                    torch.save(
                        {k: v.half() for k, v in state_dict.items()},
                        save_path
                    )

        # ============================================================
        # LOG
        # ============================================================
        if step % args.log_interval == 0 or step == iters - 1:
            actual_train_time = time.time() - start_time - val_total_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = actual_train_time / (step + 1) * iters // 60 - actual_train_time // 60
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min:')
            if wandb: wandb.log({"loss": current_loss, "lr": current_lr}, step=epoch * iters + step)

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
            torch.save({k: v.half() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scaler=scaler)
            model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Full SFT with Validation")
    # ... 原有参数保持不变 ...
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='full_sft', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-7, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=512, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_mini_512.jsonl", help="训练数据路径")
    # 【新增】验证相关参数
    parser.add_argument("--eval_interval", type=int, default=50000, help="每隔多少step进行一次验证")
    parser.add_argument('--from_weight', default='pretrain', type=str, help="基于哪个权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否续训")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT", help="wandb项目名")
    args = parser.parse_args()

    # ========== 1. 环境初始化 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置与检查点 ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 混合精度设置 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. WandB 初始化 ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import wandb
        wandb.init(project=args.wandb_project, name=f"SFT-{time.strftime('%m%d-%H%M')}")
    
    # ========== 5. 模型与数据加载 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    
    # 加载数据集
    full_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # --- 新增划分逻辑 ---
    val_size = int(0.1 * len(full_ds))
    train_size = len(full_ds) - val_size
    # 由于前面执行了 setup_seed(42)，此处 random_split 在多进程下也是确定且一致的
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    # 训练集加载
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # 验证集加载
    val_sampler = DistributedSampler(
        val_ds,
        shuffle=False
    ) if dist.is_initialized() else None


    # val_loader = DataLoader(val_ds, batch_size=args.batch_size, pin_memory=True, drop_last=False)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 状态恢复与 DDP ==========
    start_epoch, start_step = 0, 0
    best_val_loss = float('inf') # 初始化全局最佳损失为无穷大
    
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 7. 训练循环 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        val_sampler and val_sampler.set_epoch(epoch)
        if epoch == start_epoch and start_step > 0:
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)

            val_batch_sampler = SkipBatchSampler(val_sampler or range(len(val_ds)), args.batch_size, start_step + 1)
            val_loader = DataLoader(val_ds, batch_sampler=val_batch_sampler, num_workers=args.num_workers, pin_memory=True)
            
            train_epoch(epoch, model, loader, val_loader, optimizer, scaler, autocast_ctx, start_step, len(loader) + start_step + 1, args, wandb)
        else:
            loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=(val_sampler is None), sampler=val_sampler, num_workers=args.num_workers, pin_memory=True)
            train_epoch(epoch, model, loader, val_loader, optimizer, scaler, autocast_ctx, 0, len(loader), args, wandb)
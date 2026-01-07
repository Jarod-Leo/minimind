"""
训练工具函数集合
"""
import os
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from transformers import AutoTokenizer
from model.model_minimind import MiniMindForCausalLM


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def init_distributed_mode():
    """
    初始化分布式训练（DDP）模式。
    通过环境变量判断是否处于分布式环境。
    """
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    # 使用 NVIDIA NCCL 后端进行多卡通信
    dist.init_process_group(backend="nccl")
    # LOCAL_RANK 是当前节点上 GPU 的编号
    local_rank = int(os.environ["LOCAL_RANK"])
    # 将模型和数据绑定到指定的 GPU
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    """
    设置全局随机种子，确保实验结果的可复现性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # cpu种子
    torch.cuda.manual_seed(seed) # GPU种子
    torch.cuda.manual_seed_all(seed) # 所有GPU种子
    # 强制 cuDNN 使用确定性算法（略微降低性能，但结果一致）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    """
    双重任务：1. 保存当前权重和训练状态 2. 加载状态以实现续训
    """
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if lm_config.use_moe else ''
    # 路径1：最终模型权重路径 (仅含模型参数)
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    # 路径2：恢复训练所需的完整状态路径 (模型 + 优化器 + 步数 + WandB ID)
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        from torch.nn.parallel import DistributedDataParallel
        # 如果是分布式模型，需要取出 .module 里的原始权重，否则加载时会报错
        state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
        # 1. 保存轻量化权重 (用于评估和部署)
        ckp_tmp = ckp_path + '.tmp'
        # 转为半精度 (half) 以节省 50% 的磁盘空间
        torch.save({k: v.half() for k, v in state_dict.items()}, ckp_tmp)
        os.replace(ckp_tmp, ckp_path) # 原子替换，防止写入崩溃导致文件损坏
        # 2. 准备续训数据
        wandb_id = None
        if wandb: # 记录 wandb ID 以便续训时合并曲线
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        # 处理其他状态（如 scaler 混合精度缩放器）
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    if isinstance(value, DistributedDataParallel):
                        resume_data[key] = value.module.state_dict()
                    else:
                        resume_data[key] = value.state_dict()
                else:
                    resume_data[key] = value
        # 保存续训文件
        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            # 关键逻辑：如果 GPU 数量发生变化（如从 8 卡减到 4 卡），需要等比缩放 step
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device='cuda'):
    """
    初始化模型架构并加载初始权重。
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    if from_weight!= 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        # 加载预训练权重
        weights = torch.load(weight_path, map_location=device)
        # strict=False 允许加载不完全匹配的权重（例如只加载部分层）
        model.load_state_dict(weights, strict=False)

    Logger(f'所加载Model可训练参数：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model.to(device), tokenizer


class SkipBatchSampler(Sampler):
    """
    自定义采样器：用于在续训时跳过已经训练过的 batch。
    比如训练在中途断了，重启时跳过前 1000 个 batch，直接从 1001 开始。
    """
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                # 如果还没跳够指定的 batch 数量，则抛弃当前 batch 且不 yield
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        # 处理最后剩余的样本        
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        # 计算总 batch 数减去跳过的数
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


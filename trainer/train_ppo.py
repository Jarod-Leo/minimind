import os
import sys

# 设置包路径，确保能导入 model 和 dataset 模块
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import re
import warnings
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model

warnings.filterwarnings('ignore')


# 自定义的Critic模型，继承自MiniMindLM
# 作用：预测一个状态（文本序列）的预期总奖励，用于计算 Advantage（优势）
class CriticModel(MiniMindForCausalLM):
    def __init__(self, params):
        super().__init__(params)
        # 替换lm_head为输出单一价值的线性层
        self.value_head = nn.Linear(params.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # 使用基础模型获取隐藏状态
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = self.model.norm(outputs[0])
        # 使用value_head获取价值估计，将隐藏层向量映射为单一的价值得分
        values = self.value_head(hidden_states).squeeze(-1)
        return values


def calculate_rewards(prompts, responses, reward_model, reward_tokenizer):
    """
    整合所有奖励函数计算总奖励
    推理模型专用奖励：鼓励模型使用 <think> 和 <answer> 标签
    """
    def reasoning_model_reward(rewards):
        # 1. 格式奖励（仅针对训练推理模型时使用）检查回答是否严格符合特定的 XML 结构
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"

        matches_pattern = [re.match(pattern, response, re.S) for response in responses]
        matches_pattern2 = [re.match(pattern2, response, re.S) for response in responses]

        format_rewards = []
        for match_pattern, match_pattern2 in zip(matches_pattern, matches_pattern2):
            if match_pattern:
                format_rewards.append(0.5) # 格式正确加 0.5 分
            elif match_pattern2:
                format_rewards.append(0.5)
            else:
                format_rewards.append(0.0)
        rewards += torch.tensor(format_rewards, device=args.device)

        # 2. 标记奖励（防止严格奖励稀疏，仅针对训练推理模型时使用）
        def mark_num(text):
            reward = 0
            # 每个标签出现且仅出现一次加 0.25
            if text.count("<think>") == 1:
                reward += 0.25
            if text.count("</think>") == 1:
                reward += 0.25
            if text.count("<answer>") == 1:
                reward += 0.25
            if text.count("</answer>") == 1:
                reward += 0.25
            return reward

        mark_rewards = [mark_num(response) for response in responses]
        rewards += torch.tensor(mark_rewards, device=args.device)
        return rewards

    rewards = torch.zeros(len(responses), device=args.device)

    # 格式奖励，如果是推理模型训练，增加规则奖励
    if args.reasoning == 1:
        rewards = reasoning_model_reward(rewards)

    # 使用外部 Reward Model (如 InternLM2-Reward) 计算内容质量分
    with torch.no_grad():
        reward_model_scores = []
        for prompt, response in zip(prompts, responses):
            # 这里的正则用于解析 ChatML 格式，提取对话历史
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()} for role, content in matches]

            # 拼入模型当前的回答，送入奖励模型打分
            tmp_chat = messages + [{"role": "assistant", "content": response}]
            score = reward_model.get_score(reward_tokenizer, tmp_chat)

            # 限制评分范围 (Clipping)
            scale = 3.0
            score = max(min(score, scale), -scale)

            # 推理模型特殊处理：如果存在 <answer> 标签，重点对答案内容打分
            # 当args.reasoning=1时，额外计算<answer>内容的奖励
            if args.reasoning == 1:
                answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                if answer_match:
                    answer_content = answer_match.group(1).strip()
                    # 对answer内容单独计算reward
                    tmp_chat = messages + [{"role": "assistant", "content": answer_content}]
                    answer_score = reward_model.get_score(reward_tokenizer, tmp_chat)
                    answer_score = max(min(answer_score, scale), -scale)
                    # 综合评分：40% 整体表现 + 60% 最终答案质量
                    score = score * 0.4 + answer_score * 0.6
            reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards

# PPO 训练主循环，涉及 4 个模型：Actor (训练中), Old Actor (采样), Ref (KL惩罚), Critic (估值)
def ppo_train_epoch(epoch, loader, iters, old_actor_model, ref_model, actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, start_step=0, wandb=None):
    actor_model.train()
    critic_model.train()

    for step, batch in enumerate(loader, start=start_step + 1):
        # 1. 编码 Prompt
        prompts = batch["prompt"]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_len).to(args.device)
        prompt_lengths = enc.attention_mask.sum(dim=1) # [B]

        # 2. 采样阶段 (Experience Collection)
        with torch.no_grad():
            model_for_gen = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            gen_out = model_for_gen.generate(
                input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                max_new_tokens=args.max_gen_len, do_sample=True, temperature=0.8,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        
        B, total_len = gen_out.shape
        full_mask = (gen_out != tokenizer.pad_token_id).long() # [B, L](L = P + R_actual)

        # 3. 计算所有模型的 Log-probs
        # labels 是生成序列的后移一位
        labels = gen_out[:, 1:].clone() # [B, L-1]
        
        # 计算当前 Actor, Old Actor, Reference 的 Log-probs
        def get_logp(model, ids, mask):
            logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1, :]
            return F.log_softmax(logits, dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)

        logp_tokens = get_logp(actor_model, gen_out, full_mask) # [B, L-1]
        with torch.no_grad():
            old_logp_tokens = get_logp(old_actor_model, gen_out, full_mask) # [B, L-1]
            ref_logp_tokens = get_logp(ref_model, gen_out, full_mask) # [B, L-1]
            # 计算 RM 奖励 (Sequence-level)
            responses_text = [tokenizer.decode(gen_out[i, prompt_lengths[i]:], skip_special_tokens=True) for i in range(B)]
            rm_rewards = calculate_rewards(prompts, responses_text, reward_model, reward_tokenizer) # [B]

        # 4. 奖励塑形 (Reward Shaping) - Token-level
        # 构造每一个 token 的奖励：即时奖励 = KL 惩罚
        kl_penalty = -args.kl_coef * (logp_tokens.detach() - ref_logp_tokens) # [B, L-1]
        
        # 定义 Response Mask (仅在回答部分计算)
        seq_len_m1 = total_len - 1 #[L-1]
        resp_mask = torch.arange(seq_len_m1, device=args.device).unsqueeze(0) >= (prompt_lengths.unsqueeze(1) - 1) # [1, L-1]
        final_mask = resp_mask & (~labels.eq(tokenizer.pad_token_id)) # [B, L-1]

        step_rewards = kl_penalty * final_mask # [B, L-1]
        # 将 RM 奖励加在最后一个有效 token 上
        last_token_indices = final_mask.sum(dim=1) + prompt_lengths - 2 # [B]
        for i in range(B):
            step_rewards[i, last_token_indices[i]] += rm_rewards[i]

        # 5. 计算 Advantage (GAE)
        values_seq = critic_model(input_ids=gen_out, attention_mask=full_mask) # [B, L]
        # 对齐索引：values_seq[:, t] 是在看到 token t 之前的预估值，用来预测 token t+1 的收益
        v_preds = values_seq[:, :-1] # [B, L-1]
        
        advantages = torch.zeros_like(step_rewards) # [B, L-1]
        last_gae_lam = 0
        gamma = args.gamma
        lam = args.gae_lambda

        # 逆向 GAE 计算
        for t in reversed(range(seq_len_m1)):
            next_v = values_seq[:, t+1] if t < seq_len_m1 - 1 else 0
            delta = step_rewards[:, t] + gamma * next_v - values_seq[:, t]
            advantages[:, t] = last_gae_lam = delta + gamma * lam * last_gae_lam
        
        returns = advantages + v_preds # [B, L-1]
        # Advantage 标准化 (重要：稳定训练)
        advantages = (advantages - advantages[final_mask].mean()) / (advantages[final_mask].std() + 1e-8)

        # 6. PPO Loss 构造 (Token-level)
        ratio = torch.exp(logp_tokens - old_logp_tokens)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages
        
        # 只在 response 区域计算损失
        policy_loss = - (torch.min(surr1, surr2) * final_mask).sum() / final_mask.sum()
        # Value loss (Critic 拟合 returns)
        value_loss = ((v_preds - returns.detach())**2 * final_mask).sum() / final_mask.sum()
        
        # 综合 Loss
        loss = policy_loss + args.vf_coef * value_loss
        loss.backward()

        # 8. 梯度累积 & 参数更新
        if (step + 1) % args.accumulation_steps == 0:
            clip_grad_norm_(actor_model.parameters(), args.grad_clip)
            clip_grad_norm_(critic_model.parameters(), args.grad_clip)
            actor_optimizer.step()
            critic_optimizer.step()
            actor_scheduler.step()
            critic_scheduler.step()
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            torch.cuda.empty_cache()

        # 9. 日志与监控（主进程）
        if is_main_process():
            response_ids = gen_out[:, enc.input_ids.shape[1]:]
            is_eos = (response_ids == tokenizer.eos_token_id)
            eos_indices = torch.argmax(is_eos.int(), dim=1)
            has_eos = is_eos.any(dim=1)
            lengths = torch.where(has_eos, eos_indices + 1, torch.tensor(response_ids.shape[1], device=is_eos.device))
            avg_len = lengths.float().mean()

            actor_loss_val = policy_loss.item()
            critic_loss_val = value_loss.item()
            reward_val = rewards.mean().item()
            kl_val = kl.item()
            kl_ref_val = kl_ref.item()
            avg_len_val = avg_len.item()
            actor_lr = actor_optimizer.param_groups[0]['lr']
            critic_lr = critic_optimizer.param_groups[0]['lr']

            if wandb is not None:
                wandb.log({
                    "actor_loss": actor_loss_val,
                    "critic_loss": critic_loss_val,
                    "reward": reward_val,
                    "kl": kl_val,
                    "kl_ref": kl_ref_val,
                    "avg_response_len": avg_len_val,
                    "actor_lr": actor_lr,
                })

            Logger(f"Epoch: {epoch+1}, Step: {step}/{iters}, "
                   f"Actor Loss: {actor_loss_val:.6f}, Critic Loss: {critic_loss_val:.6f}, "
                   f"Reward: {reward_val:.6f}, KL: {kl_val:.6f}, KL_ref: {kl_ref_val:.6f}, "
                   f"Avg Response Len: {avg_len_val:.2f}, Actor LR: {actor_lr:.2e}, Critic LR: {critic_lr:.2e}")
        
        # 10. 同步 old_actor（PPO 关键步骤）
        if (step + 1) % args.update_old_actor_freq == 0:
            state_dict = actor_model.module.state_dict() if isinstance(actor_model, DistributedDataParallel) else actor_model.state_dict()
            old_actor_model.load_state_dict({k: v.detach().cpu() for k, v in state_dict.items()})
            old_actor_model.to(args.device)

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            actor_model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            actor_state = actor_model.module.state_dict() if isinstance(actor_model, DistributedDataParallel) else actor_model.state_dict()
            torch.save({k: v.half() for k, v in actor_state.items()}, ckp)
            
            # 使用 lm_checkpoint 保存完整状态（包括 critic）
            lm_checkpoint(lm_config, weight=args.save_weight, model=actor_model, optimizer=actor_optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints',
                         scheduler=actor_scheduler, critic_model=critic_model, 
                         critic_optimizer=critic_optimizer, critic_scheduler=critic_scheduler)
            actor_model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind PPO (Proximal Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='ppo_actor', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=8e-8, help="Actor学习率")
    parser.add_argument("--critic_learning_rate", type=float, default=8e-8, help="Critic学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=66, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1536, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif-mini.jsonl", help="RLAIF数据路径")
    parser.add_argument("--clip_epsilon", type=float, default=0.1, help="PPO裁剪参数")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function系数")
    parser.add_argument("--kl_coef", type=float, default=0.02, help="KL散度惩罚系数")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子 (Discount Factor)")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE 平滑因子")
    parser.add_argument("--reasoning", type=int, default=1, choices=[0, 1], help='推理模型类型（0=普通模型，1=推理模型）')
    parser.add_argument("--update_old_actor_freq", type=int, default=4, help="更新old_actor_model的频率")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-PPO", help="wandb项目名")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-PPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 初始化模型和数据 ==========
    base_weight = "reason" if args.reasoning == 1 else "full_sft"
    # Actor模型
    actor_model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    tokenizer.padding_side = 'left'  # PPO需要左侧padding
    # Old Actor模型
    old_actor_model, _ = init_model(lm_config, base_weight, device=args.device)
    old_actor_model = old_actor_model.eval().requires_grad_(False)
    # Reference模型
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    # Critic模型
    moe_suffix = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/{base_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    critic_model = CriticModel(lm_config)
    critic_model.load_state_dict(state_dict, strict=False)
    critic_model = critic_model.to(args.device)
    # Reward模型
    reward_model = AutoModel.from_pretrained(
        args.reward_model_path, torch_dtype=torch.float16, trust_remote_code=True
    )
    reward_model = reward_model.to(args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)
    # 数据和优化器
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=(args.max_seq_len + args.max_gen_len))
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = (iters // args.accumulation_steps) * args.epochs
    actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=total_optimizer_steps, eta_min=args.critic_learning_rate / 10)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        actor_model.load_state_dict(ckp_data['model'])
        critic_model.load_state_dict(ckp_data['critic_model'])
        actor_optimizer.load_state_dict(ckp_data['optimizer'])
        critic_optimizer.load_state_dict(ckp_data['critic_optimizer'])
        actor_scheduler.load_state_dict(ckp_data['scheduler'])
        critic_scheduler.load_state_dict(ckp_data['critic_scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        actor_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        critic_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])
        critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank])
        old_actor_model.to(args.device)
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        if epoch == start_epoch and start_step > 0:  # 第一个epoch且存在检查点
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            ppo_train_epoch(epoch, loader, len(loader) + start_step + 1, old_actor_model, ref_model, 
                           actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, start_step, wandb)
        else:  # 默认从头开始
            loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None), 
                              sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
            ppo_train_epoch(epoch, loader, len(loader), old_actor_model, ref_model, 
                           actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, 0, wandb)

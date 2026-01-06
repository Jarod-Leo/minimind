# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = 'silu', # 激活函数默认使用silu
            hidden_size: int = 512, # 隐藏层维度
            intermediate_size: int = None, # FFN中间层维度
            max_position_embeddings: int = 32768, # 最大序列长度
            num_attention_heads: int = 8, # Query头总数
            num_hidden_layers: int = 8, # Transformer层数
            num_key_value_heads: int = 2, # key/value头数
            vocab_size: int = 6400, # 词表大小
            rms_norm_eps: float = 1e-05, # RMSNorm 的 epsilon
            rope_theta: int = 1000000.0, # RoPE 基数，较大值有助于长文本
            inference_rope_scaling: bool = False, # 是否开启推理时的 RoPE 缩放 (YaRN)
            flash_attn: bool = True, # 是否使用 Flash Attention
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = False,
            num_experts_per_tok: int = 2, # 每个 Token 激活的专家数量 (Top-K)
            n_routed_experts: int = 4, # 总的可选专家（路由专家）数量
            n_shared_experts: int = 1, # 共享专家数量（始终参与计算）
            scoring_func: str = 'softmax', # 门控评分函数
            aux_loss_alpha: float = 0.1, # 负载均衡辅助损失的系数
            seq_aux: bool = True, # 是否在序列级别计算辅助损失
            norm_topk_prob: bool = True, # 是否对 Top-K 的概率进行归一化
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        # YaRN 旋转位置编码缩放参数：用于长文本外推
        # 外推长度 = factor * original_max_position_embeddings
        self.rope_scaling = {
            "beta_fast": 4,
            "beta_slow": 1,
            "factor": 4,
            "original_max_position_embeddings": 2048,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(torch.nn.Module):
    """均方根归一化，比标准 LayerNorm 更高效，Llama 标配。"""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    """预计算 RoPE 的余弦和正弦值。支持 YaRN 插值以扩展上下文长度。"""
    # 基础频率计算,freqs=\frac{1}{10000^{\frac{t}{d}}}, t是dim的第t个分量，d是dim维度                     
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 如果启用了 YaRN 缩放，则根据公式调整频率（用于处理更长的序列
    if rope_scaling is not None:
        # orig_max: 模型原始预训练时的最大位置长度（例如 Llama 是 2048）
        # factor: 扩展倍数（例如 factor=4 意味着想支持 2048 * 4 = 8192 长度）
        # beta_fast / beta_slow: YaRN 用于控制不同维度分支平滑过渡的阈值参数
        orig_max, factor, beta_fast, beta_slow = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 4),
            rope_scaling.get("beta_fast", 4.0), rope_scaling.get("beta_slow", 1.0)
        )
        # 2. 判断当前序列长度是否超过了原始预训练长度
        # 如果超过了，才需要进行频率缩放（插值）
        if end / orig_max > 1.0:
            # 3. 计算修正维度界限 (corr_dim)
            # YaRN 的核心思想：不同频率的维度（dim）感受到的长度变化不同
            # 低频维度对应的波长长，不需要剧烈插值；高频维度波长短，需要插值。
            # 这里寻找第一个波长（2*pi/freq）超过原始最大长度的维度索引。
            corr_dim = next((i for i in range(dim // 2) if 2 * math.pi / freqs[i] > orig_max), dim // 2)
            
            # 4. 计算线性平滑系数 (beta)
            # 在不同维度之间建立一个线性过渡，使得从“不缩放”到“全缩放”的过程平滑。
            # power 从 0 到 1 线性增加
            power = torch.arange(0, dim // 2, device=freqs.device).float() / max(dim // 2 - 1, 1)
            beta = beta_slow + (beta_fast - beta_slow) * power
            
            # 5. 计算 YaRN 的频率缩放系数 (scale)
            # --- YaRN 标准公式核心逻辑 ---
            # 对于波长较短的维度（索引 < corr_dim），应用 YaRN 特有的修正公式：
            # λ = (β * α - β + 1) / (β * α)  其中 α 是 factor
            # 这个公式能在拉伸位置的同时，通过 β 调整来保留高频维度的分辨率。
            # 对于波长极长的维度（索引 >= corr_dim），则直接按 1/factor 进行简单的线性插值缩放。           
            scale = torch.where(torch.arange(dim // 2, device=freqs.device) < corr_dim, (beta * factor - beta + 1) / (beta * factor), 1.0 / factor)
            freqs = freqs * scale

    # 计算旋转矩阵所需的 cos 和 sin 缓存
    t = torch.arange(end, device=freqs.device) # 第t个token位置
    freqs = torch.outer(t, freqs).float() # 每个token有dim个维度，对应的位置embedding要扩展, shape[end, dim//2]
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) # shape[end, dim]
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1) # [-x_n, -x_{n-1}, .., x_{dim//2}, x_0, x_1, ..., x_{dim//2}]

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    ) # expand扩展维度


class Attention(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = args.num_attention_heads # N_h
        self.n_local_kv_heads = self.num_key_value_heads # N_kv
        self.n_rep = self.n_local_heads // self.n_local_kv_heads # 重复倍数（GQA）
        # 计算每个头的维度
        self.head_dim = args.hidden_size // args.num_attention_heads # D_h = H / N_h
        # QKV投影矩阵
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        # 是否使用 PyTorch 内置的 Flash Attention (高效能)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self,
                x: torch.Tensor, # shape: [B, S, H]
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 修改为接收cos和sin
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None):
        # 1. 投影并重塑形状 [Batch, SeqLen, Heads, HeadDim]
        # 2. 应用 RoPE 旋转位置编码
        # 3. 处理 KV Cache (用于自回归推理)
        # 4. 执行注意力计算 (Flash Attention 或 手写 Softmax)
        # 5. 输出投影
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x) # shape: [B, S, N_h * D_h]
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim) # shape:[B, S, N_h, D_h]
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim) # shape: [B, S, N_kv, D_h]
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim) # shape: [B, S, N_kv, D_h]

        cos, sin = position_embeddings # each: [max_pos, D_h]
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # kv_cache实现
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1) # [B, S_past+S, N_kv, D_h]
            xv = torch.cat([past_key_value[1], xv], dim=1) # [B, S_past+S, N_kv, D_h]
        past_kv = (xk, xv) if use_cache else None # past_kv[0]存xk，past_kv[1]存xv

        # 转为 head-first 格式
        xq, xk, xv = (
            xq.transpose(1, 2), # [B, N_h, S, D_h]
            repeat_kv(xk, self.n_rep).transpose(1, 2), # [B, N_kv, S+past, D_h]
            repeat_kv(xv, self.n_rep).transpose(1, 2) # [B, N_kv, S+past, D_h]
        )

        # Flash Attention 或手动实现
        if self.flash and seq_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            attn_mask = (
                None
                if attention_mask is None
                else attention_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_local_heads, seq_len, -1).bool()
            )
            # output shape: [B, N_h, S, D_h]
            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim) # [B, N_h, S, S(+past)]
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)  # scores+mask

            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv # [B, N_h, S, D_h]

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1) # [B, S, H]
        output = self.resid_dropout(self.o_proj(output)) # [B, S, H]
        return output, past_kv


class FeedForward(nn.Module): # MLP模块
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        # 如果用户没指定中间层大小，就按 Llama 标准设为 hidden_size 的 2.66 倍，
        # 并微调这个数值，让它成为 64 的倍数，以便让 显卡跑得最快。
        if config.intermediate_size is None: # FFN中间层维度
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act] # 激活函数

    def forward(self, x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoEGate(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok # 每个token激活的专家数量
        self.n_routed_experts = config.n_routed_experts # 总的专家数量

        self.scoring_func = config.scoring_func # 评分函数默认softmax
        self.alpha = config.aux_loss_alpha # 辅助损失alpha参数
        self.seq_aux = config.seq_aux # 是否在序列级别上计算辅助损失

        self.norm_topk_prob = config.norm_topk_prob # 是否对 Top-K 的概率进行归一化
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5)) # kaiming初始化参数

    def forward(self, hidden_states): 
        """
        MoE 门控机制：决定每个 Token 该由哪些专家处理。
        
        参数:
            hidden_states: 输入张量，形状为 [batch_size, seq_len, hidden_dim]
        返回:
            topk_idx: 每个 token 选中的 top-k 专家的索引 [B*S, top_k]
            topk_weight: 每个选中的专家对应的权重（归一化后） [B*S, top_k]
            aux_loss: 负载均衡辅助损失（标量）
        """
        bsz, seq_len, h = hidden_states.shape  # hidden_states: [B, S, H]
        hidden_states = hidden_states.view(-1, h) # [B*S, H]
        logits = F.linear(hidden_states, self.weight, None) # [B*S, n_routed_experts]
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1) # [B*S, n_routed_experts]
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False) # 在最后一个维度取topk
        # topk_weight: [B*S, top_k], topk_idx: [B*S, top_k]
        # 如果 top_k > 1，通常需要重新归一化，使得选中的专家权重之和为 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # 6. 计算负载均衡辅助损失 (Auxiliary Loss)
        # 目的：防止“富者愈富”，避免所有 token 都流向少数几个专家导致其它专家闲置
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # 将 topk_idx 恢复成 [batch_size, seq_len * top_k] 方便按序列计算
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            # 方案 A: 序列级辅助损失 (Sequence-level auxiliary loss)
            if self.seq_aux:
                # 还原为 [B, S, E]
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                # ce (Count of Experts): 记录每个专家被选中的频率
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                # 将 topk_idx 的位置填入 1，统计每个专家被选中的次数
                # scatter_add_：统计了每个序列中每个专家被选中的次数 $c_{b,i}$
                #.div_(seq_len * aux_topk / E)：这里是一个数学转换。实际的选择频率应该是 $f_{b,i} = \frac{c_{b,i}}{\text{Total Slots}}$
                #其中 Total Slots = $S \times K$）。代码执行的是 $c_{b,i} \div (\frac{S \cdot K}{E}) = \frac{c_{b,i} \cdot E}{S \cdot K} = E \cdot f_{b,i}$。
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts) 
                # 计算均衡损失：专家的平均评分 * 专家被选中的频率
                # 目标是让这两个分布都接近均匀分布
                # scores_for_seq_aux 是 Softmax 后的概率。.mean(dim=1) 对序列维度（dim=1）求平均，
                # 得到该序列中每个专家的平均概率 $P_{b,i} = \frac{1}{S} \sum_{s=1}^S \text{score}_{b,s,i}$。
                # (ce * P_b_mean).sum(dim=1)：对应 $\sum_{i=1}^{E} (E \cdot f_{b,i}) \cdot P_{b,i}$
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            # 方案 B: 全局级辅助损失 (Global-level auxiliary loss)
            else:
                # mask_ce: 将选中的索引转为 one-hot 编码 [Total_Tokens, E]
                # topk_idx_for_aux_loss.view(-1):将整个 Batch 中所有 Token 选中的 Top-K 专家索引展平为一维长向量。
                # F.one_hot：将索引转为 One-hot 编码。
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                # fi: 专家被实际选中的比例
                # 在 Token 维度求平均。这实际上统计了每个专家在整个 Batch 中出现的次数占比。此时 ce 就是 $f_i$
                ce = mask_ce.float().mean(0)
                # Pi: 专家评分的平均值
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts # 对应公式中的 E * f_i
                # 经典的 Load Balancing Loss 公式: sum(Pi * fi)
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    """混合专家前馈网络。"""
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # 创建多个独立的专家 (FeedForward)
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config) # 门控器
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x): # x: [B, S, H]
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x) # topk_idx/weight: [B*S, top_k]
        x = x.view(-1, x.shape[-1]) # [B*S, H]
        flat_topk_idx = topk_idx.view(-1) # 展平为一个维度
        # 训练时：复制输入给每个专家
        if self.training:
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0) # [B*S*top_k, H]
            y = torch.empty_like(x, dtype=torch.float16)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)  # 确保类型一致, 将输入输给expert
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1) # [B*S, H], y.view(*topk_weight.shape, -1):[B*S, top_k, H]
            y = y.view(*orig_shape) # [B, S, H]
        # 推理时：使用高效 moe_infer
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        # 共享专家（始终参与）
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """高效推理模式：通过排序和索引，仅对分配到任务的专家进行计算。"""
        # 核心逻辑：
        # 1. 将 Token 按所属专家 ID 排序
        # 2. 循环每个专家，从输入中挑出属于它的 Token 进行计算 # 目的：防止“富者愈富”，避免所有 token 都流向少数几个专家导致其它专家闲置
        # 3. 使用 scatter_add 将结果写回原始位置        
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0) # cumsum(0)在维度0计算前缀和，这个代码计算每个expert的token结束索引
        token_idxs = idxs // self.config.num_experts_per_tok # 它将“展开后的任务索引”还原回“原始 Token 的编号”。
        # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
        # 接下来9个位置token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx: # 2. 空任务检查：如果起始位置等于结束位置，说明该专家没有分配到任何 token
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx] # 找到这个专家处理的tokens的id
            expert_tokens = x[exp_token_idx] # 找到这个专家对应的tokens
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # 7. 加权：将专家的输出乘以门控权重 (Gating Weight)
            # idxs[start_idx:end_idx] 找回了排序前对应的权重位置
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 8. 结果写回 (核心步骤)：
            # scatter_add_ 将算好的结果根据原始 ID (exp_token_idx) 累加回缓存中
            # 因为一个 Token 可能被多个专家处理（Top-K），所以用累加（add）的方式合并结果
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config) # 选择Dense FFN还是 MoE

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None): 
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        ) # 输入attention_block前先经过一层RMSNorm
        hidden_states += residual # 残差链接
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states)) # 经过MLP block前经过一层RMSNorm
        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size) # 分词经过embedding层做嵌入
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)]) # Miniblock层数
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) # 从MiniBlock层出来之后经过RMSNorm

        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, rope_base=config.rope_theta,
                                                    rope_scaling=config.rope_scaling) # 产生位置编码，推理时用YaRN外推
        # 预计算并注册 RoPE 缓存，避免重复计算
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None, # [B, S]
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        batch_size, seq_length = input_ids.shape # tokenizer处理完句子后产生的token的id
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.dropout(self.embed_tokens(input_ids)) # [B, S, H]

        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length], # [S, D_h]
            self.freqs_sin[start_pos:start_pos + seq_length]  # [S, D_h]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            ) # hidden_states: [B, S, H]
            presents.append(present)

        hidden_states = self.norm(hidden_states) # [B, S, H]

        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp, MOEFeedForward)
        ) # 如果是MoE层的FFN，还有aux loss

        return hidden_states, presents, aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    """用于因果语言建模的包装类（输出 Logits）。"""
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight
        self.OUT = CausalLMOutputWithPast()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None, # [B, S]
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        h, past_kvs, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        ) # h: [B, S, H]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(h[:, slice_indices, :]) # [B, S', V]
        self.OUT.__setitem__('last_hidden_state', h)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT

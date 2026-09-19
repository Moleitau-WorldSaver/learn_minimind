import math, torch, torch.nn.functional as F
from typing import Any
from torch import nn
from transformers.activations import ACT2FN
from transformers import Optional, PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)

# 继承nn.Module类，定义模型的结构和前向传播逻辑
class RMSNorm(nn.Module):
# __init__初始化
    def __init__(self, dim : int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
# _norm
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # 计算均方根
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
# forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 归一化
        output = self._norm(x.float())
        # 缩放
        output = output * self.weight
        return output.type_as(x)

def precompute_freqs(
        dim: int,
        end: int = int(32 * 1024),
        rope_base: float = 1e6,
        rope_scaling: Optional[dict] = None,    
) :
    #1 初始化: 频率freqs 和 注意力温度补偿系数attn_factor
    freqs, attn_factor = (
        1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)),
        1.0, 
    )

    if rope_scaling is not None:
        #2 从配置字典中提取超参数
        # 预训练原始最大长度, 拓展倍数s, 高频边界, 低频边界, 注意力温度补偿系数   
        original_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0),
        )

        #只有推断长度大于原始长度, 才应用缩放
        scale = end > original_max
        if scale > 1.0:
            # 3. 使用前文推导的公式，输入参数 β 算出到维度索引 i 的映射函数
            inv_dim = lambda b: (
                dim / (2 * math.log(rope_base)) * (math.log(original_max / (2 * math.pi * b)))
            )  

            # 4. 计算高频和低频的切分点
            #low 算出低频区的下标
            #high 算出高频区的下标
            low, high = (
                max(math.floor(inv_dim(beta_fast)), 0),
                min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            )

            # 5. 计算低频区和高频区的缩放因子
            ramp = torch.clamp(
                (torch.arange(dim // 2, device = freqs.device).float() - low)
                / max(high - low, 0.001),
                0,
                1,
            )
            #freqs混合, 执行分32组快满指针
            freqs = freqs * (1 - ramp + ramp / factor)

        # 7. 根据目标长度 end，生成位置索引向量 t
        t = torch.arange(end, device=freqs.device)

        # 8. 计算外积：将位置 t 与处理好的频率 freqs 相乘，得到每个位置的旋转角度 θ
        #"每 token 转多少弧度" × "第几个 token" = "一共转了多少弧度"
        freqs = torch.outer(t, freqs).float()

        # 9. 计算 Cos 和 Sin，并应用注意力补偿系数 (attn_factor)
        #张量就是存其对应的 cos 和 sin 值, 维度为 [end, dim // 2],之后cat变成 [end, dim]
        freqs_cos, freqs_sin = (
            torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor,
            torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor,
        )
        #返回结果 
        return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim = 1):
    def rotate_half(x):
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )
    #到这里, 注意 minimind的旋转位置编码是是前半部分全是x, 后半部分全是y
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed

def repeat_kv(x:torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # 高效的重复实现：
    # 1. x[:, :, :, None, :]: 在第4维插入新维度 -> [bs, slen, num_kv_heads, 1, head_dim]
    # 2. .expand(...): 扩展第4维到n_rep -> [bs, slen, num_kv_heads, n_rep, head_dim]
    # 3. .reshape(...): 合并第3、4维 -> [bs, slen, num_kv_heads * n_rep, head_dim]
    return (
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        # 处理GQA
        # 如果num_key_value_heads为None, 则使用num_attention_heads, 也就是query 头的个数
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads

        # 确保能Q头能整除K/V头, 否则报错
        assert config.num_attention_heads % self.num_key_value_heads == 0

        #注意力头配置
        self.n_local_heads = config.num_attention_heads # Q头数
        self.n_local_kv_heads = self.num_key_value_heads # kv头数
        self.n_rep = config.num_attention_heads // self.num_key_value_heads # 复制份数
        self.head_dim = config.head_dim # 每个头维度

        #使用因果掩码
        self.is_causal = True
        # 定义线性投影层 (无偏置，节省参数)
        # nn.Linear(in_features, out_features, bias=False)
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)

        #对每个头的q 和 k 向量做RMSNorm归一化
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.attn_dropout = nn.Dropout(config.dropout) #注意力权重
        self.resid_dropout = nn.Dropout(config.dropout) #残差连接
        self.dropout = config.dropout #保存dropout参数

        #检查是否支持flash attention, 问有没有torch.nn.functional 这个函数, 如果有, 并且配置中flash_attn为True, 则启用flash attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(
            self,
            x: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
            use_cache = False,
            attention_mask: Optional[torch.Tensor] = None):
        # x: [batch_size, seq_len, hidden]
        #[批大小(一次并行处理多少序列), 序列长度, 隐藏维]
        bsz, seq_len, _ = x.shape

        #初始化
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        #投影Q, K, V  
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # position_embeddings是预计算的(cos, sin)，按序列位置切片并应用RoPE
        cos, sin = position_embeddings

        # 对 q/k 施加旋转位置编码：各自按【绝对位置】旋转，
        # 效果是 q·k 只依赖两个 token 的【位置差】
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        #-------------- kv_cache处理 -----------------
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        # 如果后续需要缓存,返回更新后的新KV cache
        # 通常是推理时use_cache = 1, 训练时等于 0 
        past_kv = (xk,xv) if use_cache else None

        # ------------- GQA: 对KV重复以匹配Q头 ----------
        # transpose到形状 [bsz, n_heads, seq_len, head_dim] 以便矩阵乘法
        xq = xq.transpose(1, 2)

        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep) .transpose(1, 2)

        # -------------------- Attention计算 --------------------
        # 优先使用PyTorch 2.0+的scaled_dot_product_attention（Flash Attention实现）
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and(attention_mask is None or torch.all(attention_mask == 1)) :
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                dropout_p = self.dropout if self.training else 0.0,
                is_causal = self.is_causal
            )
        else:
            # 标准实现：scores = Q @ K^T / sqrt(d)
            # 点积会随着维度膨胀, 除以根号下head_dim正好能把标准差压回1
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

            #三角掩码
            if self.is_causal: 
                #先二维置为 -inf, 然后掩住上三角
                scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)

            # 如果有attention_mask(0/1)，将其扩展后转为 -1e9 的加性mask（掩掉pad位置）
            # padding 掩码, padding 是"无效位置"这些位置没有真实语义,但模型仍会给它们 embedding 并参与计算
            # 所以需要 attention_mask 把它们的注意力权重压成 0
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask
            #sortmax得到注意力权重
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        # 恢复形状并做输出投影 + 标记残差流支路
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)  # [bsz, seq_len, hidden]
        output = self.resid_dropout(self.o_proj(output)) # type: ignore
        return output, past_kv
            
class FeedForward(nn.Module):
    def __init__(self, config:MiniMindConfig, intermediate_size: int = 0):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias = False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias = False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias = False)
        # ACT2FN是transformers里激活函数的映射表，支持'silu','gelu'等
        self.act_fn = ACT2FN[config.hidden_act]

    def forword(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class MOEFeedForward(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

class MinimindBlock(nn.Module):

    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layerNorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache = False, attention_mask = None):
        #保存初始状态
        residual = hidden_states
        #流程顺序: LayerNorm -> Attention -> 残差相加 -> LayerNorm -> FFN -> 残差相加
        #先从Attention, 返回hidden_states和present_key_value（用于cache）
        hidden_states, present_key_value = self.self_attn(
            self.input_layerNorm(hidden_states),  # pre-norm
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask
        )
        # 加上 初始状态完成残差
        hidden_states = hidden_states + residual
        #每次进入一个层都进行 pre-norm
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))

        return hidden_states, present_key_value






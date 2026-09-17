import math

from transformers import Optional, PretrainedConfig


class MyMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int | None = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
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
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

import torch
import torch.nn as nn

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
            rope_scaling.get("beta_fast", 32),
            rope_scaling.get("beta_slow", 1),
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


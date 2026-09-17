import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import torch

dim, end, factor = 64, 3, 16
rope_base, orig_max, attn_factor = 1e6, 2048, 1.277

# 复现:先算 freqs(角度), 再 cos -> cat -> *attn_factor
inv_freq = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
t = torch.arange(end)
freqs = torch.outer(t, inv_freq)            # 这里 freqs 是"角度矩阵"

step1 = torch.cos(freqs)
step2 = torch.cat([step1, step1], dim=-1)
result = step2 * attn_factor

print("逐步看每一层的张量:")
for name, t_ in [("freqs(角度)", freqs), ("torch.cos(freqs)", step1),
                 ("cat([c,c], dim=-1)", step2), ("... * attn_factor", result)]:
    print(f"  {name:<22} shape={str(tuple(t_.shape)):<12} dtype={str(t_.dtype):<16} "
          f"min={t_.min():.4f} max={t_.max():.4f}")

print("\n--- 关键结论 ---")
print(f"  最终张量: shape={tuple(result.shape)}   dtype={result.dtype}   元素个数={result.numel()}")
print(f"  attn_factor={attn_factor} 是标量 -> 只缩放数值, 不改变 shape/dtype")

print("\n--- 取值范围的变化(重要) ---")
print(f"  cos 原本在 [-1, 1]: 实测 [{step1.min():.4f}, {step1.max():.4f}]")
print(f"  乘 attn_factor 后:  [{result.min():.4f}, {result.max():.4f}]  <- 越界了!")
print(f"  原因: cos 的最大值 1.0 x {attn_factor} = {attn_factor}")

print("\n--- dtype 敏感性: 整型 attn_factor 会不会改变 dtype? ---")
c = torch.cos(freqs)
print(f"  float 张量 * int 标量   -> dtype = {(c * 2).dtype}")
print(f"  float 张量 * float 标量 -> dtype = {(c * 1.5).dtype}")

print("\n--- 逐元素对照(位置1, 前6个) ---")
print("  cos      =", [round(v, 4) for v in step1[1][:6].tolist()])
print("  cat 后   =", [round(v, 4) for v in step2[1][:12].tolist()])
print("  * factor =", [round(v, 4) for v in result[1][:12].tolist()])

print("\n--- 前后半是否完全一致 ---")
half = result.shape[-1] // 2
print("  前半 == 后半 ?", bool(torch.equal(result[..., :half], result[..., half:])))

print("\n--- 这个张量拿来做什么 ---")
q = torch.randn(1, 1, end, dim)
print(f"  q.shape = {tuple(q.shape)}")
print(f"  result.shape = {tuple(result.shape)}  <- 需要 unsqueeze 成 (1,1,end,dim) 才能广播")
r = result.unsqueeze(0).unsqueeze(0)
print(f"  unsqueeze 后 = {tuple(r.shape)}")
print(f"  q * r -> shape {tuple((q * r).shape)}  ✅")

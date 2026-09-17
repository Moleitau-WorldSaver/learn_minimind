import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import torch

dim, end, rope_base, factor = 64, 3, 1e6, 16
attn_factor = 0.1 * torch.log(torch.tensor(float(factor))).item() + 1.0

# ---- 阶段1: 频率向量 (dim//2,) ----
inv_freq = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))

# ---- 阶段2: 外积 -> 角度矩阵 (end, dim//2) ----
t = torch.arange(end)
angles = torch.outer(t, inv_freq).float()

# ---- 阶段3: cos / sin (end, dim//2) ----
cos_half = torch.cos(angles)
sin_half = torch.sin(angles)

# ---- 阶段4: cat 复制 + 温度 -> (end, dim) ----
freqs_cos = torch.cat([cos_half, cos_half], dim=-1) * attn_factor
freqs_sin = torch.cat([sin_half, sin_half], dim=-1) * attn_factor

print("=" * 78)
print(f"{'张量':<14}{'shape':<16}{'dtype':<18}{'值域':<24}物理含义")
print("=" * 78)
rows = [
    ("inv_freq",  inv_freq,      "频率(角速度)"),
    ("angles",    angles,        "角度(弧度) = 位置x频率"),
    ("cos_half",  cos_half,      "cos(角度)"),
    ("sin_half",  sin_half,      "sin(角度)"),
    ("freqs_cos", freqs_cos,     "cos 复制一份 + 乘温度"),
    ("freqs_sin", freqs_sin,     "sin 复制一份 + 乘温度"),
]
for name, t_, meaning in rows:
    print(f"{name:<14}{str(tuple(t_.shape)):<16}{str(t_.dtype):<18}"
          f"[{t_.min():.4f}, {t_.max():.4f}]{'':<6}{meaning}")

print("\n" + "=" * 78)
print("关键区别一: angles 是'角度', freqs_cos/sin 是'角度的三角函数值'")
print("=" * 78)
print(f"  angles[1][:4]    = {[round(v,4) for v in angles[1][:4].tolist()]}   <- 弧度, 可以无限大")
print(f"  cos_half[1][:4]  = {[round(v,4) for v in cos_half[1][:4].tolist()]}   <- cos 值, 恒在 [-1,1]")
print(f"  sin_half[1][:4]  = {[round(v,4) for v in sin_half[1][:4].tolist()]}   <- sin 值")
print(f"  验证: cos^2+sin^2=1 ? {bool(torch.allclose(cos_half**2+sin_half**2, torch.ones_like(cos_half)))}")
print(f"  验证: angles[1][0] = {angles[1][0]:.4f}, cos = {cos_half[1][0]:.4f}  (确实是 cos)")

print("\n" + "=" * 78)
print("关键区别二: shape 从 dim//2 变到 dim")
print("=" * 78)
print(f"  angles / cos_half / sin_half : {tuple(angles.shape)}   <- 每2维共享1个角度, 所以是 dim//2")
print(f"  freqs_cos / freqs_sin        : {tuple(freqs_cos.shape)}   <- 复制一份凑齐 head_dim")
print(f"  前后半是否相同 ? {bool(torch.equal(freqs_cos[..., :dim//2], freqs_cos[..., dim//2:]))}")

print("\n" + "=" * 78)
print("关键区别三: 只有 freqs_cos/sin 能直接乘到 q 上")
print("=" * 78)
q = torch.randn(1, 1, end, dim)
print(f"  q.shape = {tuple(q.shape)}")
for name, t_ in [("angles", angles), ("cos_half", cos_half), ("freqs_cos", freqs_cos)]:
    try:
        r = q * t_
        print(f"  q * {name:<10} -> OK    shape {tuple(r.shape)}")
    except RuntimeError as e:
        print(f"  q * {name:<10} -> 失败  {str(e)[:58]}")

print("\n" + "=" * 78)
print("角度到底能有多大(说明为什么必须有 cos/sin)")
print("=" * 78)
big_end = 32768
big_angles = torch.outer(torch.arange(big_end), inv_freq)
print(f"  end={big_end} 时 angles 最大值 = {big_angles.max():.1f} 弧度")
print(f"  折合圈数 = {big_angles.max()/(2*torch.pi):.1f} 圈")
print("  -> 角度是'转了多少弧度'的累积量, 数值很大")
print("  -> 必须先取 cos/sin 变成 [-1,1] 的系数, 才能和 q 相乘")
print("  -> 这就是名字 freqs_cos / freqs_sin 的由来: 它们才是'可用的系数'")

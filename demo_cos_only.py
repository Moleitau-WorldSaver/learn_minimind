import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import torch

dim, end = 64, 3
rope_base = 1e6
freqs = torch.outer(
    torch.arange(end),
    1.0 / (rope_base ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim)),
)

y = torch.cos(freqs)

print("输入 freqs:")
print(f"  type   = {type(freqs).__module__}.{type(freqs).__name__}")
print(f"  shape  = {tuple(freqs.shape)}")
print(f"  dtype  = {freqs.dtype}")
print(f"  device = {freqs.device}")
print(f"  numel  = {freqs.numel()}")

print("\n输出 torch.cos(freqs):")
print(f"  type   = {type(y).__module__}.{type(y).__name__}")
print(f"  shape  = {tuple(y.shape)}")
print(f"  dtype  = {y.dtype}")
print(f"  device = {y.device}")
print(f"  numel  = {y.numel()}")

print("\n--- 它到底装了什么东西 ---")
print("  就是'每个角度各自的余弦值', 逐元素 cos, 一一对应:")
print(f"    freqs[1][:4] = {[round(v,4) for v in freqs[1][:4].tolist()]}")
print(f"    cos  [1][:4] = {[round(v,4) for v in y[1][:4].tolist()]}")

print("\n--- 数学性质 ---")
print(f"  值域 [-1, 1]: 实测 min={y.min():.4f}  max={y.max():.4f}")
print(f"  行0(t=0,角度全0) 整行等于1? {bool((y[0]==1).all())}")
print(f"  和 sin 的关系 cos^2+sin^2=1 ? {bool(torch.allclose(y**2+torch.sin(freqs)**2, torch.ones_like(y)))}")

print("\n--- 它是'实数'张量 ---")
print(f"  y.is_complex()      = {y.is_complex()}")
print(f"  y.dtype             = {y.dtype}")
print(f"  对比 torch.polar    = {torch.polar(torch.ones_like(freqs), freqs).dtype}  <- 那个才是复数")

print("\n--- 一个关键推论: 它是实数, 所以不能做复数乘法 ---")
print("  只靠 cos 无法完成旋转(旋转需要 cos 和 sin 配合)")
print("  所以 cat([cos, cos]) 之后, 还要 sin 那边也来一份, 才能用向量化公式")

print("\n--- 它在内存里长什么样(前两行) ---")
print("  ", y[:2, :8].tolist())

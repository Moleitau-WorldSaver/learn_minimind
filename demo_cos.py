import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import torch

# 用真实参数:head_dim=64, end=4(缩小便于观察)
dim, end, rope_base = 64, 4, 1e6

freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
t = torch.arange(end)
angles = torch.outer(t, freqs)          # YaRN 修正后的角度矩阵

print("输入 angles:")
print(f"  shape = {tuple(angles.shape)}   dtype = {angles.dtype}   元素个数 = {angles.numel()}")
print(f"  行0(位置0,前5个) = {[round(v,6) for v in angles[0][:5].tolist()]}")

cos_t = torch.cos(angles)
print("\n输出 torch.cos(angles):")
print(f"  shape = {tuple(cos_t.shape)}   dtype = {cos_t.dtype}   元素个数 = {cos_t.numel()}")
print(f"  行0(位置0,前5个) = {[round(v,6) for v in cos_t[0][:5].tolist()]}")
print(f"  行0(位置0,后3个) = {[round(v,6) for v in cos_t[0][-3:].tolist()]}")

print("\n--- 核心事实 ---")
print("  1) shape 完全不变:", tuple(angles.shape), "->", tuple(cos_t.shape))
print("  2) dtype 完全不变:", angles.dtype, "->", cos_t.dtype)
print("  3) 是逐元素运算, 不是归约(元素个数不变:", angles.numel(), "->", cos_t.numel(), ")")
print("  4) 返回实数张量, 不是复数")
print(f"  5) 取值范围 [-1, 1], 实测 min={cos_t.min():.4f} max={cos_t.max():.4f}")

print("\n--- 位置 0 的特殊性 ---")
print("  angles 第0行全是 0(因为 t=0)  ->  cos(0)=1")
print(f"  cos 第0行全部 = 1 ? {bool((cos_t[0] == 1).all())}")

print("\n--- 和 torch.sin 的关系 ---")
sin_t = torch.sin(angles)
print("  sin 第0行全 = 0 ?", bool((sin_t[0] == 0).all()), " (因为 sin(0)=0)")
print("  cos^2 + sin^2 == 1 ?", bool(torch.allclose(cos_t ** 2 + sin_t ** 2, torch.ones_like(cos_t))))

print("\n--- 对比:polar 路线(另一种实现) ---")
cis = torch.polar(torch.ones_like(angles), angles)
print(f"  torch.polar(ones, angles): shape = {tuple(cis.shape)}   dtype = {cis.dtype}")
print(f"  复数元素示例: {cis[1][:3].tolist()}")
print("  -> polar 直接给出 e^(i*theta), 一个复数同时含 cos 和 sin")
print("  -> cos/sin 路线要存两个实数张量, polar 只存一个复数张量(内存减半, 但需复数运算支持)")

print("\n--- 你项目里的实际用法(cos/sin 路线) ---")
freqs_cos = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1)
print(f"  torch.cos(angles)          -> {tuple(torch.cos(angles).shape)}   (dim//2 = {dim//2})")
print(f"  cat([cos, cos], dim=-1)    -> {tuple(freqs_cos.shape)}   (凑齐 head_dim = {dim})")
print("  然后就和 q/k 逐元素相乘")

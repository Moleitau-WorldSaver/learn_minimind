import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import math
import torch

dim, rope_base, end = 8, 1e6, 4        # head_dim=8 -> 4 个维度对, 便于完整打印
inv_freq = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
t = torch.arange(end)

print("输入两样东西:")
print(f"  t (位置)     = {t.tolist()}                      shape {tuple(t.shape)}")
print(f"  inv_freq     = {[round(v,4) for v in inv_freq.tolist()]}   shape {tuple(inv_freq.shape)}")

angles = torch.outer(t, inv_freq).float()
print(f"\n输出 angles = torch.outer(t, inv_freq)   shape {tuple(angles.shape)}")
print("  (行=位置 t, 列=维度对 i, 每格装一个弧度值)")
print()
print(f"  {'':<10}" + "".join(f"{'i='+str(i):>14}" for i in range(dim // 2)))
for ti in range(end):
    print(f"  {'t='+str(ti):<10}" + "".join(f"{angles[ti][i].item():>14.4f}" for i in range(dim // 2)))

print("\n" + "=" * 74)
print("逐格验证: 每个格子 = t * inv_freq[i]")
print("=" * 74)
for ti in [0, 1, 2, 3]:
    for i in [0, 2]:
        expect = t[ti].item() * inv_freq[i].item()
        got = angles[ti][i].item()
        print(f"  angles[{ti}][{i}] = t({t[ti].item()}) x freq({inv_freq[i].item():.4f}) "
              f"= {expect:.4f}   实测 {got:.4f}   一致 {abs(expect-got)<1e-6}")

print("\n" + "=" * 74)
print("torch.cos 逐格替换: 弧度 -> 余弦值")
print("=" * 74)
c = torch.cos(angles)
print("  角度(弧度) 表:")
for ti in range(end):
    print("    " + "".join(f"{angles[ti][i].item():>11.4f}" for i in range(dim // 2)))
print("  cos 值 表:")
for ti in range(end):
    print("    " + "".join(f"{c[ti][i].item():>11.4f}" for i in range(dim // 2)))

print("\n  逐格对照(位置1):")
for i in range(dim // 2):
    a = angles[1][i].item()
    cc = c[1][i].item()
    print(f"    i={i}: 弧度 {a:>8.4f}  ->  cos = {cc:>8.4f}   "
          f"(用 math.cos 验证: {math.cos(a):>8.4f})")

print("\n" + "=" * 74)
print("形状完全不变, 只是每个格子被换成了它的 cos")
print("=" * 74)
print(f"  angles.shape = {tuple(angles.shape)}")
print(f"  cos.shape    = {tuple(c.shape)}     <- 一样")

print("\n" + "=" * 74)
print("cos 和 sin 是'同一批弧度的两种输出'")
print("=" * 74)
s = torch.sin(angles)
print(f"  同一格 angles[3][1] = {angles[3][1].item():.4f} 弧度")
print(f"    cos = {c[3][1].item():.4f}")
print(f"    sin = {s[3][1].item():.4f}")
print(f"    cos^2+sin^2 = {(c[3][1]**2+s[3][1]**2).item():.6f}  (=1)")

print("\n" + "=" * 74)
print("但注意: '字典个数' 这个说法")
print("=" * 74)
print(f"  第二维的长度 = dim//2 = {dim//2}  (不是 head_dim {dim})")
print(f"  因为 RoPE 规定: 每 2 个维度共享 1 个弧度")
print(f"  所以 rows x cols = {end} x {dim//2} = {angles.numel()} 个弧度值")
print("  这就是为什么后面要 cat([c,c]) 复制一份凑齐 head_dim")

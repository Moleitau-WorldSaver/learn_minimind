import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import torch

rope_base, dim, end = 1e6, 64, 8  # end 取小一点便于观察

# 步骤 1:freqs = 每个维度对的角频率 [dim//2]
freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
print(f"freqs: shape={tuple(freqs.shape)}  dtype={freqs.dtype}")
print("  freqs[:6] =", [f"{v:.7f}" for v in freqs[:6].tolist()])
print("  含义: 32 个维度对,各自的角频率(第0对转最快,第31对转最慢)\n")

# 步骤 2:t = 位置索引 [end]
t = torch.arange(end, device=freqs.device)
print(f"t: shape={tuple(t.shape)}  dtype={t.dtype}")
print("  t =", t.tolist())
print("  含义: 仅仅是 token 的序号。此处还没有任何'角度'信息\n")

# 步骤 3:外积 -> 每个位置 × 每个频率 = 旋转角矩阵 [end, dim//2]
angles = torch.outer(t, freqs)
print(f"outer(t, freqs): shape={tuple(angles.shape)}")
print("  第 0 行(位置0):", [f"{v:.6f}" for v in angles[0][:5].tolist()], "...")
print("  第 1 行(位置1):", [f"{v:.6f}" for v in angles[1][:5].tolist()], "...")
print("  第 5 行(位置5):", [f"{v:.6f}" for v in angles[5][:5].tolist()], "...")
print("\n  关键: 第 t 行 = 第 1 行 × t  (角度随位置线性增长)")
print("    angles[5] == angles[1]*5 ?", torch.allclose(angles[5], angles[1] * 5))

# 步骤 4:转成复数旋转因子
freqs_cis = torch.polar(torch.ones_like(angles), angles)
print(f"\ntorch.polar(1, angles): shape={tuple(freqs_cis.shape)}  dtype={freqs_cis.dtype}")
print("  位置0 的前3个旋转因子:", [f"{c:.4f}" for c in freqs_cis[0][:3].tolist()], " <- 全是 1+0j,即'不旋转'")
print("  位置1 的前3个旋转因子:", [f"{c:.4f}" for c in freqs_cis[1][:3].tolist()])
print("  位置2 的前3个旋转因子:", [f"{c:.4f}" for c in freqs_cis[2][:3].tolist()])

print("\n--- 结论 ---")
print("t 本身 = 位置序号(无角度信息)")
print("angles = t 广播乘 freqs -> 每个位置每个维度的旋转角 [end, dim//2]")
print("freqs_cis = e^(i*angles) -> 可直接乘到 q/k 上的复数旋转因子")
print("第 t 行就是'位置 t 专用的旋转表',取 freqs_cis[pos] 即得该位置该转多少度")

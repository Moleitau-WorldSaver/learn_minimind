import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import torch

print("=" * 78)
print("换几组参数, 看 angles 的第二维到底是什么")
print("=" * 78)
print(f"  {'dim':<8}{'dim//2':<10}{'end':<10}{'t.shape':<14}{'freqs.shape':<16}{'angles.shape'}")
print("  " + "-" * 70)
for dim in [8, 32, 64, 128]:
    for end in [4, 16]:
        inv = 1.0 / (1e6 ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
        t = torch.arange(end)
        ang = torch.outer(t, inv).float()
        print(f"  {dim:<8}{dim//2:<10}{end:<10}{str(tuple(t.shape)):<14}"
              f"{str(tuple(inv.shape)):<16}{tuple(ang.shape)}")

print("\n" + "=" * 78)
print("结论: angles.shape = [end, dim//2]  —— 第二维是 dim 的一半")
print("=" * 78)

dim, end, factor = 64, 8, 16
attn_factor = 0.1 * torch.log(torch.tensor(float(factor))).item() + 1.0
inv = 1.0 / (1e6 ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
t = torch.arange(end)

steps = [
    ("t",                     t),
    ("inv_freq",              inv),
    ("angles = outer(t,f)",   torch.outer(t, inv).float()),
    ("cos(angles)",           torch.cos(torch.outer(t, inv).float())),
    ("cat([cos,cos],-1)",     torch.cat([torch.cos(torch.outer(t, inv).float())] * 2, dim=-1)),
    ("freqs_cos (+温度)",      torch.cat([torch.cos(torch.outer(t, inv).float())] * 2, dim=-1) * attn_factor),
]
print(f"  {'张量':<26}{'shape':<16}{'第二维是谁'}")
print("  " + "-" * 66)
for name, s in steps:
    second = "—" if s.dim() < 2 else (f"dim//2 = {dim//2}" if s.shape[-1] == dim // 2 else f"dim = {dim}")
    print(f"  {name:<26}{str(tuple(s.shape)):<16}{second}")

print("\n" + "=" * 78)
print("维度翻倍发生在哪一步")
print("=" * 78)
print(f"  angles              -> {tuple(torch.outer(t, inv).shape)}     (end, dim//2)")
print(f"  cos(angles)         -> {tuple(torch.cos(torch.outer(t, inv)).shape)}     (end, dim//2)   形状不变")
print(f"  cat([c,c])          -> {tuple(torch.cat([torch.cos(torch.outer(t,inv))]*2,-1).shape)}     (end, dim)      ← 在这里翻倍!")
print(f"  * attn_factor       -> 形状不变")

print("\n" + "=" * 78)
print("为什么 angles 只有 dim//2 列: 每 2 个维度共享 1 个弧度")
print("=" * 78)
print("  q 的 64 维被看成 32 对:")
print("    (q0, q32) (q1, q33) (q2, q34) ... (q31, q63)")
print("    每对共享同一个弧度 -> 只需要 32 个弧度 -> 就是 dim//2")
print()
print("  32 个弧度 = angles 的列数")
print("  64 个系数 = freqs_cos 的列数(复制后)")

print("\n" + "=" * 78)
print("能直接乘 q 的只有最后那个")
print("=" * 78)
q = torch.randn(2, 4, end, dim)
print(f"  q.shape = {tuple(q.shape)}")
for name, s in [("angles", torch.outer(t, inv)), ("freqs_cos", torch.cat([torch.cos(torch.outer(t,inv))]*2,-1))]:
    try:
        r = q * s
        print(f"  q * {name:<10} -> OK   {tuple(r.shape)}")
    except RuntimeError as e:
        print(f"  q * {name:<10} -> 失败 第二维 {s.shape[-1]} vs {dim}: {str(e)[:40]}")

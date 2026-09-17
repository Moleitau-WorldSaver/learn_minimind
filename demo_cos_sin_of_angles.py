import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import torch

dim, end, rope_base, factor = 64, 3, 1e6, 16
attn_factor = 0.1 * torch.log(torch.tensor(float(factor))).item() + 1.0

inv_freq = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
angles = torch.outer(torch.arange(end), inv_freq).float()   # 角度矩阵 (end, dim//2)

freqs_cos = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1) * attn_factor
freqs_sin = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1) * attn_factor

print("角度示例 (位置1, 前4个维度对):")
print("  angles[1][:4] =", [round(v, 4) for v in angles[1][:4].tolist()])
print()
print("对每个角度取 cos / sin:")
print("  cos           =", [round(v, 4) for v in torch.cos(angles[1][:4]).tolist()])
print("  sin           =", [round(v, 4) for v in torch.sin(angles[1][:4]).tolist()])

print("\n" + "=" * 66)
print("验证 1: freqs_cos 就是这些 cos 值(再乘 attn_factor)")
print("=" * 66)
manual_cos = torch.cos(angles[1][:4]) * attn_factor
manual_sin = torch.sin(angles[1][:4]) * attn_factor
print(f"  手算 cos(theta)*{attn_factor:.4f} = {[round(v,4) for v in manual_cos.tolist()]}")
print(f"  freqs_cos[1][:4]            = {[round(v,4) for v in freqs_cos[1][:4].tolist()]}")
print(f"  一致 ? {torch.allclose(manual_cos, freqs_cos[1][:4], atol=1e-6)}")
print()
print(f"  手算 sin(theta)*{attn_factor:.4f} = {[round(v,4) for v in manual_sin.tolist()]}")
print(f"  freqs_sin[1][:4]            = {[round(v,4) for v in freqs_sin[1][:4].tolist()]}")
print(f"  一致 ? {torch.allclose(manual_sin, freqs_sin[1][:4], atol=1e-6)}")

print("\n" + "=" * 66)
print("验证 2: 三角恒等式(证明它们确实来自同一个角度)")
print("=" * 66)
c = freqs_cos / attn_factor      # 去掉温度系数, 还原纯 cos
s = freqs_sin / attn_factor
print(f"  (cos/af)^2 + (sin/af)^2 = 1 ? {bool(torch.allclose(c**2 + s**2, torch.ones_like(c)))}")

print("\n" + "=" * 66)
print("验证 3: 用 freqs_cos/sin 反推角度(证明它们'装的就是角度信息')")
print("=" * 66)
theta_rec = torch.atan2(s, c)    # 两参数 arctan, 能唯一还原角度
print(f"  反推角度 atan2(sin, cos) = {[round(v,4) for v in theta_rec[1][:4].tolist()]}")
print(f"  原始角度 angles[1][:4]   = {[round(v,4) for v in angles[1][:4].tolist()]}")
print(f"  一致 ? {torch.allclose(theta_rec[1][:4], angles[1][:4], atol=1e-5)}")
print("  -> 说明 cos/sin 这一对确实完整携带了角度信息")

print("\n" + "=" * 66)
print("验证 4: sin 在位置0 全为 0, cos 全为 attn_factor")
print("=" * 66)
print(f"  angles[0] 全是 0 ? {bool((angles[0]==0).all())}")
print(f"  freqs_cos[0] 全是 attn_factor({attn_factor:.4f}) ? {bool(torch.allclose(freqs_cos[0], torch.full_like(freqs_cos[0], attn_factor)))}")
print(f"  freqs_sin[0] 全是 0 ? {bool((freqs_sin[0]==0).all())}")
print("  因为 cos(0)=1, sin(0)=0")

print("\n" + "=" * 66)
print("但要注意: 中间还有两步, 不只是 cos/sin")
print("=" * 66)
print(f"  torch.cos(angles)                 -> {tuple(torch.cos(angles).shape)}   (end, dim//2)")
print(f"  经过 cat 复制一份                  -> {tuple(freqs_cos.shape)}   (end, dim)")
print(f"  再乘 attn_factor={attn_factor:.4f}          -> 值域从 [-1,1] 变成 [{freqs_cos.min():.4f}, {freqs_cos.max():.4f}]")
print()
print("  所以精确表述: freqs_cos = cat([cos(angles), cos(angles)], -1) * attn_factor")
print("                '对角度取 cos' 只是中间那一步, 最终还要复制 + 调温度")

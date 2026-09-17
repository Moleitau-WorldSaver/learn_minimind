import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import math
import torch

print("=" * 70)
print("1) 直接用 60 会得到什么")
print("=" * 70)
print(f"  torch.cos(torch.tensor(60.0)) = {torch.cos(torch.tensor(60.0))}")
print(f"  含义: 60 弧度(不是 60 度!) = {60/(2*math.pi):.4f} 圈")
print(f"  cos(60 rad) = {math.cos(60):.6f}  <- 完全不同的数")

print("\n" + "=" * 70)
print("2) 正确做法: 先转弧度")
print("=" * 70)
deg60_rad = 60 * math.pi / 180
print(f"  60 度 = 60 * pi / 180 = {deg60_rad:.6f} 弧度")
print(f"  math.pi/3             = {math.pi/3:.6f}   (两者相等: {abs(deg60_rad - math.pi/3) < 1e-12})")
r = torch.cos(torch.tensor(deg60_rad))
print(f"  torch.cos(torch.tensor({deg60_rad:.6f})) = {r}")
print(f"  float(r) = {float(r):.10f}")
print(f"  等于 0.5 ? {abs(float(r) - 0.5) < 1e-6}   (浮点误差 {abs(float(r)-0.5):.2e})")

print("\n" + "=" * 70)
print("3) 返回的是 Tensor, 不是 float")
print("=" * 70)
print(f"  type(r)        = {type(r)}")
print(f"  r.shape        = {tuple(r.shape)}    <- 标量张量, 0 维")
print(f"  r.dim()        = {r.dim()}")
print(f"  r.dtype        = {r.dtype}")
print(f"  r.item()       = {r.item()}")
print(f"  和 Python float 比较: r == 0.5 ? {bool(r == 0.5)}")
print(f"  注意: repr 显示 'tensor(0.5000)', 不是 '0.5'")

print("\n" + "=" * 70)
print("4) 几个特殊角的对照表(验证单位)")
print("=" * 70)
print(f"  {'角度':<8}{'弧度':<14}{'cos':<22}{'sin'}")
print("  " + "-" * 58)
for deg in [0, 30, 45, 60, 90, 180]:
    rad = deg * math.pi / 180
    c = torch.cos(torch.tensor(rad))
    s = torch.sin(torch.tensor(rad))
    print(f"  {deg:>4}°   {rad:<14.6f}{str(c):<22}{s}")
print("\n  对照: cos 60°=0.5, cos 90°=0(实测约 0), cos 180°=-1  ✅ 单位是弧度")

print("\n" + "=" * 70)
print("5) 张量版: 一次算一批角度")
print("=" * 70)
degs = torch.tensor([0.0, 30.0, 45.0, 60.0, 90.0])
rads = degs * math.pi / 180          # 度数张量 -> 弧度张量
print(f"  角度(度) = {degs.tolist()}")
print(f"  角度(弧度) = {[round(v,4) for v in rads.tolist()]}")
print(f"  torch.cos(rads) = {torch.cos(rads)}")

print("\n" + "=" * 70)
print("6) 你项目里的情况: angles 本来就是弧度, 不需要转换")
print("=" * 70)
dim, rope_base = 64, 1e6
inv_freq = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
angles = torch.outer(torch.arange(3), inv_freq)
print(f"  angles[1][:4] = {[round(v,4) for v in angles[1][:4].tolist()]}  <- 已经是弧度")
print(f"  torch.cos(angles[1][:4]) = {torch.cos(angles[1][:4])}")
print(f"  torch.sin(angles[1][:4]) = {torch.sin(angles[1][:4])}")
print("  RoPE 的公式天生产弧度(theta = t * w), 所以直接进 cos/sin 正确")

print("\n" + "=" * 70)
print("7) 换算公式(记住这个)")
print("=" * 70)
print("  弧度 = 角度 × pi / 180")
print("  角度 = 弧度 × 180 / pi")
print("\n  Python 可用: math.pi 或 torch.pi (两者数值相同)")
print(f"  math.pi  = {math.pi}")
print(f"  torch.pi = {torch.pi}")
print(f"  相等 ? {math.pi == torch.pi}")

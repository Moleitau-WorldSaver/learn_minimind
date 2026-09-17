import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import torch

dim, end = 8, 1          # head_dim=8, 只看 1 个位置
angles = torch.tensor([[0.1, 0.2, 0.3, 0.4]])   # shape (1, dim//2=4)
cos = torch.cos(angles)
sin = torch.sin(angles)

print("head_dim =", dim, "   角度个数 = dim//2 =", dim // 2)
print("angles =", [round(v, 3) for v in angles[0].tolist()])
print("cos    =", [round(v, 3) for v in cos[0].tolist()])
print("sin    =", [round(v, 3) for v in sin[0].tolist()], " <- 这才是'另一半'")

print("\n" + "=" * 66)
print("方案A(你的写法): cat([cos, cos])  <- 两份相同")
print("=" * 66)
A = torch.cat([cos, cos], dim=-1)
print("  结果 shape =", tuple(A.shape), " 值 =", [round(v, 3) for v in A[0].tolist()])

print("\n" + "=" * 66)
print("方案B(凭直觉的错法): cat([cos, sin])  <- 拼接不同的一半")
print("=" * 66)
B = torch.cat([cos, sin], dim=-1)
print("  结果 shape =", tuple(B.shape), " 值 =", [round(v, 3) for v in B[0].tolist()])
print("  为什么错: 后 4 维是 8 个不同维度的 cos/sin, 而 cos/sin 是两个不同的量")
print("  它们参与运算的方式不同(cos 乘原值, sin 乘旋转后的值), 不能混在同一维里")

print("\n" + "=" * 66)
print("为什么必须重复: 数量对不上")
print("=" * 66)
q = torch.randn(1, dim)
print(f"  q 的最后一维 = {q.shape[-1]}   (head_dim = {dim})")
print(f"  角度个数     = {cos.shape[-1]}   (dim//2 = {dim//2})")
print(f"  {q.shape[-1]} != {cos.shape[-1]}  -> 无法逐元素相乘!")
try:
    q * cos
except RuntimeError as e:
    print(f"  实测 q * cos -> RuntimeError: {str(e)[:72]}")
print(f"  重复一份后: {cos.shape[-1]} * 2 = {A.shape[-1]} = head_dim  ✅")
print(f"  实测 q * A -> shape {tuple((q * A).shape)}")

print("\n" + "=" * 66)
print("真正的'配对'发生在前面的维度对半分组")
print("=" * 66)
print("  q 被看成前后两半:")
print(f"    前半 x1 = q[..., :{dim//2}]  -> 与 cos 的前 {dim//2} 个相乘")
print(f"    后半 x2 = q[..., {dim//2}:]  -> 与 cos 的后 {dim//2} 个相乘(值是同一批)")
print("  而 x2 是'换了位置并取负'的那个半, 这就是 rotate_half:")
qh = torch.tensor([[1., 2., 3., 4., 5., 6., 7., 8.]])
rotated = torch.cat([-qh[..., dim // 2:], qh[..., :dim // 2]], dim=-1)
print("    原值       =", qh[0].tolist())
print("    rotate_half =", rotated[0].tolist(), " <- 前后互换并对新后半取负")
print("\n  旋转公式: out = x * cos_cat + rotate_half(x) * sin_cat")
print("  关键: cos 和 sin 各自都要 cat 两份, 因为 x 和 rotate_half(x) 都是 head_dim 维")

print("\n" + "=" * 66)
print("验证: 用两个张量确实能还原成真正的旋转")
print("=" * 66)
xh = torch.tensor([[3.0, 4.0]])
th = torch.tensor([[0.5]])           # 单个角度
c, s = torch.cos(th), torch.sin(th)
# 复数旋转: (x1 + i*x2) * (c + i*s)
x1, x2 = xh[..., 0], xh[..., 1]
new1 = x1 * c - x2 * s
new2 = x1 * s + x2 * c
print(f"  原始向量 (x1,x2) = ({x1.item()}, {x2.item()}), 旋转角 = {th.item()} rad")
print(f"  旋转后     = ({new1.item():.4f}, {new2.item():.4f})")
print(f"  模长不变? {torch.allclose(torch.sqrt(x1**2+x2**2), torch.sqrt(new1**2+new2**2), atol=1e-6)}")

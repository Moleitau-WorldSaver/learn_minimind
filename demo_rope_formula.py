import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import torch

dim = 8
q = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8.])
theta = 0.1
angles = torch.full((1, dim // 2), theta)     # 所有维度对用同一角度, 便于对照
cos_c = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1)[0]
sin_c = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1)[0]

# 路线1: cos/sin + rotate_half 的向量化写法
rotated = torch.cat([-q[dim // 2:], q[: dim // 2]])
out_vec = q * cos_c + rotated * sin_c

# 路线2: 手写半拆分配对旋转(逐对计算)
c, s = torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))
out_manual = torch.empty(dim)
for i in range(dim // 2):
    x1, x2 = q[i], q[i + dim // 2]
    out_manual[i] = x1 * c - x2 * s
    out_manual[i + dim // 2] = x1 * s + x2 * c

print("q           =", q.tolist())
print("rotate_half =", rotated.tolist())
print("cos_cat     =", [round(v, 4) for v in cos_c.tolist()])
print("sin_cat     =", [round(v, 4) for v in sin_c.tolist()])
print()
print("路线1 (q*cos + rotate_half(q)*sin) =", [round(v, 4) for v in out_vec.tolist()])
print("路线2 (逐对半拆分手算)             =", [round(v, 4) for v in out_manual.tolist()])
print("两者一致 ?", torch.allclose(out_vec, out_manual, atol=1e-6))
print()
print("模长:", round(q.norm().item(), 6), "->", round(out_vec.norm().item(), 6),
      " 不变?", torch.allclose(q.norm(), out_vec.norm(), atol=1e-6))
print()
print("配对关系(半拆分): (q0,q4) (q1,q5) (q2,q6) (q3,q7)")
print("每对共享同一角度; cos 的前4个管前半, 后4个管后半(值相同)")
print()
print("如果只 cat 一份(4个), 后4维就没东西乘 -> 报错:")
try:
    q * cos_c[: dim // 2]
except RuntimeError as e:
    print("  RuntimeError:", str(e)[:70])

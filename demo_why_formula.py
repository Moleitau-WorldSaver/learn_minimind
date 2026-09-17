import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import torch

dim = 4                      # head_dim=4 -> 2 对维度, 便于手算
q = torch.tensor([1.0, 2.0, 3.0, 4.0])
theta = [0.3, 0.7]           # 两个维度对各自的角度
c = [torch.cos(torch.tensor(t)).item() for t in theta]
s = [torch.sin(torch.tensor(t)).item() for t in theta]

print("设定: head_dim =", dim, " 共", dim // 2, "对维度")
print("q =", q.tolist())
print("角度: 第0对 theta0 =", theta[0], "  第1对 theta1 =", theta[1])
print("cos =", [round(v, 4) for v in c], "  sin =", [round(v, 4) for v in s])

print("\n" + "=" * 64)
print("步骤0: RoPE 的定义 —— 每 2 维做一次旋转")
print("=" * 64)
print("维度对 (q0, q2) 共享 theta0,  维度对 (q1, q3) 共享 theta1  [半拆分配对]")
print()
print("对第 i 对:  [q_i      ]   ->   [cos_i  -sin_i] [q_i    ]")
print("            [q_{i+d/2}]        [sin_i   cos_i] [q_{i+d/2}]")
print()
print("展开:")
print(f"  out0 = q0*cos0 - q2*sin0 = {q[0].item()}*{c[0]:.4f} - {q[2].item()}*{s[0]:.4f} = {(q[0]*c[0]-q[2]*s[0]).item():.4f}")
print(f"  out2 = q0*sin0 + q2*cos0 = {q[0].item()}*{s[0]:.4f} + {q[2].item()}*{c[0]:.4f} = {(q[0]*s[0]+q[2]*c[0]).item():.4f}")
print(f"  out1 = q1*cos1 - q3*sin1 = {(q[1]*c[1]-q[3]*s[1]).item():.4f}")
print(f"  out3 = q1*sin1 + q3*cos1 = {(q[1]*s[1]+q[3]*c[1]).item():.4f}")
manual = torch.tensor([q[0]*c[0]-q[2]*s[0], q[1]*c[1]-q[3]*s[1],
                       q[0]*s[0]+q[2]*c[0], q[1]*s[1]+q[3]*c[1]])
print("\n手算结果 out =", [round(v.item(), 4) for v in manual])

print("\n" + "=" * 64)
print("步骤1: 发现 out 的每一项都能写成 'A*cos + B*sin' 的形式")
print("=" * 64)
print("  out0 = q0*cos0 + (-q2)*sin0")
print("  out1 = q1*cos1 + (-q3)*sin1")
print("  out2 = q2*cos0 + ( q0)*sin0")
print("  out3 = q3*cos1 + ( q1)*sin1")
print()
print("如果我们能把 cos 排成 [cos0, cos1, cos0, cos1], sin 排成 [sin0, sin1, sin0, sin1],")
print("再把上面那个 B 部分排成 [-q2, -q3, q0, q1], 就可以整体写成逐元素乘加!")

print("\n" + "=" * 64)
print("步骤2: 那个 B 部分就是 rotate_half(q)")
print("=" * 64)
h = dim // 2
rotated = torch.cat([-q[h:], q[:h]])
print("  q           =", q.tolist())
print("  rotate_half =", [round(v.item(), 4) for v in rotated], "   (后半取负搬到前, 前半搬到后)")
print("  对照上面需要的 B = [-q2, -q3, q0, q1]  ✅ 完全一致")

print("\n" + "=" * 64)
print("步骤3: cos 排成 [cos0, cos1, cos0, cos1] 就是从 [cos0, cos1] 复制一份!")
print("=" * 64)
cos_half = torch.tensor(c)
sin_half = torch.tensor(s)
cos_full = torch.cat([cos_half, cos_half], dim=-1)
sin_full = torch.cat([sin_half, sin_half], dim=-1)
print("  cos 半份 [cos0, cos1]      =", [round(v.item(), 4) for v in cos_half])
print("  cos 全份 cat([c,c], -1)    =", [round(v.item(), 4) for v in cos_full], "  <- 复制一份")
print("  sin 半份 [sin0, sin1]      =", [round(v.item(), 4) for v in sin_half])
print("  sin 全份 cat([s,s], -1)    =", [round(v.item(), 4) for v in sin_full], "  <- 复制一份")

print("\n" + "=" * 64)
print("步骤4: 向量化公式 = 手算结果?")
print("=" * 64)
vec = q * cos_full + rotated * sin_full
print("  q * cos_full               =", [round(v.item(), 4) for v in (q*cos_full)])
print("  rotate_half(q) * sin_full  =", [round(v.item(), 4) for v in (rotated*sin_full)])
print("  相加                       =", [round(v.item(), 4) for v in vec])
print("  手算                       =", [round(v.item(), 4) for v in manual])
print("  两者一致 ?", torch.allclose(vec, manual, atol=1e-6), " ✅ 这就是'为什么可以这么求'")

print("\n" + "=" * 64)
print("步骤5: 为什么必须复制而不能只出一份")
print("=" * 64)
print("  维度 0 需要 cos0,  维度 1 需要 cos1")
print("  维度 2 也需要 cos0, 维度 3 也需要 cos1   <- 和前半用的是同一批值!")
print("  所以 cos 的排布必须是 [cos0, cos1, cos0, cos1]")
print("  半份只有 2 个, 要覆盖 4 个维度, 只能复制 -> cat([c, c])")

print("\n" + "=" * 64)
print("模长自检(旋转的物理性质)")
print("=" * 64)
print(f"  |q|     = {q.norm().item():.6f}")
print(f"  |out|   = {vec.norm().item():.6f}")
print(f"  相等 ? {torch.allclose(q.norm(), vec.norm(), atol=1e-6)}  <- 旋转不改变长度")

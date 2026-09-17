import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import math
import torch


def rotate_half(x):
    return torch.cat((-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1)


dim = 4
q = torch.tensor([1.0, 2.0, 3.0, 4.0])
h = dim // 2
print("=" * 78)
print("设定")
print("=" * 78)
print(f"  head_dim = {dim},  h = {h}")
print(f"  q = {q.tolist()}")
print(f"  配对方式(半拆分): (q0, q{h}) (q1, q{h+1})  <- 每对共享一个旋转角")
print()

print("=" * 78)
print("第 1 步: RoPE 的定义 —— 每一对做一次 2D 旋转")
print("=" * 78)
print("  对第 i 对 (a, b) = (q_i, q_{i+h}), 角度 theta_i:")
print()
print("      out_i     = a * cos_i  -  b * sin_i")
print("      out_{i+h} = a * sin_i  +  b * cos_i")
print()
theta = [0.3, 0.7]
c = [math.cos(t) for t in theta]
s = [math.sin(t) for t in theta]
print(f"  cos = {[round(v,4) for v in c]}    sin = {[round(v,4) for v in s]}")
print()
manual = torch.zeros(dim)
for i in range(h):
    a, b = q[i].item(), q[i + h].item()
    manual[i] = a * c[i] - b * s[i]
    manual[i + h] = a * s[i] + b * c[i]
print("  手算结果:")
for i in range(h):
    print(f"    out{i} = q{i}*cos{i} - q{i+h}*sin{i} = {q[i]:.0f}*{c[i]:.4f} - {q[i+h]:.0f}*{s[i]:.4f} = {manual[i]:.4f}")
for i in range(h):
    print(f"    out{i+h} = q{i}*sin{i} + q{i+h}*cos{i} = {q[i]:.0f}*{s[i]:.4f} + {q[i+h]:.0f}*{c[i]:.4f} = {manual[i+h]:.4f}")
print(f"\n  手算结果 = {[round(v.item(),4) for v in manual]}")

print("\n" + "=" * 78)
print("第 2 步: 把公式重排成 'A*cos + B*sin' 的形式")
print("=" * 78)
print("  观察: 每个 out 都能写成 '某个值 × cos + 某个值 × sin'")
print()
print("    out0 = q0*cos0 + (-q2)*sin0        <- 换行看系数")
print("    out1 = q1*cos1 + (-q3)*sin1")
print("    out2 = q2*cos0 + ( q0)*sin0")
print("    out3 = q3*cos1 + ( q1)*sin1")
print()
print("  如果把 cos 排成 [cos0, cos1, cos0, cos1], sin 同样, 就能写成:")
print("      out = A * cos_full  +  B * sin_full")
print()
print("  那么 A 和 B 分别是什么?")
print("    A = [q0, q1, q2, q3]              <- 就是 q 本身 ✅")
print("    B = [-q2, -q3, q0, q1]            <- 需要构造 ❓")

print("\n" + "=" * 78)
print("第 3 步: B 正好就是 rotate_half(q)!")
print("=" * 78)
rh = rotate_half(q)
print(f"  q           = {q.tolist()}")
print(f"  rotate_half = {[round(v.item(),1) for v in rh]}")
print()
print("  展开 rotate_half 的每个部分:")
print(f"    x[..., {h}:]  = {q[h:].tolist()}      <- 后半")
print(f"    -x[..., {h}:] = {(-q[h:]).tolist()}      <- 后半取负 -> 放在前面")
print(f"    x[..., :{h}]  = {q[:h].tolist()}      <- 前半      -> 放在后面")
print()
print(f"  拼起来 = {[round(v.item(),1) for v in rh]}")
print(f"  需要的 B = [-q2, -q3, q0, q1] = {[-q[2].item(), -q[3].item(), q[0].item(), q[1].item()]}")
print(f"  完全一致 ? {torch.equal(rh, torch.tensor([-q[2], -q[3], q[0], q[1]]))}")

print("\n" + "=" * 78)
print("第 4 步: 验证向量化公式 == 手算旋转")
print("=" * 78)
cos_full = torch.tensor([c[0], c[1], c[0], c[1]])
sin_full = torch.tensor([s[0], s[1], s[0], s[1]])
vec = q * cos_full + rotate_half(q) * sin_full
print(f"  q * cos_full              = {[round(v.item(),4) for v in q*cos_full]}")
print(f"  rotate_half(q) * sin_full = {[round(v.item(),4) for v in rotate_half(q)*sin_full]}")
print(f"  相加                      = {[round(v.item(),4) for v in vec]}")
print(f"  手算                      = {[round(v.item(),4) for v in manual]}")
print(f"  一致 ? {torch.allclose(vec, manual, atol=1e-6)}  ✅")

print("\n" + "=" * 78)
print("逐项验算(out0 为什么对)")
print("=" * 78)
print(f"  out0 = q0*cos_full[0] + rotate_half(q)[0]*sin_full[0]")
print(f"       = {q[0]:.0f}*{cos_full[0]:.4f} + ({rotate_half(q)[0]:.0f})*{sin_full[0]:.4f}")
print(f"       = {(q[0]*cos_full[0]).item():.4f} + {(rotate_half(q)[0]*sin_full[0]).item():.4f}")
print(f"       = {vec[0]:.4f}")
print(f"  手算 out0 = q0*cos0 - q2*sin0 = {(q[0]*c[0] - q[2]*s[0]).item():.4f}")
print(f"  一致 ? {abs(vec[0].item() - (q[0]*c[0] - q[2]*s[0]).item()) < 1e-6}")
print()
print("  关键: rotate_half(q)[0] = -q2, 所以那一项 = (-q2)*sin0 = 减号 ✅")

print("\n" + "=" * 78)
print("所以这个切法要干什么 —— 一句话")
print("=" * 78)
print("  把'后半取负搬到前面, 前半搬到后面', 造出旋转公式里的第二个加数")
print()
print("  不这么切会怎样: 就得写循环或者复杂的索引, 无法用一次逐元素运算完成")
print("  这么切之后: out = q*cos + rotate_half(q)*sin   一行搞定整个张量")

print("\n" + "=" * 78)
print("配套的 cos/sin 也要对应地复制(呼应前面的 cat)")
print("=" * 78)
print(f"  角度只有 {h} 个(dim//2), 但 q 有 {dim} 维")
print(f"  cos_full 必须排成 [c0, c1, c0, c1] 才能和 A、B 两个加数都对齐")
print(f"  这就是 torch.cat([cos, cos], dim=-1) 的来历")

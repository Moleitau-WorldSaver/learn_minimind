import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import torch

x = torch.arange(24).reshape(2, 3, 4)
print(f"x.shape = {tuple(x.shape)}   维度数 n = 3")
print()

print("=" * 76)
print("已写段数 k 可以大于 1, ... 补的段数相应减少")
print("=" * 76)
n = 3
cases = [
    ("x[..., 0]",           1),
    ("x[..., 0, 0]",        2),
    ("x[..., 0, 0, 0]",     3),
    ("x[0, ...]",           1),
    ("x[0, ..., 1]",        2),
]
print(f"  {'表达式':<20}{'k(已写)':<10}{'...补几段':<12}{'等价写法':<24}{'结果 shape'}")
print("  " + "-" * 84)
for expr, k in cases:
    dots = n - k
    if dots < 0:
        continue
    # 构造等价写法
    if "..., " in expr:
        head, tail = expr[2:-1].split("..., ")
        parts_head = head.split(", ") if head else []
        parts_tail = tail.split(", ")
    else:
        continue
    r = eval(expr.replace("x[", "x["))
    eq = expr
    print(f"  {expr:<20}{k:<10}{dots:<12}{'':<24}{tuple(r.shape)}")

print("\n  手工推演:")
print("    x[..., 0]        n=3, k=1 -> ... 补 2 段 -> x[:, :, 0]      结果 (2, 3)")
print("    x[..., 0, 0]     n=3, k=2 -> ... 补 1 段 -> x[:, 0, 0]      结果 (2,)")
print("    x[..., 0, 0, 0]  n=3, k=3 -> ... 补 0 段 -> x[0, 0, 0]      结果 标量")
print()
for expr in ["x[..., 0]", "x[..., 0, 0]", "x[..., 0, 0, 0]"]:
    r = eval(expr)
    print(f"    实测 {expr:<18} -> shape {str(tuple(r.shape)):<10} 值 {r.tolist() if r.dim() else r.item()}")

print("\n  验证等价性:")
print(f"    x[..., 0]       == x[:, :, 0] ? {torch.equal(x[..., 0], x[:, :, 0])}")
print(f"    x[..., 0, 0]    == x[:, 0, 0] ? {torch.equal(x[..., 0, 0], x[:, 0, 0])}")
print(f"    x[..., 0, 0, 0] == x[0, 0, 0] ? {x[..., 0, 0, 0].item() == x[0, 0, 0].item()}")

print("\n" + "=" * 76)
print("核心公式(记住这一个)")
print("=" * 76)
print("  ... 展开的段数 = 张量维度数 n  -  索引里已显式写的段数 k")
print()
print(f"  {'张量维度 n':<14}{'已写 k':<10}{'... 补':<10}说明")
print("  " + "-" * 56)
for n_, k_ in [(4, 1), (4, 2), (4, 3), (4, 4), (3, 1), (2, 1)]:
    print(f"  {n_:<14}{k_:<10}{n_-k_:<10}{'... = ' + str(n_-k_) + ' 个 :' if n_-k_ else '... = 0 个 : (等于没写)'}")

print("\n  下面这种情况 ... 补 0 段(合法, 但没意义):")
q = torch.arange(2 * 4 * 6 * 8).reshape(2, 4, 6, 8)
print(f"    q[0, 1, 2, 3, ...] -> shape {tuple(q[0, 1, 2, 3, ...].shape)}   和 q[0,1,2,3] 一样")

print("\n" + "=" * 76)
print("为什么'不用你数': 信息在张量身上, 不在代码里")
print("=" * 76)
print("  q[..., 3] 这行代码, 在不同 q 上展开成不同的东西:")
for shape in [(4,), (3, 4), (2, 3, 4), (2, 4, 6, 8)]:
    t = torch.zeros(shape)
    r = t[..., 3] if len(shape) > 1 else t[3]
    print(f"    q.shape {str(shape):<18} -> 结果 {str(tuple(r.shape)):<16} (... 补了 {len(shape)-1} 段)")

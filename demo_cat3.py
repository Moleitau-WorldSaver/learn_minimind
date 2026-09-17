import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import torch

a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])


def show(t):
    return f"shape={tuple(t.shape)}  {t.tolist()}"


print("输入:")
print("  a =", show(a))
print("  b =", show(b))

print("\n输出(dim=0):")
r0 = torch.cat((a, b), dim=0)
print("  ", show(r0))
print("  图形化:  a 的 2 行  +  b 的 2 行  =  4 行(竖着接)")

print("\n输出(dim=1):")
r1 = torch.cat((a, b), dim=1)
print("  ", show(r1))
print("  图形化:  a 的 2 列  +  b 的 2 列  =  4 列(横着接)")

print("\n--- 元素个数对照 ---")
print(f"  a: {a.numel()} 个,  b: {b.numel()} 个")
print(f"  cat dim=0: {r0.numel()} 个  =  2 + 2   (相加)")
print(f"  cat dim=1: {r1.numel()} 个  =  2 + 2   (相加)")
print(f"  对比 stack: {torch.stack((a, b)).numel()} 个  =  4 x 2  (相乘)")

print("\n--- 原张量有没有被改动? ---")
before = a.tolist()
torch.cat((a, b), dim=0)
print(f"  cat 之前 a = {before}")
print(f"  cat 之后 a = {a.tolist()}")
print(f"  a 没变 -> cat 是'新建',不是'就地修改'")

print("\n--- 返回的是新张量 ---")
r = torch.cat((a, b), dim=0)
print(f"  r is a ? {r is a}    r is b ? {r is b}")
print(f"  新对象: id 不同 = {id(r) != id(a) and id(r) != id(b)}")

print("\n--- 多维情形: (2,3,4) 拼 (2,3,4) ---")
t1 = torch.arange(24).reshape(2, 3, 4)
t2 = torch.arange(24, 48).reshape(2, 3, 4)
for d in [0, 1, 2]:
    print(f"  dim={d} -> {tuple(torch.cat((t1, t2), dim=d).shape)}   (第 {d} 维 {t1.shape[d]}+{t1.shape[d]}={2*t1.shape[d]})")

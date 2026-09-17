import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import torch


def show(t):
    return f"shape={tuple(t.shape)}  {t.tolist()}"


print("=" * 70)
print("A) 可以 cat 任意多个(2个、3个、5个都行)")
print("=" * 70)
t1 = torch.tensor([[1, 2]])
t2 = torch.tensor([[3, 4]])
t3 = torch.tensor([[5, 6]])
t4 = torch.tensor([[7, 8]])
t5 = torch.tensor([[9, 10]])

for n in [2, 3, 5]:
    ts = [t1, t2, t3, t4, t5][:n]
    r = torch.cat(ts, dim=0)
    print(f"  cat({n}个, dim=0) -> shape={tuple(r.shape)}  {r.flatten().tolist()}")

print("\n" + "=" * 70)
print("B) 拼接维的长度可以不同(这才是 cat 的常态)")
print("=" * 70)
a = torch.tensor([[1, 2], [3, 4]])      # (2,2)
c = torch.tensor([[9, 10]])             # (1,2)  只有 1 行!
b = torch.tensor([[5, 6], [7, 8], [11, 12]])  # (3,2)  有 3 行!

print("  a =", show(a))
print("  c =", show(c), " <- 只有 1 行")
print("  b =", show(b), " <- 有 3 行")
r = torch.cat((a, c, b), dim=0)
print("  cat((a, c, b), dim=0) ->", show(r))
print("  行数: 2 + 1 + 3 = 6  ✅ 各不相同也能拼")

print("\n" + "=" * 70)
print("C) 非拼接维必须相同(这才是唯一的硬约束)")
print("=" * 70)
ok = torch.tensor([[1, 2], [3, 4]])       # (2,2)
bad = torch.tensor([[1, 2, 3], [4, 5, 6]])  # (2,3)
print("  ok  =", show(ok), " 非拼接维(dim=1) = 2")
print("  bad =", show(bad), " 非拼接维(dim=1) = 3  <- 不匹配")
try:
    torch.cat((ok, bad), dim=0)
except RuntimeError as e:
    print("  cat(dim=0) -> RuntimeError:", str(e)[:88])

print("\n" + "=" * 70)
print("D) 一维张量:长度完全随意")
print("=" * 70)
parts = [torch.tensor([1.0]), torch.tensor([2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0])]
print("  三段的长度分别是:", [len(p) for p in parts])
r = torch.cat(parts, dim=-1)
print("  cat -> shape", tuple(r.shape), r.tolist())
print("  长度 = 1+2+3 = 6 ✅ 完全自由")

print("\n" + "=" * 70)
print("E) 边界情况")
print("=" * 70)
try:
    torch.cat(t1, dim=0)
except TypeError as e:
    print(f"  传单个张量(没打包)  -> TypeError: {str(e)[:75]}")
try:
    torch.cat((), dim=0)
except Exception as e:
    print(f"  传空序列             -> {type(e).__name__}: {str(e)[:70]}")
try:
    torch.cat((t1,), dim=0)
    print(f"  只传 1 个张量        -> 合法!返回 {show(torch.cat((t1,), dim=0))}(相当于复制一份)")
except Exception as e:
    print("  只传 1 个 ->", type(e).__name__, e)

print("\n" + "=" * 70)
print("F) list 也能传(不一定是 tuple)")
print("=" * 70)
print("  cat([t1, t2], dim=0) ->", show(torch.cat([t1, t2], dim=0)))

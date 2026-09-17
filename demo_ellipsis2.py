import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import torch

print("=" * 76)
print("为什么一个索引里只能有一个 ...  —— 数一下索引个数")
print("=" * 76)

x = torch.arange(2 * 3 * 4).reshape(2, 3, 4)   # 3 维
print(f"  x.shape = {tuple(x.shape)}  -> 3 维, 索引里最多给 3 个位置")
print()
print("  x[..., 0]      -> ... 展开成 2 个冒号 + 1 个 0 = 3 个位置  OK")
print(f"    实测 shape {tuple(x[..., 0].shape)}")
print()
print("  x[..., :2]     -> ... 展开成 2 个冒号 + 1 个切片 = 3 个位置  OK")
print(f"    实测 shape {tuple(x[..., :2].shape)}")
print()

for expr, fn in [("x[..., ..., 0]", lambda: x[..., ..., 0]),
                 ("x[..., :, ...]", lambda: x[..., :, ...])]:
    try:
        r = fn()
        print(f"  {expr:<18} -> OK shape {tuple(r.shape)}")
    except Exception as e:
        print(f"  {expr:<18} -> {type(e).__name__}: {str(e)[:60]}")

print("\n  规则: 两个 ... 会让展开后的总位置数超过张量维度数 -> 报错")
print("  直觉理解: ... 是'剩下的全包了', 出现两次就'抢'了, 语义有歧义")

print("\n" + "=" * 76)
print("4 维张量再验一次(更清楚)")
print("=" * 76)
q = torch.arange(2 * 4 * 6 * 8).reshape(2, 4, 6, 8)
print(f"  q.shape = {tuple(q.shape)}  -> 4 维")
print(f"  q[..., 0]        -> shape {tuple(q[..., 0].shape)}   (... = 3 个冒号)")
print(f"  q[0, ...]        -> shape {tuple(q[0, ...].shape)}   (... = 3 个冒号)")
print(f"  q[0, ..., 0]     -> shape {tuple(q[0, ..., 0].shape)}   (... = 2 个冒号, 夹在中间)")
try:
    q[..., ..., 0]
except Exception as e:
    print(f"  q[..., ..., 0]   -> {type(e).__name__}: {str(e)[:55]}")

print("\n" + "=" * 76)
print("... 展开规则总结")
print("=" * 76)
print("  设张量 n 维, 索引里已经显式写了 k 个位置, 那么 ... 展开成 (n - k - 1) 个冒号")
print()
for shape, idx_desc, k in [((2, 3, 4), "x[..., 0]", 1),
                           ((2, 3, 4), "x[..., :2]", 1),
                           ((2, 4, 6, 8), "q[..., 0]", 1),
                           ((2, 4, 6, 8), "q[0, ..., 0]", 2)]:
    n = len(shape)
    print(f"  {str(shape):<16} {idx_desc:<16} n={n}, k={k}  ->  ... 展开成 {n-k-1} 个冒号")

print("\n" + "=" * 76)
print("Ellipsis 作为普通对象的补充验证")
print("=" * 76)
print(f"  ... is Ellipsis       { ... is Ellipsis}")
print(f"  bool(...)             {bool(...)}          <- 真值")
print(f"  ... == Ellipsis       {... == Ellipsis}    <- 单例比较")
print(f"  [1, ..., 2]           {[1, ..., 2]}")
print(f"  'x' in (...,)         {'x' in (...,)}")
print(f"  ... == ...            {... == ...}")

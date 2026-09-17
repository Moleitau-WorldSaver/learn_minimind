import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import torch

print("=" * 78)
print("核心公式: ... 展开成 (维度数 - 已写段数) 个冒号")
print("=" * 78)
print("  维度数 = len(x.shape)   <- 张量自己知道!")
print()

for shape in [(4,), (3, 4), (2, 3, 4), (2, 4, 6, 8)]:
    n = len(shape)
    print(f"  x.shape = {str(shape):<16} 维度数 n = {n}")

print("\n" + "=" * 78)
print("同一个表达式 x[..., 3], 在不同张量上展开结果不同")
print("=" * 78)
print(f"  {'x.shape':<16}{'n':<5}{'已写段数 k':<12}{'... 补几个 :':<16}{'等价写法'}")
print("  " + "-" * 74)
for shape in [(4,), (3, 4), (2, 3, 4), (2, 4, 6, 8)]:
    n = len(shape)
    k = 1                                   # 显式写了 "3" 这一段
    dots = n - k
    equiv = "x[" + ", ".join([":"] * dots + ["3"]) + "]"
    print(f"  {str(shape):<16}{n:<5}{k:<12}{dots:<16}{equiv}")

print("\n" + "=" * 78)
print("直接证据: 用 __getitem__ 看 torch 实际收到的索引元组")
print("=" * 78)


class Spy(torch.Tensor):
    """一个只用来偷看索引的子类(只是个包装, 不改变行为)"""


def peek(shape):
    t = torch.zeros(shape)
    # 手动模拟: 把 ... 换成实际会展开的东西
    n = len(shape)


class Show:
    def __init__(self, shape):
        self.shape = shape

    def __getitem__(self, key):
        return key


for shape in [(3, 4), (2, 3, 4), (2, 4, 6, 8)]:
    t = torch.arange(int(torch.tensor(shape).prod())).reshape(shape)
    # 真张量: 看结果 shape 就知道展开成了几段
    r = t[..., 3] if len(shape) > 1 else t[..., 3]
    print(f"  x.shape {str(shape):<16} x[..., 3] -> 结果 shape {str(tuple(r.shape)):<16}"
          f" (原 n={len(shape)}, 减掉 1 因为最后那段是数字 3)")

print("\n" + "=" * 78)
print("为什么张量'知道': 索引的段数必须等于维度数")
print("=" * 78)
print("  Python 把索引交给 x.__getitem__(), torch 那边能看到 x 的 shape")
print("  torch 的规则: 展开 ... 之后, 总段数必须正好等于维度数")
print()
x3 = torch.arange(24).reshape(2, 3, 4)
print(f"  x3.shape = {tuple(x3.shape)}  维度数 3")
print()
print("  合法索引(展开后都是 3 段):")
for expr, fn in [("x3[..., 3]", lambda t: t[..., 3]),
                 ("x3[..., :2]", lambda t: t[..., :2]),
                 ("x3[0, ...]", lambda t: t[0, ...]),
                 ("x3[0, ..., 1]", lambda t: t[0, ..., 1])]:
    r = fn(x3)
    print(f"    {expr:<18} -> 展开成 3 段, 结果 shape {tuple(r.shape)}")
print()
print("  非法索引(展开后段数不对):")
for expr, fn in [("x3[..., 0, 0, 0]", lambda t: t[..., 0, 0, 0])]:
    try:
        fn(x3)
        print(f"    {expr:<18} -> OK")
    except Exception as e:
        print(f"    {expr:<18} -> {type(e).__name__}: {str(e)[:50]}")

print("\n" + "=" * 78)
print("所以: 写代码的人不需要数几段 —— 张量的维度数就是答案")
print("=" * 78)
print("  你写:  q[..., 3]")
print("  张量告诉 torch: 我是 4 维的")
print("  torch 算: 总共要 4 段, 你已经给了 1 段(那个 3), 所以 ... = 3 个冒号")
print("  实际执行: q[:, :, :, 3]")
print()
print("  证据: 维度不同但表达式相同, 结果段数不同")
for shape in [(1, 2, 3), (1, 2, 3, 4), (1, 2, 3, 4, 5)]:
    t = torch.zeros(shape)
    r = t[..., 0]
    print(f"    t.shape {str(shape):<20} t[..., 0] -> 结果 {str(tuple(r.shape)):<18}"
          f" 省略了 {len(shape) - 1} 维")

print("\n" + "=" * 78)
print("对比: : 永远只占 1 段, 不会自适应")
print("=" * 78)
print("  :        固定一个维度")
print("  ...      自适应, 吃掉剩余所有维度(可能是 0 个)")
print()
t = torch.zeros(2, 3, 4)
print(f"  t[:, 0]      -> {tuple(t[:, 0].shape)}    第一个 : 只管第 0 维")
print(f"  t[..., 0]    -> {tuple(t[..., 0].shape)}    ... 管了第 0 和第 1 维")

import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import torch

x = torch.arange(2 * 3 * 4).reshape(2, 3, 4)
print(f"x.shape = {tuple(x.shape)}   (2, 3, 4)\n")

print("=" * 84)
print("穷举: ... 后面可以接什么(不是固定搭配)")
print("=" * 84)
print(f"  {'写法':<22}{'索引元组':<44}{'结果 shape'}")
print("  " + "-" * 80)
cases = [
    ("x[...]",            lambda t: t[...]),
    ("x[..., :]",         lambda t: t[..., :]),
    ("x[..., 0]",         lambda t: t[..., 0]),
    ("x[..., :2]",        lambda t: t[..., :2]),
    ("x[..., 1:3]",       lambda t: t[..., 1:3]),
    ("x[..., ::2]",       lambda t: t[..., ::2]),
    ("x[..., None]",      lambda t: t[..., None]),
    ("x[0, ...]",         lambda t: t[0, ...]),
    ("x[0, ..., 0]",      lambda t: t[0, ..., 0]),
    ("x[:, ..., 0]",      lambda t: t[:, ..., 0]),
    ("x[..., :, 0]",      lambda t: t[..., :, 0]),
    ("x[1, ..., 2, 3]",   lambda t: t[1, ..., 2, 3]),
]
for expr, fn in cases:
    try:
        r = fn(x)
        print(f"  {expr:<22}{'':<44}{tuple(r.shape)}")
    except Exception as e:
        print(f"  {expr:<22}{'':<44}{type(e).__name__}: {str(e)[:30]}")

print("\n" + "=" * 84)
print("关键 1: x[...] 单独用也合法, 后面不一定有逗号和 :)")
print("=" * 84)
print(f"  x[...]        shape = {tuple(x[...].shape)}")
print(f"  等于 x 本身 ? {torch.equal(x[...], x)}")
print("  ... 展开成 3 个冒号 -> 等于 x[:, :, :]")
print(f"  验证 x[...] == x[:, :, :] ? {torch.equal(x[...], x[:, :, :])}")

print("\n" + "=" * 84)
print("关键 2: x[..., :] 里的 : 是'多余的'")
print("=" * 84)
print(f"  x[..., :]     shape = {tuple(x[..., :].shape)}")
print(f"  x[...]        shape = {tuple(x[...].shape)}")
print(f"  两者相同 ? {torch.equal(x[...], x[..., :])}")
print("  -> x[..., :] 的意思是: ... 吃掉前 2 维, : 管最后 1 维")
print("     = x[:, :, :]  完全一样")

print("\n" + "=" * 84)
print("关键 3: ... 也可以完全不用, 全靠 :")
print("=" * 84)
print(f"  x[0, :, :]    shape = {tuple(x[0, :, :].shape)}")
print(f"  x[0, ...]     shape = {tuple(x[0, ...].shape)}")
print(f"  等价 ? {torch.equal(x[0, :, :], x[0, ...])}")
print()
print(f"  x[:, :, 0]    shape = {tuple(x[:, :, 0].shape)}")
print(f"  x[..., 0]     shape = {tuple(x[..., 0].shape)}")
print(f"  等价 ? {torch.equal(x[:, :, 0], x[..., 0])}")

print("\n" + "=" * 84)
print("关键 4: : 也可以单独用, 没有 ...")
print("=" * 84)
print(f"  x[:]          shape = {tuple(x[:].shape)}      <- 全部要")
print(f"  x[0]          shape = {tuple(x[0].shape)}      <- 取第 0 个")
print(f"  x[:, 1]       shape = {tuple(x[:, 1].shape)}   <- 每行取第 1 列")

print("\n" + "=" * 84)
print("组合规律: 逗号分隔的每一项是'独立的一个维度指定'")
print("=" * 84)
print("  x[  A  ,  B  ,  C  ]")
print("      维度0  维度1  维度2      <- 每项管一个维度")
print()
print("  每一项可以是:")
print("    整数   -> 取该维的第 n 个(维度消失)")
print("    :      -> 该维全要(维度保留)")
print("    a:b    -> 该维切片(维度保留)")
print("    ...    -> 吃掉若干个维度(展开成若干个 :)")
print("    None   -> 新增一个长度 1 的维度")
print()
print("  '有 ... 必有 :' 不成立: ... 可以单独出现, : 也可以单独出现")

print("\n" + "=" * 84)
print("其他常见组合(都合法)")
print("=" * 84)
print(f"  x[..., None]      -> {tuple(x[..., None].shape)}     <- 末尾加一维")
print(f"  x[None, ...]      -> {tuple(x[None, ...].shape)}     <- 开头加一维")
print(f"  x[..., -1]        -> {tuple(x[..., -1].shape)}       <- 负索引")
print(f"  x[..., ::2]       -> {tuple(x[..., ::2].shape)}      <- 步长切片")

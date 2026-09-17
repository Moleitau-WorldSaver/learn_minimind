import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import torch

print("=" * 78)
print("反例调查: 两个 ... 有时能跑, 有时报错")
print("=" * 78)
x = torch.arange(2 * 3 * 4).reshape(2, 3, 4)
n = x.dim()
print(f"  x.shape = {tuple(x.shape)}  共 {n} 维")
print()

cases = [
    ("x[..., 0]",      lambda t: t[..., 0]),
    ("x[..., :, ...]", lambda t: t[..., :, ...]),
    ("x[:, ..., ...]", lambda t: t[:, ..., ...]),
    ("x[..., ..., 0]", lambda t: t[..., ..., 0]),
    ("x[..., ...]",    lambda t: t[..., ...]),
    ("x[..., ..., ...]", lambda t: t[..., ..., ...]),
]
for expr, fn in cases:
    try:
        r = fn(x)
        print(f"  {expr:<20} -> OK   shape {tuple(r.shape)}")
    except Exception as e:
        print(f"  {expr:<20} -> {type(e).__name__}: {str(e)[:58]}")

print("\n" + "=" * 78)
print("规律: 多个 ... 只在'多余的 ... 能展开成 0 个冒号'时合法")
print("=" * 78)
print("  每个 ... 至少要吃掉 0 个维度, 但所有 ... 加起来不能超过剩余维度数")
print()
print("  x[..., :, ...]  : 已显式 1 个位置(:), 剩 2 维给两个 ... 分")
print("                    -> 前一个 ... 吃 2 维, 后一个吃 0 维 -> 合法")
print("  x[..., ..., 0]  : 已显式 1 个位置(0), 剩 2 维给两个 ... 分")
print("                    -> 但 0 是标量索引, 要求它落在最后一维")
print("                    -> 实际展开后维度对不上 -> 报错")
print()
print("  NumPy 同样规则验证:")
a = np.arange(24).reshape(2, 3, 4)
for expr, fn in [("a[..., :, ...]", lambda t: t[..., :, ...]),
                 ("a[..., ..., 0]", lambda t: t[..., ..., 0])]:
    try:
        r = fn(a)
        print(f"    {expr:<20} -> OK   shape {r.shape}")
    except Exception as e:
        print(f"    {expr:<20} -> {type(e).__name__}: {str(e)[:45]}")

print("\n" + "=" * 78)
print("实用建议")
print("=" * 78)
print("  别在索引里写两个 ... —— 即使侥幸能跑, 可读性极差")
print("  一个 ... 足够表达'其余维度全要'")
print()
print("  安全写法对照:")
print(f"    x[..., 0]       -> {tuple(x[..., 0].shape)}")
print(f"    x[..., :, 0]    -> {tuple(x[..., :, 0].shape)}   <- 用 : 代替第二个 ..., 更明确")
print(f"    两者等价 ? {torch.equal(x[..., 0], x[..., :, 0])}")

print("\n" + "=" * 78)
print("... 展开规则(最终版)")
print("=" * 78)
print("  设张量 n 维, 索引里显式写了 k 个非 ... 的位置, 且有 m 个 ...")
print("  通常 m=1, 此时 ... 展开为 (n - k) 个完整切片 :")
print()
for shape, desc, k in [((2, 3, 4), "x[..., 0]", 1),
                       ((2, 3, 4), "x[0, ...]", 1),
                       ((2, 4, 6, 8), "q[..., :4]", 1),
                       ((2, 4, 6, 8), "q[0, ..., 0]", 2)]:
    print(f"  {str(shape):<16} {desc:<16} n={len(shape)}, k={k}  ->  ... = {len(shape)-k} 个冒号")

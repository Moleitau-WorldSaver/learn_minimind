import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import builtins
import numpy as np
import torch

print("=" * 78)
print("语法 1: ... 是一个真实的内置对象, 不是标点")
print("=" * 78)
e = ...
print(f"  e = ...          类型 = {type(e)}")
print(f"  type(...).__name__ = {type(...).__name__}")
print(f"  ... is Ellipsis   = {... is Ellipsis}")
print(f"  repr(...)         = {repr(...)}")
print(f"  它是单例(全局只有一个): {eval('...') is ...}")
print(f"  id(...) 两次相同 ? {id(...) == id(Ellipsis)}")
print()
print("  验证: 它可以像普通变量一样用")
myvar = ...
print(f"    myvar = ...           -> {myvar}")
print(f"    myvar is Ellipsis     -> {myvar is Ellipsis}")
print(f"    放在 list 里           -> {[1, ..., 2]}")
print(f"    做字典的值             -> {{'a': ...}}")
print(f"    布尔值(它是真值)       -> {bool(...)}")

print("\n" + "=" * 78)
print("语法 2: 在索引里, ... 表示'其余所有维度全都要'")
print("=" * 78)
x = torch.arange(2 * 3 * 4).reshape(2, 3, 4)
print(f"  x.shape = {tuple(x.shape)}")
print()
print("  这三种写法完全等价:")
print(f"    x[..., 0]        -> shape {tuple(x[..., 0].shape)}")
print(f"    x[:, :, 0]       -> shape {tuple(x[:, :, 0].shape)}")
print(f"    x[0, :, :] 不是同一个! 那是指定第0个batch")
print()
print(f"  x[..., 0] == x[:, :, 0] ? {torch.equal(x[..., 0], x[:, :, 0])}")
print()
print("  ... 的作用: 自动补上足够多的冒号, 让索引的维度数对上")
print(f"    x[..., 0]  等价于 x[:, :, 0]     (补了 2 个冒号)")
print(f"    x[0, ...]  等价于 x[0, :, :]     (补了 2 个冒号)")
print(f"    验证: {torch.equal(x[0, ...], x[0, :, :])}")
print(f"    x[0, ..., 1] 等价于 x[0, :, 1]")
print(f"    验证: {torch.equal(x[0, ..., 1], x[0, :, 1])}")

print("\n" + "=" * 78)
print("语法 3: 冒号 : 和 ... 的区别")
print("=" * 78)
print("  :          = 一个完整的切片(这一维全要)")
print("  ...        = 若干维的 : 的简写(补足剩余维度)")
print("  ,          = 分隔维度")
print()
print(f"  x[..., :2]  -> shape {tuple(x[..., :2].shape)}     ... 管前 2 维, :2 管最后 1 维")
print(f"  x[..., 2:]  -> shape {tuple(x[..., 2:].shape)}")
print(f"  x[..., :]   -> shape {tuple(x[..., :].shape)}     等于 x 本身")
print(f"  验证 x[..., :] is x(内容相同) ? {torch.equal(x[..., :], x)}")

print("\n" + "=" * 78)
print("语法 4: 每个张量最多只能有一个 ...")
print("=" * 78)
try:
    x[..., ..., 0]
except IndexError as err:
    print(f"  x[..., ..., 0] -> IndexError: {str(err)[:70]}")

print("\n" + "=" * 78)
print("语法 5: 4 维张量(你的 q)的实际好处")
print("=" * 78)
q = torch.arange(2 * 4 * 6 * 8).reshape(2, 4, 6, 8)
print(f"  q.shape = {tuple(q.shape)}   (batch, heads, seq, head_dim)")
print()
print("  切最后一维的前半:")
print(f"    q[..., :4]            -> {tuple(q[..., :4].shape)}")
print(f"    q[:, :, :, :4]        -> {tuple(q[:, :, :, :4].shape)}")
print(f"    等价 ? {torch.equal(q[..., :4], q[:, :, :, :4])}")
print()
print("  取第 0 个 batch 的第 2 个 head:")
print(f"    q[0, 2, ...]          -> {tuple(q[0, 2, ...].shape)}")
print(f"    q[0, 2, :, :]         -> {tuple(q[0, 2, :, :].shape)}")
print(f"    等价 ? {torch.equal(q[0, 2, ...], q[0, 2, :, :])}")
print()
print("  核心价值: 维度数从 3 变 4 时, q[..., :h] 一行都不用改")

print("\n" + "=" * 78)
print("语法 6: 省略号在别处的用法")
print("=" * 78)
print("  (1) 类型注解里表示'参数类型任意':")
print("      def f(x: ...) -> ...:  ...")


def f(x: ...) -> ...:
    return x


print(f"      实测 f(5) = {f(5)}")
print()
print("  (2) typing 里的 Ellipsis(可变参数):")
import typing
print(f"      typing.Tuple[int, ...] 表示'任意长度的 int 元组'")
print()
print("  (3) NumPy 里同样规则(它比 PyTorch 更早支持):")
a = np.arange(24).reshape(2, 3, 4)
print(f"      a.shape {a.shape}  a[..., 1].shape = {a[..., 1].shape}")
print(f"      和 torch 行为一致 ? {np.array_equal(a[..., 1], np.array(x[..., 1]))}")

print("\n" + "=" * 78)
print("总结")
print("=" * 78)
print("  ... 是内置对象 Ellipsis 的字面量(单例)")
print("  在索引里它展开为 '若干个 :', 补足到张量的维度数")
print("  一个索引里最多写一个 ...")
print("  作用: 代码与张量维度数解耦")

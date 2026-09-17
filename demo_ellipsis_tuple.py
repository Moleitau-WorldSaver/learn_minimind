import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import torch

x = torch.arange(2 * 3 * 4).reshape(2, 3, 4)
print(f"x.shape = {tuple(x.shape)}")

print("\n" + "=" * 78)
print("层面一: 语法层面 —— 逗号让索引变成一个元组(这是 Python 语法)")
print("=" * 78)
print("  x[0]         -> 索引是 0, 不是元组")
print("  x[0, 1]      -> 索引是 (0, 1), 是元组!")
print("  x[..., :h]   -> 索引是 (Ellipsis, slice(None, h)), 是元组!")
print()
print("  证据: 用 __getitem__ 手动传, 看到的就是元组")


class Show:
    def __getitem__(self, key):
        print(f"    收到 key = {key!r}")
        print(f"    type(key) = {type(key).__name__}")
        if isinstance(key, tuple):
            for i, k in enumerate(key):
                print(f"      key[{i}] = {k!r}  ({type(k).__name__})")
        return None


s = Show()
print("  s[0]:")
s[0]
print("  s[0, 1]:")
s[0, 1]
print("  s[..., :2]:")
s[..., :2]

print("\n" + "=" * 78)
print("层面二: 语义层面 —— 真正执行的是'张量索引', 不是'元组操作'")
print("=" * 78)
print("  ... 不在元组上做任何计算, 而是被'索引机制'解释成'若干个 :'")
print()
print("  调用链:")
print("    x[..., :h]")
print("      -> Python 把 (Ellipsis, slice(None, h)) 传给 x.__getitem__")
print("      -> torch 的 __getitem__ 解析这个元组:")
print("           Ellipsis  -> 展开成若干个 slice(None)(即 :)")
print("           slice      -> 当作切片处理")
print("      -> 返回新的张量")
print()
print("  结论: 元组只是'参数打包的载体', ... 的语义由 torch 定义")

print("\n" + "=" * 78)
print("对比: 同样写法的索引元组, 但对象不是张量")
print("=" * 78)
d = {(0, 1): "a"}
print(f"  字典 d = {{(0,1): 'a'}}")
try:
    d[0, 1]
except Exception as e:
    print(f"  d[0, 1] -> {type(e).__name__}: {str(e)[:50]}")
print("             -> KeyError, 因为字典不做'切片展开', 它只做精确查找")
print(f"  正确写法 d[(0,1)] = {d[(0, 1)]}")
print()
d2 = {0: {1: "b"}}
print(f"  嵌套字典 d2[0][1] = {d2[0][1]}")
print("  -> 同样用 [0, 1] 这种'逗号'语法, 字典里必须写成 d[0][1]")
print("     (字典不把逗号当多参数索引)")

print("\n" + "=" * 78)
print("再看: 不加逗号时, 一切都不是元组")
print("=" * 78)
print(f"  x[0]      -> 索引 = 0,    类型 {type(0).__name__},  结果 shape {tuple(x[0].shape)}")
print(f"  x[:]      -> 索引 = slice(None),  结果 shape {tuple(x[:].shape)}")
print(f"  x[0, :]   -> 索引 = (0, slice(None)),  结果 shape {tuple(x[0, :].shape)}")
print()
print("  逗号的数量 = 显式索引的位置数")
print("  元组里的每一项 = 对应一个维度")

print("\n" + "=" * 78)
print("... 在元组里是'一个元素', 和标量/切片平级")
print("=" * 78)
idx = (Ellipsis, slice(None, 2))
print(f"  手工构造索引元组: {idx!r}")
print(f"  x[idx] 结果 shape = {tuple(x[idx].shape)}")
print(f"  和 x[..., :2] 一致 ? {torch.equal(x[idx], x[..., :2])}")
print()
print("  所以 ... 就是元组里的一个普通元素 Ellipsis")
print("  '这个元素该怎么解释' 由 torch 的索引逻辑决定")

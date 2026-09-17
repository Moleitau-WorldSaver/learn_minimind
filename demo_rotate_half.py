import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import torch


def rotate_half(x):
    return torch.cat(
        (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
    )


print("=" * 78)
print("整个表达式拆成 4 层:")
print("=" * 78)
print("  torch.cat(")
print("      (                             <- 外层圆括号: 把两个张量包成元组")
print("          -x[..., h:],              <- 第二段: 后半 + 取负")
print("          x[..., :h]                <- 第一段: 前半")
print("      ),")
print("      dim=-1                        <- 沿最后一维拼接")
print("  )")
print()

print("=" * 78)
print("语法 1: x.shape[-1] —— 取最后一维的长度")
print("=" * 78)
x = torch.arange(8.0)
print(f"  x = {x.tolist()}")
print(f"  x.shape       = {tuple(x.shape)}")
print(f"  x.shape[-1]   = {x.shape[-1]}      <- -1 表示'最后一维'(从右数第1个)")
print(f"  x.shape[0]    = {x.shape[0]}      <- 一维时等价")
print()
x2 = torch.arange(24.0).reshape(2, 3, 4)
print(f"  三维例子 x2.shape = {tuple(x2.shape)}")
print(f"    x2.shape[-1] = {x2.shape[-1]}   (最后一维)")
print(f"    x2.shape[-2] = {x2.shape[-2]}   (倒数第二维)")
print(f"    x2.shape[-3] = {x2.shape[-3]}   (倒数第三维)")
print("  好处: 不管几维, -1 永远是最后一维, 代码不用改")

print("\n" + "=" * 78)
print("语法 2: // 整除运算符")
print("=" * 78)
for n in [4, 6, 8]:
    print(f"  {n} // 2 = {n // 2}     {n} / 2 = {n / 2}   ({type(n/2).__name__})")
print()
print("  为什么必须用 //:")
print("    切片语法 x[a:b] 要求 a、b 是整数")
print("    如果写成 x.shape[-1] / 2 = 4.0, 切片会报 TypeError")
try:
    x[3.0:]
except TypeError as e:
    print(f"    实测: x[3.0:] -> TypeError: {e}")

print("\n" + "=" * 78)
print("语法 3: x[..., a:b] —— 省略号(ellipsis)索引")
print("=" * 78)
print("  ... 的意思是'前面所有维度照旧, 不用写出来'")
print()
x3 = torch.arange(2 * 3 * 4).reshape(2, 3, 4)
print(f"  x3.shape = {tuple(x3.shape)}")
print(f"  x3[..., :2]   -> shape {tuple(x3[..., :2].shape)}     只切最后一维的前半")
print(f"  x3[:, :, :2]  -> shape {tuple(x3[:, :, :2].shape)}     完全等价")
print(f"  内容相同 ? {torch.equal(x3[..., :2], x3[:, :, :2])}")
print()
print("  对比 4 维张量(就是 q 的形状):")
q = torch.arange(2 * 4 * 6 * 8).reshape(2, 4, 6, 8)
print(f"    q.shape = {tuple(q.shape)}")
print(f"    q[..., :4]         -> {tuple(q[..., :4].shape)}      用 ... 只需改一处")
print(f"    q[:, :, :, :4]     -> {tuple(q[:, :, :, :4].shape)}  手动写要写 4 个冒号")
print("  -> ... 的核心价值: 维度数变了代码不用改")

print("\n" + "=" * 78)
print("语法 4: x[..., h:] 和 x[..., :h] —— 后半 / 前半")
print("=" * 78)
h = x.shape[-1] // 2
print(f"  x = {x.tolist()}   (长度 8, h = 8//2 = {h})")
print(f"  x[..., {h}:]   = {x[..., h:].tolist()}      <- 下标 {h} 到末尾(后半)")
print(f"  x[..., :{h}]   = {x[..., :h].tolist()}      <- 下标 0 到 {h-1}(前半)")
print()
print("  切片规则: [起点:终点], 含起点不含终点, 省略起点默认 0, 省略终点默认到末尾")

print("\n" + "=" * 78)
print("语法 5: -x[...] —— 一元负号(逐元素取负)")
print("=" * 78)
part2 = x[..., h:]
print(f"  x[..., {h}:]      = {part2.tolist()}")
print(f"  -x[..., {h}:]     = {(-part2).tolist()}      <- 每个元素变号")
print(f"  等价于 * -1 ? {torch.equal(-part2, part2 * -1)}")
print(f"  运算优先级: -x[...] 中负号作用在索引结果上, 不加括号也对")
print(f"  实测 torch.equal(-x[..., {h}:], -(x[..., {h}:])) = {torch.equal(-x[..., h:], -(x[..., h:]))}")

print("\n" + "=" * 78)
print("语法 6: (a, b) —— 圆括号创建元组")
print("=" * 78)
a = torch.tensor([1.0, 2.0])
b = torch.tensor([3.0, 4.0])
tup = (a, b)
print(f"  (a, b) 的类型   = {type(tup).__name__}")
print(f"  长度            = {len(tup)}")
print(f"  第一个元素      = {tup[0].tolist()}")
print(f"  逗号才是关键: (a) 不是元组, 是 a 本身; (a,) 才是单元组")
print(f"    实测 type((a))  = {type((a)).__name__}")
print(f"    实测 type((a,)) = {type((a,)).__name__}")

print("\n" + "=" * 78)
print("语法 7: torch.cat(..., dim=-1) —— 沿最后一维拼接")
print("=" * 78)
r = torch.cat((-x[..., h:], x[..., :h]), dim=-1)
print(f"  第一段 -x[..., {h}:] = {(-x[..., h:]).tolist()}")
print(f"  第二段  x[..., :{h}] = {x[..., :h].tolist()}")
print(f"  拼接结果             = {r.tolist()}")
print(f"  形状                  {tuple(x.shape)} -> {tuple(r.shape)}   长度不变, 只是顺序变了")

print("\n" + "=" * 78)
print("语法 8: 圆括号换行(隐式续行)")
print("=" * 78)
print("  return torch.cat(")
print("      (...),          <- 只要还在括号里, 就可以换行")
print("      dim=-1")
print("  )")
print("  不需要反斜杠 \\ 续行符")

print("\n" + "=" * 78)
print("完整效果演示")
print("=" * 78)


def rotate_half_verbose(x):
    h = x.shape[-1] // 2
    first_half = x[..., :h]
    second_half = x[..., h:]
    return torch.cat((-second_half, first_half), dim=-1)


for vec in [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]:
    v = torch.tensor(vec)
    print(f"  x           = {v.tolist()}")
    print(f"  rotate_half = {rotate_half(v).tolist()}")
    print(f"  一致(与展开版) ? {torch.equal(rotate_half(v), rotate_half_verbose(v))}")
    print()

print("  多维情况(4 维 q):")
q = torch.arange(2 * 4 * 6 * 8).reshape(2, 4, 6, 8).float()
print(f"    q.shape            = {tuple(q.shape)}")
print(f"    rotate_half(q).shape = {tuple(rotate_half(q).shape)}   <- 形状不变")
print(f"    q[0,0,0]           = {q[0,0,0].tolist()}")
print(f"    rotate_half(q)[0,0,0] = {rotate_half(q)[0,0,0].tolist()}")

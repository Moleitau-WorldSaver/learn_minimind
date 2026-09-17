import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import torch


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    )
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    )
    return q_embed, k_embed


print("=" * 76)
print("语法 1: 嵌套函数定义(函数里定义函数)")
print("=" * 76)
print("  def outer(...):")
print("      def inner(x):  ...     <- 内层函数只在外层作用域可见")
print("  特点: 每次调用 apply_rotary_pos_emb 都会重新创建一次 rotate_half")
print("  好处: 定义在用的地方, 不用污染模块命名空间")


def outer_demo():
    def inner(x):
        return x * 2
    return inner(5)


print(f"  实测: outer_demo() = {outer_demo()}")
try:
    inner(1)
except NameError as e:
    print(f"  外部访问 inner -> NameError: {e}")

print("\n" + "=" * 76)
print("语法 2: x[..., a:b] 省略号索引")
print("=" * 76)
x = torch.arange(2 * 3 * 4).reshape(2, 3, 4)
print(f"  x.shape = {tuple(x.shape)}")
print("  x[..., :2]      -> 前面所有维度照旧, 只切最后一维的前 2 个")
print(f"     shape {tuple(x[..., :2].shape)}")
print("  x[..., 2:]      -> 最后一维从下标 2 到末尾")
print(f"     shape {tuple(x[..., 2:].shape)}")
print("  x[:, :, :2]     -> 完全等价, 但要写明每一维")
print(f"     shape {tuple(x[:, :, :2].shape)}, 内容一致 {torch.equal(x[..., :2], x[:, :, :2])}")
print("  ... 的好处: 不管张量几维都不用改代码(4维的 q 也能用)")

print("\n" + "=" * 76)
print("语法 3: x.shape[-1] 取最后一维的长度")
print("=" * 76)
print(f"  x.shape       = {tuple(x.shape)}")
print(f"  x.shape[-1]   = {x.shape[-1]}     <- -1 表示最后一维")
print(f"  x.shape[-2]   = {x.shape[-2]}")
print(f"  x.shape[-1]//2= {x.shape[-1] // 2}   <- // 整除, 保证是整数(切片要求)")
print(f"  x.shape[-1]/2 = {x.shape[-1] / 2}    <- / 是浮点, 不能用于切片!")

print("\n" + "=" * 76)
print("语法 4: -x[...] 一元负号 (逐元素取负)")
print("=" * 76)
v = torch.tensor([1.0, -2.0, 3.0])
print(f"  v      = {v.tolist()}")
print(f"  -v     = {(-v).tolist()}      <- 每个元素变号")
print(f"  -v 等价于 v * -1 ? {torch.equal(-v, v * -1)}")
print("  注意: 这是 torch 的逐元素负号运算, 返回新张量, v 本身不变")
print(f"  验证 v 没变: {v.tolist()}")

print("\n" + "=" * 76)
print("语法 5: torch.cat((a, b), dim=-1) 用元组的第一参数")
print("=" * 76)
a, b = torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]])
print(f"  torch.cat((a, b), dim=-1) = {torch.cat((a, b), dim=-1).tolist()}")
print(f"  注意圆弧括号里包的是 (a, b) 一个元组, dim=-1 是第二个位置参数之外的关键字参数")

print("\n" + "=" * 76)
print("语法 6: cos.unsqueeze(dim) 增加一个长度为1的维度")
print("=" * 76)
cos = torch.randn(6, 64)          # (seq_len, head_dim)
print(f"  cos.shape = {tuple(cos.shape)}")
for d in [0, 1, 2, -1]:
    print(f"  cos.unsqueeze({d:>2}).shape = {str(tuple(cos.unsqueeze(d).shape)):<20}"
          f" (在第 {d} 位插入一个 1)")
print("\n  unsqueeze 不复制数据, 只改形状视图(内存零开销)")

print("\n" + "=" * 76)
print("语法 7: unsqueeze_dim 作为参数来控制插在哪一维")
print("=" * 76)
q = torch.randn(2, 4, 6, 64)      # (batch, heads, seq, head_dim)
print(f"  q.shape          = {tuple(q.shape)}")
print(f"  cos.shape        = {tuple(cos.shape)}   <- (seq, head_dim), 2 维")
print()
for ud in [0, 1, 2]:
    c = cos.unsqueeze(ud)
    try:
        r = q * c
        print(f"  unsqueeze_dim={ud}: cos -> {str(tuple(c.shape)):<18} q*cos -> OK {tuple(r.shape)}")
    except RuntimeError as e:
        print(f"  unsqueeze_dim={ud}: cos -> {str(tuple(c.shape)):<18} q*cos -> 失败 {str(e)[:44]}")

print("\n  为什么 unsqueeze_dim=1 是对的:")
print("    q      = (batch, heads, seq, head_dim)")
print("    cos    = (      1, heads, seq, head_dim)   <- 补出 heads 和 batch 两个 1")
print("    广播:    batch 和 heads 两个维度自动展开")

print("\n" + "=" * 76)
print("语法 8: 广播(broadcasting)")
print("=" * 76)
A = torch.ones(2, 1, 6, 64)
B = torch.ones(1, 4, 6, 64)
print(f"  A.shape = {tuple(A.shape)}")
print(f"  B.shape = {tuple(B.shape)}")
print(f"  A * B   -> {tuple((A * B).shape)}   <- 长度为 1 的维度被'拉伸'到对方长度")
print(f"  A + 5   -> {tuple((A + 5).shape)}   <- 标量也能广播")

print("\n" + "=" * 76)
print("语法 9: 圆括号换行(隐式续行)")
print("=" * 76)
total = (1 +
         2 +
         3)
print(f"  括号内的表达式可以跨行: total = {total}")
print("  这就是 q_embed = (...) + (...) 能写成多行的原因")
print("  注意: 运算符放在行尾或行首都行, 只要在括号里")

print("\n" + "=" * 76)
print("语法 10: position_ids=None 默认参数(本函数里未被使用)")
print("=" * 76)
import inspect
sig = inspect.signature(apply_rotary_pos_emb)
print(f"  签名: {sig}")
print("  position_ids 是'兼容性占位参数': 保持和其他实现(HF transformers)签名一致")
print("  本函数体内没有用到它 -> 传入也无效果(静默忽略)")
print("  这种参数常见于'统一接口'场景, 调用方可以统一写法")

print("\n" + "=" * 76)
print("整体验证: 函数真的实现了旋转")
print("=" * 76)
torch.manual_seed(0)
batch, heads, seq, dim = 1, 1, 4, 64
q = torch.randn(batch, heads, seq, dim)
k = torch.randn(batch, heads, seq, dim)

# 构造 cos/sin 表 (seq, dim)
rope_base = 1e6
inv = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
ang = torch.outer(torch.arange(seq), inv)
cos_t = torch.cat([torch.cos(ang)] * 2, dim=-1)
sin_t = torch.cat([torch.sin(ang)] * 2, dim=-1)

q2, k2 = apply_rotary_pos_emb(q, k, cos_t, sin_t)
print(f"  q.shape  = {tuple(q.shape)}")
print(f"  q2.shape = {tuple(q2.shape)}   <- 形状不变")
for i in range(seq):
    n1, n2 = q[0, 0, i].norm().item(), q2[0, 0, i].norm().item()
    print(f"  位置{i}: |q|={n1:.6f}  |q_rot|={n2:.6f}  相等 {abs(n1-n2)<1e-5}")

print("\n  位置0 应该是'不旋转'(角度全0):")
print(f"    q[0,0,0] == q2[0,0,0] ? {torch.allclose(q[0,0,0], q2[0,0,0], atol=1e-6)}")

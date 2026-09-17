import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import torch

x = torch.arange(2 * 3 * 4).reshape(2, 3, 4)
print(f"x.shape = {tuple(x.shape)}   (2, 3, 4)\n")

print("=" * 76)
print("1) 合法性: x[..., 1:2] 完全合法")
print("=" * 76)
r = x[..., 1: 2]
print(f"  x[..., 1:2] -> shape {tuple(r.shape)}")
print(f"  和 x[:, :, 1:2] 等价 ? {torch.equal(r, x[:, :, 1:2])}")
print()
print("  切片规则 [起点:终点] 含起点不含终点, 所以 1:2 只取下标 1 这一个元素")
print(f"  实测内容 x[0,0] = {x[0,0].tolist()},  x[0,0,1:2] = {x[0,0,1:2].tolist()}")

print("\n" + "=" * 76)
print("2) 为什么这样写: 切片 vs 整数, 维度是否保留")
print("=" * 76)
print(f"  {'写法':<18}{'结果 shape':<16}{'最后一维':<12}{'是否保留维度'}")
print("  " + "-" * 62)
for expr, fn in [("x[..., 1]", lambda t: t[..., 1]),
                 ("x[..., 1:2]", lambda t: t[..., 1:2]),
                 ("x[..., 1:3]", lambda t: t[..., 1:3])]:
    rr = fn(x)
    last = rr.shape[-1] if rr.dim() == 3 else "—"
    keep = "否(降维)" if rr.dim() == 2 else "是(保留)"
    print(f"  {expr:<18}{str(tuple(rr.shape)):<16}{str(last):<12}{keep}")

print("\n  核心区别:")
print("    x[..., 1]     -> 整数索引, 那一维'消失' -> (2, 3)")
print("    x[..., 1:2]   -> 切片索引, 那一维'保留' -> (2, 3, 1)  长度 1")
print(f"  数值内容相同 ? {torch.equal(x[..., 1], x[..., 1:2].squeeze(-1))}")

print("\n" + "=" * 76)
print("3) : 的三种常见形式(都合法)")
print("=" * 76)
forms = [("1:2", "取下标 1(长度1)"),
         ("1:3", "取下标 1,2(长度2)"),
         ("1:",  "从 1 到末尾"),
         (":2",  "从开头到下标 1"),
         (":",   "全部"),
         ("::2", "步长 2"),
         ("1:4:2", "1 到 3, 步长 2")]
for f, desc in forms:
    code = f"x[..., {f}]"
    rr = eval(code)
    print(f"  {code:<18}-> shape {str(tuple(rr.shape)):<12}  {desc}")

print("\n" + "=" * 76)
print("4) 补一个: 起点终点可以相等吗(空切片)")
print("=" * 76)
print(f"  x[..., 2:2]   -> shape {tuple(x[..., 2:2].shape)}   <- 长度为 0, 合法但通常没用")
print(f"  元素个数 {x[..., 2:2].numel()}")
print(f"  x[..., 5:9]   -> shape {tuple(x[..., 5:9].shape)}   <- 越界也是空, 不报错(切片特性)")

print("\n" + "=" * 76)
print("5) 负数下标")
print("=" * 76)
print(f"  x[..., -1:]   -> shape {tuple(x[..., -1:].shape)}   <- 最后一个(保留维度)")
print(f"  x[..., -2:]   -> shape {tuple(x[..., -2:].shape)}   <- 最后两个")
print(f"  x[..., -1]    -> shape {tuple(x[..., -1].shape)}   <- 最后一个(降维)")

print("\n" + "=" * 76)
print("6) 在 RoPE 里的实际用途: 取一半")
print("=" * 76)
q = torch.arange(2 * 4 * 6 * 8).reshape(2, 4, 6, 8).float()
print(f"  q.shape = {tuple(q.shape)}   head_dim = 8")
print(f"  q[..., :4]    -> {tuple(q[..., :4].shape)}   前半")
print(f"  q[..., 4:]    -> {tuple(q[..., 4:].shape)}   后半")
print(f"  q[..., 3:4]   -> {tuple(q[..., 3:4].shape)}   只要第 3 个, 但保留维度")
print(f"  q[..., 3]     -> {tuple(q[..., 3].shape)}   只要第 3 个, 维度消失")

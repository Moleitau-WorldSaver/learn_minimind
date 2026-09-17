import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import torch

v = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
print(f"v = {v.tolist()}   下标 0~7\n")

print("=" * 78)
print("唯一规则: [起点 : 终点]  含起点, 不含终点")
print("=" * 78)
print("  丢掉的是: 下标 < 起点的部分, 和 下标 >= 终点的部分")
print("  保留的是: 起点 <= 下标 < 终点")
print()
print(f"  {'写法':<12}{'起点':<8}{'终点':<10}{'保留的下标':<22}{'结果'}")
print("  " + "-" * 70)
rows = [
    ("v[4:]",  "4", "末尾", "4,5,6,7"),
    ("v[:4]",  "0", "4",    "0,1,2,3"),
    ("v[2:5]", "2", "5",    "2,3,4"),
    ("v[:]",   "0", "末尾", "0~7"),
    ("v[3:3]", "3", "3",    "空"),
]
for expr, s, e, keep in rows:
    print(f"  {expr:<12}{s:<8}{e:<10}{keep:<22}{eval(expr).tolist()}")

print("\n" + "=" * 78)
print("你的说法核对: a: 在 : 前 = 砍掉前面")
print("=" * 78)
print(f"  v[4:] = {v[4:].tolist()}")
print("         └─ 下标 0,1,2,3 被丢掉了(前面 4 个)")
print("         保留 4,5,6,7")
print()
print("  ✅ 正确: : 前面的数字 = 起点, 起点之前的内容被砍掉")

print("\n" + "=" * 78)
print("对称的一条: :b 在 : 后 = 砍掉后面")
print("=" * 78)
print(f"  v[:4] = {v[:4].tolist()}")
print("         └─ 下标 4,5,6,7 被丢掉了(后面 4 个)")
print("         保留 0,1,2,3")
print()
print("  : 后面的数字 = 终点, 终点及之后的内容被砍掉")

print("\n" + "=" * 78)
print("速记: 冒号是'切刀', 数字在左边砍左, 在右边砍右")
print("=" * 78)
print("        [ 起点 :  终点 ]")
print("            ↓        ↓")
print("        丢掉左边   丢掉右边")
print()
print("  位置记忆:")
print("    : 左边的数字 -> 从这里开始  -> 前面的被砍")
print("    : 右边的数字 -> 到这里为止  -> 后面的被砍")
print("    两边都有     -> 只保留中间")
print("    两边都没有   -> 什么都不砍")

print("\n" + "=" * 78)
print("负数: 从右边数")
print("=" * 78)
print(f"  v[-2:]  = {v[-2:].tolist()}      <- 最后 2 个")
print(f"  v[:-2]  = {v[:-2].tolist()}   <- 砍掉最后 2 个")
print(f"  v[-5:-2]= {v[-5:-2].tolist()}      <- 从倒数第5到倒数第3")

print("\n" + "=" * 78)
print("回到 RoPE 的两句")
print("=" * 78)
vec = torch.arange(8.0)
h = 4
print(f"  x = {vec.tolist()}    h = {h}")
print()
print(f"  x[..., :{h}]  = {vec[:h].tolist()}   <-  : 在 {h} 左边? 不, {h} 在 : 右边 -> 砍右")
print(f"                 保留下标 0~{h-1}")
print()
print(f"  x[..., {h}:]  = {vec[h:].tolist()}   <-  {h} 在 : 左边 -> 砍左")
print(f"                 保留下标 {h}~7")
print()
print("  所以两句合起来正好把 8 个元素分成两半, 不重不漏:")
print(f"    前半 {vec[:h].tolist()} + 后半 {vec[h:].tolist()} = {torch.cat((vec[:h], vec[h:])).tolist()}  (= 原数组)")
print(f"    验证: {torch.equal(torch.cat((vec[:h], vec[h:])), vec)}")

print("\n" + "=" * 78)
print("步长(第三个位置, 进阶)")
print("=" * 78)
print("  完整语法: [起点 : 终点 : 步长]")
print(f"    v[::2]   = {v[::2].tolist()}   每隔一个取")
print(f"    v[1::2]  = {v[1::2].tolist()}   从 1 开始每隔一个")
print(f"    v[::-1]  = {v[::-1].tolist()}   倒序(步长 -1)")

import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import torch

a = torch.tensor([[1, 2], [3, 4]])       # shape (2, 2)
b = torch.tensor([[5, 6], [7, 8]])       # shape (2, 2)
c = torch.tensor([[9, 10]])              # shape (1, 2)
d = torch.tensor([[11, 12, 13], [14, 15, 16]])  # shape (2, 3)

print("a =", a.tolist(), "shape", tuple(a.shape))
print("b =", b.tolist(), "shape", tuple(b.shape))
print("c =", c.tolist(), "shape", tuple(c.shape))
print("d =", d.tolist(), "shape", tuple(d.shape))

print("\n1) cat((a, b), dim=0) 沿第0维(行方向)拼接")
print("   -> shape", tuple(torch.cat((a, b), dim=0).shape))
print("   ", torch.cat((a, b), dim=0).tolist())

print("\n2) cat((a, b), dim=1) 沿第1维(列方向)拼接")
print("   -> shape", tuple(torch.cat((a, b), dim=1).shape))
print("   ", torch.cat((a, b), dim=1).tolist())

print("\n3) 可以拼 3 个以上,非拼接维必须完全一致")
print("   cat((a, b, c), dim=0) -> shape", tuple(torch.cat((a, b, c), dim=0).shape), torch.cat((a, b, c), dim=0).tolist())

print("\n4) 位置参数形式(dim 可省,默认 0)")
print("   torch.cat((a, b))        ->", torch.cat((a, b)).tolist())
print("   torch.cat((a, b), 1)     ->", torch.cat((a, b), 1).tolist())

print("\n5) 负数维度(和 Python 索引一样从后往前数)")
print("   cat((a,b), dim=-1) ->", torch.cat((a, b), dim=-1).tolist())

print("\n6) 出错情况")
try:
    torch.cat((a, d), dim=0)
except RuntimeError as e:
    print(f"   cat((a, d), dim=0) -> RuntimeError: {str(e)[:95]}...")
try:
    torch.cat((a, b), dim=2)
except IndexError as e:
    print(f"   cat((a, b), dim=2) -> IndexError: {str(e)[:70]}")

print("\n7) 一维张量(最常用于拼接 cos/sin)")
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([1.0, 2.0, 3.0])
print("   cat((x, y), dim=-1) ->", torch.cat((x, y), dim=-1).tolist(), "shape", tuple(torch.cat((x, y), dim=-1).shape))

print("\n8) 广播? 不支持! 形状必须对齐")
e = torch.tensor([[1], [2]])   # (2,1)
try:
    torch.cat((a, e), dim=1)
except RuntimeError as err:
    print(f"   cat((a(2,2), e(2,1)), dim=1) -> RuntimeError: {str(err)[:80]}...")
print("   注意:这需要的是 torch.cat 的兄弟函数? 不,是形状不匹配 -> 该用 a + e 的广播,或先把 e 补齐")

print("\n9) 对比 stack(会新增一维)")
print("   torch.cat((a, b), dim=0).shape   =", tuple(torch.cat((a, b), dim=0).shape), " (2+2=4 行)")
print("   torch.stack((a, b), dim=0).shape =", tuple(torch.stack((a, b), dim=0).shape), " (新增维度!)")

print("\n10) 你项目里的真实用法(apply_rotary 的 cos/sin 复制)")
dim = 8
cos_part = torch.tensor([0.54, 0.80, 0.91, 0.99])
print("   cos[:4]        =", cos_part.tolist())
print("   cat([c,c], -1) =", torch.cat([cos_part, cos_part], dim=-1).tolist(), " <- 4->8 维,才能和 8 维 q 逐元素相乘")

print("\n11) out= 参数(关键字)")
buf = torch.empty(4, 2)
torch.cat((a, b), dim=0, out=buf)
print("   buf =", buf.tolist())

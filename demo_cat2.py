import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import torch

a = torch.tensor([[1, 2], [3, 4]])   # (2,2)
e = torch.tensor([[1], [2]])          # (2,1)

print("a =", a.tolist(), "shape", tuple(a.shape))
print("e =", e.tolist(), "shape", tuple(e.shape))

print("\ncat((a, e), dim=1): 非拼接维(dim=0)都是 2,匹配 -> 应该成功")
try:
    r = torch.cat((a, e), dim=1)
    print("   -> shape", tuple(r.shape), r.tolist())
except RuntimeError as err:
    print("   RuntimeError:", err)
print("   (拼接维长度可以不同: 2+1=3;非拼接维必须相同)")

print("\ndim=0 时非拼接维是 dim=1: a 是 2,e 是 1 -> 不匹配")
try:
    torch.cat((a, e), dim=0)
except RuntimeError as err:
    print("   RuntimeError:", str(err)[:100])

print("\n--- 形状规则总结 ---")
print("所有张量除了 dim 指定的那一维,其余维度必须完全相同")
print("cat((a,b), dim=0): 要求 a.shape[1:] == b.shape[1:]")
print("cat((a,b), dim=1): 要求 a.shape[0] == b.shape[0]")

print("\n--- 对照:stack 要求所有形状完全相同 ---")
try:
    torch.stack((a, e), dim=0)
except RuntimeError as err:
    print("   stack((a,e)) -> RuntimeError:", str(err)[:85])
print("   stack((a, torch.tensor([[5,6],[7,8]]))).shape =",
      tuple(torch.stack((a, torch.tensor([[5, 6], [7, 8]]))).shape))

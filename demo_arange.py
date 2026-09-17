import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import torch

print("1) 基本三种形式")
print("   arange(5)        =", torch.arange(5))
print("   arange(2, 6)     =", torch.arange(2, 6))
print("   arange(0, 10, 3) =", torch.arange(0, 10, 3))

print("\n2) 浮点步长")
print("   arange(0, 1, 0.25) =", torch.arange(0, 1, 0.25))
print("   arange(1, 2, 0.3)  =", torch.arange(1, 2, 0.3), " <- 注意末尾不是 1.9!")

print("\n3) 降序(负步长)")
print("   arange(5, 0, -1) =", torch.arange(5, 0, -1))
try:
    print("   arange(5, 0, 1)  =", torch.arange(5, 0, 1))
except RuntimeError as e:
    print(f"   arange(5, 0, 1)  -> RuntimeError: {e}")

print("\n4) dtype 推断规则(最重要的坑)")
for a in [torch.arange(5), torch.arange(0, 1, 0.25), torch.arange(5, dtype=torch.float32)]:
    print(f"   {str(a):45s} dtype={a.dtype}")

print("\n5) 和 Python range 的对比")
print("   range(5)          =", list(range(5)))
print("   range(0, 1, 0.25) -> TypeError: range 只接受整数步长")
print("   torch.arange 可以用小数步长:", torch.arange(0, 1, 0.25).tolist())

print("\n6) 你 model.py:105 和 :140 的两种用法")
dim = 64
print("   切片索引用法: torch.arange(0, dim, 2)[:dim//2]")
print("     =", torch.arange(0, dim, 2)[: dim // 2][:8].tolist(), f"... 共 {len(torch.arange(0, dim, 2)[:dim//2])} 个, dtype={torch.arange(0, dim, 2)[:dim//2].dtype}")
print("   位置索引用法: torch.arange(dim//2).float()")
print("     =", torch.arange(dim // 2).float()[:8].tolist(), f"... 共 {dim//2} 个, dtype={torch.arange(dim//2).float().dtype}")

print("\n7) 为什么要 .float() —— 整数张量做除法会出错")
idx_int = torch.arange(4)
print("   arange(4) / 3        =", idx_int / 3)
print("   arange(4).float() / 3 =", idx_int.float() / 3)
print("   arange(4) ** -1      -> ", end="")
try:
    print(idx_int ** -1)
except Exception as e:
    print(f"{type(e).__name__}: {str(e)[:60]}")

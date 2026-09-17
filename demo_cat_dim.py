import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import torch

a = torch.zeros(1, 2, 5)
b = torch.zeros(1, 2, 5)
print("a.shape =", tuple(a.shape), "  b.shape =", tuple(b.shape))

print("\nshape 的每个数字对应的 dim 编号:")
print("  shape[0]=1  ->  dim=0  (也可写 dim=-3)")
print("  shape[1]=2  ->  dim=1  (也可写 dim=-2)")
print("  shape[2]=5  ->  dim=2  (也可写 dim=-1)")
print()

print(f"{'dim':<6}{'等价负数':<10}{'拼接后 shape':<18}{'哪个数字变了'}")
print("-" * 62)
for d, neg in [(0, -3), (1, -2), (2, -1)]:
    r = torch.cat((a, b), dim=d)
    sizes = list(a.shape)
    sizes[d] = sizes[d] * 2
    print(f"{d:<6}{neg:<10}{str(tuple(r.shape)):<18}shape[{d}]: {a.shape[d]} -> {r.shape[d]}")

print("\n--- 负数形式等价性验证 ---")
for d, neg in [(0, -3), (1, -2), (2, -1)]:
    same = torch.equal(torch.cat((a, b), dim=d), torch.cat((a, b), dim=neg))
    print(f"  cat(dim={d}) == cat(dim={neg}) ? {same}")

print("\n--- dim=2 时,其他维度必须一致,只有 5 那一维可以不同 ---")
x = torch.zeros(1, 2, 5)
y = torch.zeros(1, 2, 3)      # 只有第 2 维不同
z = torch.zeros(1, 2, 7)
print("  x=(1,2,5)  y=(1,2,3)  z=(1,2,7)")
print("  cat((x,y,z), dim=2) ->", tuple(torch.cat((x, y, z), dim=2).shape), " (5+3+7=15)")

print("\n  dim=1 时,第 2 维必须相同:")
try:
    torch.cat((x, y), dim=1)
except RuntimeError as e:
    print("   cat((x,y), dim=1) -> RuntimeError:", str(e)[:80])

print("\n--- dim 越界的报错 ---")
for bad in [3, -4]:
    try:
        torch.cat((a, b), dim=bad)
    except IndexError as e:
        print(f"  cat(dim={bad}) -> IndexError: {str(e)[:75]}")

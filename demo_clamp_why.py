import math
import torch

dim, factor = 64, 16
low, high = 7, 15
idx = torch.arange(dim // 2, dtype=torch.float32)

# 线性函数(未夹紧)
linear = (idx - low) / max(high - low, 0.001)
# 夹紧后的 ramp
ramp = torch.clamp(linear, 0, 1)

print("维度索引 i =", idx[:20].tolist(), "...")
print("未夹紧 linear =", [round(v, 2) for v in linear.tolist()])
print("夹紧后 ramp   =", [round(v, 2) for v in ramp.tolist()])

print("\n--- ramp=0 处的效果(低频维度 i=0) ---")
print("ramp=0 时权重 = 1-0 + 0/16 =", 1 - 0 + 0 / factor, "→ 频率 = 原频率 ×", 1 - 0 + 0 / factor)
print("ramp=1 时权重 = 1-1 + 1/16 =", 1 - 1 + 1 / factor, "→ 频率 = 原频率 ×", 1 - 1 + 1 / factor)

print("\n--- 不加 clamp 会发生什么(取 i=0 和 i=31) ---")
for i in [0, 31]:
    w_bad = 1 - linear[i] + linear[i] / factor
    w_ok = 1 - ramp[i] + ramp[i] / factor
    print(f"i={i:2d}: linear={linear[i]:7.3f}  不加clamp权重={w_bad:7.4f}  |  加clamp权重={w_ok:.4f}")
print("\n(factor=16,正确权重只可能落在 [1/16=0.0625, 1.0] 之间)")

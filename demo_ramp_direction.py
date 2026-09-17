import math
import torch

dim, factor, rope_base, orig_max = 64, 16, 1e6, 2048

inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))

# 参考实现(minimind / DeepSeek):beta_fast -> low, beta_slow -> high
low = max(math.floor(inv_dim(32)), 0)
high = min(math.ceil(inv_dim(1)), dim // 2 - 1)
print(f"low(来自beta_fast=32) = {low},  high(来自beta_slow=1) = {high}")

idx = torch.arange(dim // 2, dtype=torch.float32)
ramp = torch.clamp((idx - low) / max(high - low, 0.001), 0, 1)

# 每个维度的波长(用 rope_base=1e6 时的 inv_freq 反推)
inv_freq = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim))
wavelength = 2 * math.pi / inv_freq

print(f"\n{'i':>2} | {'波长λ':>12} | {'λ<orig_max?':>11} | {'ramp':>5} | 结论")
print("-" * 72)
for i in [0, 3, 6, 10, 20, 31]:
    short = "短(高频)" if wavelength[i] < orig_max else "长(低频)"
    print(f"{i:>2} | {wavelength[i].item():>12.1f} | {short:>11} | {ramp[i].item():>5.2f} | "
          f"{'高频 → ramp=1' if ramp[i] > 0.99 else ('低频 → ramp=0' if ramp[i] < 0.01 else '过渡区')}")

print("\n关键检查:")
print(f"  i=0  (波长最短/最高频): ramp={ramp[0].item():.2f}  → {'保留原频率' if ramp[0] < 0.01 else '被插值'}")
print(f"  i=31 (波长最长/最低频): ramp={ramp[31].item():.2f}  → {'保留原频率' if ramp[31] > 0.99 else '被插值'}")

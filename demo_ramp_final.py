import math
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import torch

dim, factor, rope_base, orig_max = 64, 16, 1e6, 2048
inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
inv_freq = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
idx = torch.arange(dim // 2, dtype=torch.float32)


def build(low, high):
    return torch.clamp((idx - low) / max(high - low, 0.001), 0, 1)


low_ok = max(math.floor(inv_dim(32)), 0)          # 5  <- beta_fast
high_ok = min(math.ceil(inv_dim(1)), dim // 2 - 1)  # 14 <- beta_slow
low_bad = inv_dim(math.floor(inv_dim(32)))         # 9.68
high_bad = inv_dim(math.ceil(inv_dim(1)))          # 7.29

print(f"正确: low={low_ok}, high={high_ok}   (low < high)")
print(f"你的: low={low_bad:.2f}, high={high_bad:.2f}   (low > high  -> 方向反了)\n")

for name, low, high in [("正确版", low_ok, high_ok), ("你的版", low_bad, high_bad)]:
    ramp = build(low, high)
    w = 1 - ramp + ramp / factor
    print(f"[{name}] low={low if isinstance(low,int) else round(low,2)}, high={high if isinstance(high,int) else round(high,2)}")
    print("  ramp[:18] =", [round(v, 3) for v in ramp[:18].tolist()])
    for i in [0, 5, 10, 20, 31]:
        wt = w[i].item()
        act = "原样保留(不缩放)" if abs(wt - 1) < 1e-6 else ("除以16(全插值)" if abs(wt - 1 / factor) < 1e-6 else f"混合 {wt:.3f}")
        lam = (2 * math.pi / inv_freq[i]).item()
        print(f"  i={i:2d} 波长={lam:>12.1f} ramp={ramp[i]:.3f} 权重={wt:.4f} -> {act}")
    trans = ((ramp > 0.001) & (ramp < 0.999)).sum().item()
    print(f"  过渡区维度个数 = {trans}  ({'正常,有平滑过渡' if trans > 2 else '异常,退化成阶跃!'})\n")

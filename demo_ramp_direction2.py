import math
import torch

dim, factor, rope_base, orig_max, end = 64, 16, 1e6, 2048, 32768


def build(low, high):
    idx = torch.arange(dim // 2, dtype=torch.float32)
    return torch.clamp((idx - low) / max(high - low, 0.001), 0, 1)


inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
inv_freq = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim))

# 三种 low/high 的算法
ref_low, ref_high = max(math.floor(inv_dim(32)), 0), min(math.ceil(inv_dim(1)), dim // 2 - 1)
usr_low, usr_high = inv_dim(math.floor(inv_dim(32))), inv_dim(math.ceil(inv_dim(1)))

ramp_ref = build(ref_low, ref_high)
ramp_usr = build(usr_low, usr_high)


def report(name, ramp, weight_formula):
    w = weight_formula(ramp)
    # 看 i=0(最高频,波长最短)和 i=31(最低频,波长最长)被怎么处理
    def tag(i):
        if abs(w[i].item() - 1.0) < 1e-6:
            return "原样保留(不缩放)"
        if abs(w[i].item() - 1 / factor) < 1e-6:
            return "除以16(全插值)"
        return f"混合 {w[i].item():.4f}"
    print(f"{name}")
    print(f"   low={ref_low if '参考' in name else usr_low}, high={ref_high if '参考' in name else usr_high}")
    print(f"   i=0  (最高频): ramp={ramp[0]:.3f} weight={w[0]:.4f} → {tag(0)}")
    print(f"   i=31 (最低频): ramp={ramp[31]:.3f} weight={w[31]:.4f} → {tag(31)}")
    ok = "[正确]" if "原样保留" in tag(0) and "除以16" in tag(31) else "[方向反了]"
    print(f"   → {ok}\n")


print("目标:最高频(i=0) 原样保留;最低频(i=31) 除以16(全插值)\n")

report("① 参考实现 + 公式 1-ramp+ramp/factor", ramp_ref, lambda r: 1 - r + r / factor)
report("② 你的写法 + 公式 1-ramp+ramp/factor", ramp_usr, lambda r: 1 - r + r / factor)
report("③ 你的写法 + 公式 1-(1-ramp)/factor-... 即 ramp+ (1-ramp)/factor", ramp_usr, lambda r: r + (1 - r) / factor)

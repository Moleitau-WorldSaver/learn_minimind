import torch

factor = 16
low, high = 7, 15
idx = torch.arange(32, dtype=torch.float32)
ramp = torch.clamp((idx - low) / max(high - low, 0.001), 0, 1)

# 权重:实际乘到原始频率上的系数
weight = 1 - ramp + ramp / factor

print(f"{'i':>3} | {'ramp':>6} | {'weight=1-ramp+ramp/16':>22} | {'等效倍率(相对原频率)':>20}")
print("-" * 62)
for i in [0, 7, 8, 11, 15, 20, 31]:
    print(f"{i:>3} | {ramp[i].item():>6.3f} | {weight[i].item():>22.4f} | {'原样保留' if ramp[i]==1 else ('除以16=插值' if ramp[i]==0 else '两者线性混合'):>20}")

print("\nramp 的极值对照:")
print("  ramp=0  → weight =", (1 - 0 + 0 / factor), " = 1/factor  → 频率变成 1/16(全插值)")
print("  ramp=1  → weight =", (1 - 1 + 1 / factor), "= 1        → 频率不变(全保留)")
print("\n注意:weight 才是'乘到频率上的倍率';ramp 只是它的一个调节参数。")
print("ramp 越大 → weight 越小 → 越接近'原频率'。方向别记反。")

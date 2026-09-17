import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import torch

# 模拟真实的 YaRN 输出形状:freqs 是角度矩阵 [end, dim//2]
dim, end = 8, 3          # head_dim=8, 序列长度=3(缩小便于观察)
angles = torch.arange(end * (dim // 2), dtype=torch.float32).reshape(end, dim // 2) * 0.1

print(f"freqs(角度矩阵) shape={tuple(angles.shape)}   dim={dim}, end={end}")
print("  freqs =")
for i, row in enumerate(angles.tolist()):
    print(f"    行{i}: {[round(v,2) for v in row]}")

cos = torch.cos(angles)
print(f"\ntorch.cos(freqs) shape={tuple(cos.shape)}   <- 还是 (end, dim//2) = ({end}, {dim//2})")
print("  cos =")
for i, row in enumerate(cos.tolist()):
    print(f"    行{i}: {[round(v,3) for v in row]}")

out = torch.cat([cos, cos], dim=-1)
print(f"\ntorch.cat([cos, cos], dim=-1) shape={tuple(out.shape)}   <- 最后一维 4 -> 8")
print("  out =")
for i, row in enumerate(out.tolist()):
    print(f"    行{i}: {[round(v,3) for v in row]}")
print(f"\n  注意: 每行是 [c0,c1,c2,c3, c0,c1,c2,c3] —— 原始 4 个值原样复制了一遍")
print(f"  元素个数: {cos.numel()} -> {out.numel()}   (每个元素被复制成 2 份)")

print("\n--- '拼接到里面' 的说法要纠正 ---")
print(f"  dim=-1 就是 dim={out.dim()-1}(最后一维), 长度 {dim//2} -> {dim}")
print("  它不是'插入到第一个张量里', 而是: 新张量的最后一维 = 第一个的长度 + 第二个的长度")
print("  两个张量在这个维度上是'首尾相接、并列', 没有嵌套关系")

print("\n--- 为什么需要这一步 ---")
print(f"  q/k 的最后一维是 head_dim = {dim}")
print(f"  但 freqs 只有 dim//2 = {dim//2} 个角度(每 2 个维度共享 1 个角度)")
print(f"  所以要把角度复制一份, 才够 {dim} 个, 才能和 q/k 逐元素相乘")

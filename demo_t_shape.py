import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import torch

rope_base, dim, end = 1e6, 64, 8   # dim=64 是 head_dim,end=8 是可处理的最大长度

freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
t = torch.arange(end)
angles = torch.outer(t, freqs)

print(f"dim = {dim}, end = {end}\n")
print(f"{'变量':<12}{'shape':<14}{'长度由谁决定':<20}内容")
print("-" * 72)
print(f"{'freqs':<12}{str(tuple(freqs.shape)):<14}{'dim // 2 = ' + str(dim//2):<20}{'每对维度的角频率'}")
print(f"{'t':<12}{str(tuple(t.shape)):<14}{'end = ' + str(end):<20}{'位置序号 0..' + str(end-1)}")
print(f"{'angles':<12}{str(tuple(angles.shape)):<14}{'(end, dim//2)':<20}{'每个位置每对维度的角度'}")
print(f"{'polar':<12}{str(tuple(torch.polar(torch.ones_like(angles), angles).shape)):<14}{'(end, dim//2)':<20}{'复数旋转因子'}")

print("\n--- 换 end,看 t 怎么变(注意 dim 不动) ---")
for e in [4, 8, 32768]:
    print(f"  end={e:<7} t.shape={tuple(torch.arange(e).shape)}   angles.shape={tuple(torch.outer(torch.arange(e), freqs).shape)}")

print(f"\n--- 关键: t 的元素个数 = end,和 dim 无关 ---")
print(f"  dim=64 时 freqs 有 {len(freqs)} 个元素(因为 dim//2 = 32)")
print(f"  t 有 {end} 个元素(因为 end = {end})")
print(f"  两者恰好都是 32 只是巧合(end=8 时 t 就只有 8 个)")

print("\n--- 真正的 64 维在哪 ---")
print(f"  旋转表 angles 的第二个维度 = dim//2 = {dim//2}(不是 64!)")
print(f"  因为 RoPE 每 2 个维度共享一个角度:64 维 -> 32 个角度")
print(f"  最后 apply_rotary 时,cos/sin 会复制一份还原成 {dim} 维的 q/k")

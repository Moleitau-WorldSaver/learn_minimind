import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import torch

dim, end = 64, 6
rope_base, factor = 1e6, 16
attn_factor = 0.1 * torch.log(torch.tensor(float(factor))).item() + 1.0

inv_freq = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
angles = torch.outer(torch.arange(end), inv_freq)

freqs_cos = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1) * attn_factor
freqs_sin = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1) * attn_factor

print("freqs_cos:")
print(f"  shape={tuple(freqs_cos.shape)}  dtype={freqs_cos.dtype}")
print(f"  含义: 形状 [序列长度 end, head_dim], 第 t 行是'位置 t 专用的 cos 表'")
print(f"  值域: [{freqs_cos.min():.4f}, {freqs_cos.max():.4f}]  (attn_factor={attn_factor:.4f})")
print(f"\n  freqs_cos[0][:6] = {[round(v,4) for v in freqs_cos[0][:6].tolist()]}   <- 位置0: 全是 attn_factor")
print(f"  freqs_cos[1][:6] = {[round(v,4) for v in freqs_cos[1][:6].tolist()]}")

print("\n" + "=" * 62)
print("它是'求什么'的? —— 它本身不做任何计算, 是一张查表")
print("=" * 62)
print("  它是'位置 -> 旋转因子'的查找表:")
print("    输入: 位置 t (行号)")
print("    输出: 该位置每个维度要用的 cos 系数 (head_dim 个数)")

print("\n--- 用法: apply_rotary 里按位置取行 ---")
seq_len = 4
q = torch.randn(1, 1, seq_len, dim)
cos_used = freqs_cos[:seq_len].unsqueeze(0).unsqueeze(0)   # (1,1,seq_len,dim)
sin_used = freqs_sin[:seq_len].unsqueeze(0).unsqueeze(0)
print(f"  q.shape = {tuple(q.shape)}")
print(f"  取前 {seq_len} 行后 broadcast 到 {tuple(cos_used.shape)}")

def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

q_rot = q * cos_used + rotate_half(q) * sin_used
print(f"  q_rot = q*cos + rotate_half(q)*sin -> shape {tuple(q_rot.shape)}")

print("\n--- 自检: 旋转不改变模长(逐位置) ---")
print("  位置   q.norm()     q_rot.norm()   是否相等(忽略 attn_factor 的缩放)")
print("  " + "-" * 56)
for i in range(seq_len):
    n1 = q[0, 0, i].norm().item()
    n2 = q_rot[0, 0, i].norm().item()
    print(f"  {i:>4}   {n1:>10.6f}   {n2:>12.6f}   ratio={n2/n1:.6f}")

print(f"\n  attn_factor={attn_factor:.4f}, 其平方={attn_factor**2:.4f}")
print("  -> 位置0(cos全=1, sin全=0)时 q 被整体放大 attn_factor 倍,")
print("     所以 q_rot 的模长比值应接近 attn_factor")
print(f"  实测位置0的比值 = {q_rot[0,0,0].norm().item()/q[0,0,0].norm().item():.6f}")

print("\n" + "=" * 62)
print("对 attention 分数的实际影响")
print("=" * 62)
q2 = torch.randn(1, 1, seq_len, dim)
k2 = torch.randn(1, 1, seq_len, dim)

def attn(qq, kk, factor_on):
    if factor_on:
        qr = qq * cos_used + rotate_half(qq) * sin_used
        kr = kk * cos_used + rotate_half(kk) * sin_used
    else:
        c = torch.cat([torch.cos(angles[:seq_len]), torch.cos(angles[:seq_len])], -1).unsqueeze(0).unsqueeze(0)
        s = torch.cat([torch.sin(angles[:seq_len]), torch.sin(angles[:seq_len])], -1).unsqueeze(0).unsqueeze(0)
        qr = qq * c + rotate_half(qq) * s
        kr = kk * c + rotate_half(kk) * s
    return (qr @ kr.transpose(-1, -2)) / (dim ** 0.5)

a_on = attn(q2, k2, True)
a_off = attn(q2, k2, False)
print(f"  开 attn_factor: logits 标准差 = {a_on.std():.4f}")
print(f"  关 attn_factor: logits 标准差 = {a_off.std():.4f}")
print(f"  比值 = {a_on.std()/a_off.std():.4f}   attn_factor^2 = {attn_factor**2:.4f}")
print("  -> q 和 k 各被放大 attn_factor 倍, 点积被放大 attn_factor^2 倍,")
print("     除以 sqrt(d) 不变 -> logits 整体放大 attn_factor^2 倍")
print(f"  实测: {a_on.std()/a_off.std():.4f} vs {attn_factor**2:.4f} 接近")

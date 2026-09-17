import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import torch

print("=" * 76)
print("q = (batch=2, heads=4, seq=6, head_dim=64)")
print("cos = (seq=6, head_dim=64)")
print("=" * 76)
q = torch.randn(2, 4, 6, 64)
cos = torch.randn(6, 64)

print(f"  {'unsqueeze_dim':<16}{'cos 变成':<20}{'和 q 对齐情况':<46}{'结果'}")
print("  " + "-" * 92)
for ud in [0, 1, 2]:
    c = cos.unsqueeze(ud)
    # 广播规则: 从最右边开始逐维比较
    cs = list(c.shape)
    qs = list(q.shape)
    # 右侧对齐
    pad = len(qs) - len(cs)
    aligned = [1] * pad + cs if pad > 0 else cs
    ok = all(a == b or a == 1 or b == 1 for a, b in zip(qs, aligned))
    detail = "  ".join(f"{a}vs{b}" for a, b in zip(qs, aligned))
    res = "OK" if ok else "失败"
    print(f"  {ud:<16}{str(tuple(c.shape)):<20}{detail:<46}{res}")

print("\n" + "=" * 76)
print("核心: 广播是'从最右边对齐'的")
print("=" * 76)
print("  q   = (2,  4,  6, 64)")
print("  cos = (6, 64)  <- 2 维, 左边自动补 1 -> (1, 1, 6, 64)")
print("        对照 q:  2vs1 OK   4vs1 OK   6vs6 OK   64vs64 OK   -> 成功")
print()
print("  unsqueeze(0) -> (1, 6, 64)  补前导 1 后 = (1, 1, 6, 64)")
print("        对照 q:  2vs1 OK   4vs1 OK   6vs6 OK   64vs64 OK   -> 也成功!")
print()
print("  unsqueeze(1) -> (6, 1, 64)  补前导 1 后 = (1, 6, 1, 64)")
print("        对照 q:  2vs1 OK   4vs6 失败!")
print()
print("  => 所以 unsqueeze_dim=1 在这里是错的! 0 才对")
print("  => 除非 cos 原本是 (batch, seq, head_dim) 这种 3 维布局")

print("\n" + "=" * 76)
print("关键: unsqueeze_dim 取决于 cos 的原始维度数")
print("=" * 76)
print("  HuggingFace 的 LlamaRotaryEmbedding 返回的 cos 是 3 维:")
print("    cos.shape = (batch, seq, head_dim)")
print("  这时:")
for ud in [1]:
    c3 = torch.randn(2, 6, 64).unsqueeze(ud)
    print(f"    cos(2,6,64).unsqueeze({ud}) -> {tuple(c3.shape)}")
    print(f"    和 q(2,4,6,64) 对齐: 2vs2 OK  4vs1 OK  6vs6 OK  64vs64 OK  -> 成功")
print()
print("  而 minimind 的 freqs_cos 是 2 维 (seq, head_dim):")
c2_ud0 = cos.unsqueeze(0)
print(f"    cos(6,64).unsqueeze(0) -> {tuple(c2_ud0.shape)}  -> 成功")
c2_ud1 = cos.unsqueeze(1)
print(f"    cos(6,64).unsqueeze(1) -> {tuple(c2_ud1.shape)}  -> 失败 (4 vs 6)")

print("\n" + "=" * 76)
print("实测确认")
print("=" * 76)
print("  cos 是 2 维 (seq, head_dim) 时:")
for ud in [0, 1]:
    try:
        r = q * cos.unsqueeze(ud)
        print(f"    unsqueeze_dim={ud} -> OK   {tuple(r.shape)}")
    except RuntimeError as e:
        print(f"    unsqueeze_dim={ud} -> 失败  {str(e)[:60]}")

print("\n  cos 是 3 维 (batch, seq, head_dim) 时:")
cos3 = torch.randn(2, 6, 64)
for ud in [0, 1]:
    try:
        r = q * cos3.unsqueeze(ud)
        print(f"    unsqueeze_dim={ud} -> OK   {tuple(r.shape)}")
    except RuntimeError as e:
        print(f"    unsqueeze_dim={ud} -> 失败  {str(e)[:60]}")

print("\n" + "=" * 76)
print("结论")
print("=" * 76)
print("  unsqueeze_dim 的正确值取决于 cos/sin 的维度数:")
print("    cos 2 维 (seq, hd)          -> unsqueeze_dim=0")
print("    cos 3 维 (batch, seq, hd)   -> unsqueeze_dim=1  (HF 的做法)")
print("  它存在的意义就是让同一个函数适配两种上游布局")

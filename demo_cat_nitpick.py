import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import torch

print("=" * 68)
print("刺 1: shape 是 tuple 不是 list,方括号 [1, 2, 5] 是 Python list")
print("=" * 68)
a = torch.zeros(1, 2, 5)
print("  a.shape      =", a.shape, " 类型:", type(a.shape).__name__)
print("  tuple(a.shape) =", tuple(a.shape))
print("  a.shape[2]   =", a.shape[2], " <- 可以正常取下标")
print("  写成 [1,2,5] 在文档里是简写,严格说 torch.Size([1, 2, 5]),底层是 tuple")

print("\n" + "=" * 68)
print("刺 2: '在 shape 为 5 的地方拼接' 有歧义")
print("=" * 68)
b = torch.zeros(1, 2, 5)
r = torch.cat((a, b), dim=2)
print(f"  拼接前 shape[2] = {a.shape[2]}")
print(f"  拼接后 shape[2] = {r.shape[2]}   <- 长度 5 变成 10,不是'插入到第5个位置'")
print("  数据排布: [前5个(来自a)] [后5个(来自b)]  <- b 接在 a 的尾巴,不是插在中间")
x = torch.arange(5).reshape(1, 1, 5).float()
y = torch.arange(5, 10).reshape(1, 1, 5).float()
print("  实测 dim=2 拼接:", torch.cat((x, y), dim=2).flatten().tolist())

print("\n" + "=" * 68)
print("刺 3(最关键): 你只说了'在哪拼',漏了'能不能拼'")
print("=" * 68)
p = torch.zeros(1, 2, 5)
q = torch.zeros(1, 3, 5)      # 第 1 维不同(3 vs 2),第 2 维相同
print("  p=(1,2,5)  q=(1,3,5)   dim=2 时其他维必须相同")
try:
    torch.cat((p, q), dim=2)
except RuntimeError as e:
    print("  cat(dim=2) -> RuntimeError:", str(e)[:78])
print("  => 你的规则只描述了'dim 指向哪一维',没描述'其他维必须一致'")
print("  => '在5那里拼接'成立的前提是 shape[0]、shape[1] 都完全相等")

print("\n" + "=" * 68)
print("刺 4: dim 是负数时,'第几个维度'的说法要补一句'从后往前数'")
print("=" * 68)
print("  dim=-1 在 (1,2,5) 上等于 dim=2,但如果张量是 4 维:")
t4 = torch.zeros(1, 2, 5, 7)
print("    4维张量 (1,2,5,7): dim=-1 指向最后一维(长度7),不是长度5那一维!")
print("    此时'长度5'那维是 dim=2 或 dim=-2")
print("  => 硬编码 dim=2 在维度数变化时会失效,dim=-1 更稳")

print("\n" + "=" * 68)
print("刺 5: 'dim 是 Tensor 的 shape 的拼接地方' —— dim 不是'地方',是下标")
print("=" * 68)
print("  dim 是一个整数下标(0/1/2 或负数),指向 shape 中某个位置的数字")
print("  真正被改变的是'该位置上的那个长度值',不是'形状本身'")
print("  准确说法: dim = 要沿其拼接的那个维度的下标")

print("\n" + "=" * 68)
print("刺 6: shape[2]=5 只说明'长度是5',不说明'内容在第5位'")
print("=" * 68)
print("  shape[2]=5 意思是'这一维有 5 个元素(index 0~4)'")
print("  和'第5个位置'完全是两回事 —— 索引从 0 开始,最大是 4")

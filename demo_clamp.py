import torch

x = torch.tensor([-1.7, -0.5, 0.0, 0.17, 0.5, 1.3, 2.5])
print("x                =", x)
print("clamp(x, 0, 1)   =", torch.clamp(x, 0, 1))
print("clamp(x,min=0)   =", torch.clamp(x, min=0))
print("clamp(x,max=0.5) =", torch.clamp(x, max=0.5))
print("x.clamp(0, 1)    =", x.clamp(0, 1))
print("x.clip(0, 1)     =", x.clip(0, 1))
print("dtype            =", torch.clamp(x, 0, 1).dtype, "| shape =", torch.clamp(x, 0, 1).shape)

print("\n--- 界也可以是张量(逐元素) ---")
lo = torch.tensor([-1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
print("clamp(x, min=lo) =", torch.clamp(x, min=lo))

print("\n--- out= 仅关键字 ---")
buf = torch.empty_like(x)
torch.clamp(x, 0, 1, out=buf)
print("buf              =", buf)

print("\n--- 整数张量 ---")
i = torch.tensor([-3, 0, 5, 12])
print("clamp(i, 0, 10)  =", torch.clamp(i, 0, 10))

print("\n--- clamp_ 就地修改 ---")
y = torch.tensor([-2.0, 0.5, 3.0])
z = y.clamp_(0, 1)
print("y                =", y, "| z is y:", z is y)

print("\n--- 对比 max / torch.maximum / x.max() ---")
a = torch.tensor([-1.0, 2.0, 3.0])
b = torch.tensor([1.0, 1.0, 1.0])
print("torch.maximum(a,b) =", torch.maximum(a, b))
print("a.max()            =", a.max(), "| max() 是 Python 内置,标量用:", max(3.0, 1.0))

print("\n--- 在 YaRN ramp 里的真实用法 ---")
low, high = 7, 15
ramp = torch.clamp(
    (torch.arange(16, dtype=torch.float32) - low) / max(high - low, 0.001), 0, 1
)
print("i    =", torch.arange(16))
print("ramp =", ramp)

import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import torch

v_list = [0, 1, 2, 3, 4, 5, 6, 7]
v_np = np.arange(8)
v_t = torch.arange(8)

print("=" * 74)
print("负步长(倒序): 三种类型行为不同!")
print("=" * 74)
print(f"  Python list  v[::-1] = {v_list[::-1]}          ✅ 支持")
print(f"  NumPy array  v[::-1] = {v_np[::-1].tolist()}          ✅ 支持")
try:
    r = v_t[::-1]
    print(f"  torch tensor v[::-1] = {r.tolist()}          ✅ 支持")
except Exception as e:
    print(f"  torch tensor v[::-1] -> {type(e).__name__}: {e}   ❌ 不支持!")

print("\n  其他负数步长:")
for step in [-2, -1, 0]:
    for name, arr in [("list", v_list), ("numpy", v_np), ("torch", v_t)]:
        try:
            r = arr[::step]
            val = r.tolist() if hasattr(r, "tolist") else r
            print(f"    {name:<6} [::{step:<2}] = {val}")
        except Exception as e:
            print(f"    {name:<6} [::{step:<2}] -> {type(e).__name__}: {str(e)[:46]}")
    print()

print("=" * 74)
print("torch 里要倒序怎么办")
print("=" * 74)
print(f"  v.flip(0)              = {v_t.flip(0).tolist()}")
print(f"  torch.flip(v, [0])     = {torch.flip(v_t, [0]).tolist()}")
print(f"  v[[7,6,5,4,3,2,1,0]]   = {v_t[[7,6,5,4,3,2,1,0]].tolist()}")
print(f"  v[torch.arange(7,-1,-1)] = {v_t[torch.arange(7,-1,-1)].tolist()}   <- 索引用负步长 arange 可以")

print("\n" + "=" * 74)
print("正步长是支持的")
print("=" * 74)
print(f"  v[::2]    = {v_t[::2].tolist()}   每隔一个")
print(f"  v[1::2]   = {v_t[1::2].tolist()}   从 1 开始每隔一个")
print(f"  v[0:6:3]  = {v_t[0:6:3].tolist()}   0 到 5, 步长 3")
print(f"  v[::0]    -> ", end="")
try:
    v_t[::0]
except Exception as e:
    print(f"{type(e).__name__}: {str(e)[:50]}")

print("\n" + "=" * 74)
print("回到切片规则(核心记住这个)")
print("=" * 74)
print("  语法: [起点 : 终点 : 步长]   含起点, 不含终点")
print()
print(f"  {'写法':<12}{'起点':<8}{'终点':<10}{'保留':<18}{'结果'}")
print("  " + "-" * 66)
for expr in ["v[4:]", "v[:4]", "v[2:5]", "v[:]", "v[3:3]", "v[-2:]", "v[:-2]"]:
    r = v_t[eval(expr.split("[", 1)[1].rsplit("]", 1)[0])] if False else eval("v_t" + expr[1:])
    print(f"  {expr:<12}{'':<8}{'':<10}{'':<18}{r.tolist()}")
print()
print("  规则一句话: ': 左边的数字砍左, 右边的数字砍右'")

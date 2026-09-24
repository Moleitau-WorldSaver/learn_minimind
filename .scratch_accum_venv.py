"""临时探针：在项目 .venv 里测 bf16 累加器的真实精度（可删除）"""
import sys
import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding="utf-8")
dev, dt = "cuda", torch.bfloat16

print("torch", torch.__version__, "| cuda", torch.cuda.get_device_name(0))
print("allow_bf16_reduced_precision_reduction =",
      torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
print()

for K in (1024, 8192):
    torch.manual_seed(0)
    a = (torch.randn(1, K, device=dev) * 0.05)
    b = (torch.randn(K, 1, device=dev) * 0.05)
    ref = (a.double() @ b.double()).item()
    ab, bb = a.to(dt), b.to(dt)
    ref_bf16in = (ab.double() @ bb.double()).item()   # 输入舍入后的精确累加 = fp32 累加的理想值

    line = f"K={K:5d}  fp64={ref:+.6f}  输入舍入后理想={ref_bf16in:+.6f}"
    for flag in (True, False):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag
        torch._C.clear_autocast_cache()
        with torch.autocast(device_type="cuda", dtype=dt):
            out = torch.mm(a, b).float().item()
        line += f"  | flag={str(flag):5s}: {out:+.6f} (dev {abs(out - ref_bf16in):.2e})"
    print(line)

print()
print("读法：flag=False 那一列的 dev 若明显更小，说明 fp32 累加确实在兜精度；")
print("      两列相同 = 该 flag 在当前 GPU/kernel 选择下没改变累加路径。")

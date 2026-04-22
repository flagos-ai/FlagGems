import torch

torch.manual_seed(42)
device = 'cuda'

# 测试 1：用已知数据手动构造（最大值在明确位置）
print("=== 测试1：已知最大值位置 ===")
x = torch.zeros(1048576, dtype=torch.bfloat16, device=device)
x[499202] = 1.0  # 手动放最大值
ref_cpu = torch.argmax(x.cpu())
ref_xpu = torch.argmax(x)
print(f"Expected: {499202}, CPU: {ref_cpu.item()}, XPU: {ref_xpu.item()}, Match: {ref_xpu.item() == 499202}")

# 测试 2：随机数据，多 shape 多 dtype
print("\n=== 测试2：随机数据 ===")
for dtype in [torch.float32, torch.float16, torch.bfloat16]:
    for M in [2, 1048576, 1536000]:
        torch.manual_seed(123)
        x = torch.randn(M, dtype=dtype, device=device)
        cpu_res = torch.argmax(x.cpu()).item()
        xpu_res = torch.argmax(x).item()
        # 双重验证：检查 xpu 返回的索引处是否真的是最大值
        actual_val = x[xpu_res].cpu().item()
        true_max = x.cpu().max().item()
        ok = (cpu_res == xpu_res) and (abs(actual_val - true_max) < 1e-2)
        print(f"  dtype={str(dtype):14s} M={M:>10d}  cpu={cpu_res:>10d}  xpu={xpu_res:>10d}  val_check={'OK' if ok else 'FAIL'}")


"""
测试 CPU vs CUDA 的 median 行为差异
"""
import torch

print("="*80)
print("测试 CPU vs CUDA 的 median 行为差异")
print("="*80)

torch.manual_seed(0)

# 生成数据
data_f32 = torch.randn((1024, 1024), dtype=torch.float32)
data_bf16 = data_f32.to(torch.bfloat16)

# 找一个有重复中位数值的行
for row_idx in range(100):
    row_bf16 = data_bf16[row_idx]
    k = (1024 + 1) // 2
    kth_val, kth_idx = torch.kthvalue(row_bf16, k)
    mask = row_bf16 == kth_val
    positions = torch.where(mask)[0]

    if len(positions) >= 2:
        print(f"\n行 {row_idx}: 找到重复中位数值")
        print(f"  中位数值 {kth_val.item()} 出现在位置: {positions.tolist()}")

        # CPU 测试
        data_bf16_cpu = data_bf16.cpu()

        vals_cpu, idx_cpu = torch.median(data_bf16_cpu, dim=1)
        median_from_full_cpu = idx_cpu[row_idx].item()

        val_cpu, idx_val_cpu = torch.median(data_bf16_cpu[row_idx], dim=0)

        kth_val_cpu, kth_idx_cpu = torch.kthvalue(data_bf16_cpu[row_idx], k)

        print(f"\nCPU 结果:")
        print(f"  整个张量 median: {median_from_full_cpu}")
        print(f"  单行 median: {idx_val_cpu.item()}")
        print(f"  kthvalue: {kth_idx_cpu.item()}")

        # CUDA 测试
        data_bf16_cuda = data_bf16.cuda()

        vals_cuda, idx_cuda = torch.median(data_bf16_cuda, dim=1)
        median_from_full_cuda = idx_cuda[row_idx].item()

        val_cuda, idx_val_cuda = torch.median(data_bf16_cuda[row_idx], dim=0)

        kth_val_cuda, kth_idx_cuda = torch.kthvalue(data_bf16_cuda[row_idx], k)

        print(f"\nCUDA 结果:")
        print(f"  整个张量 median: {median_from_full_cuda}")
        print(f"  单行 median: {idx_val_cuda.item()}")
        print(f"  kthvalue: {kth_idx_cuda.item()}")

        print(f"\nCPU vs CUDA 比较:")
        print(f"  整个张量 median: CPU={median_from_full_cpu}, CUDA={median_from_full_cuda}, 一致={median_from_full_cpu == median_from_full_cuda}")
        print(f"  单行 median: CPU={idx_val_cpu.item()}, CUDA={idx_val_cuda.item()}, 一致={idx_val_cpu.item() == idx_val_cuda.item()}")
        print(f"  kthvalue: CPU={kth_idx_cpu.item()}, CUDA={kth_idx_cuda.item()}, 一致={kth_idx_cpu.item() == kth_idx_cuda.item()}")

        break

# 使用 float64 测试
print("\n" + "="*80)
print("使用 float64 数据测试")
print("="*80)

torch.manual_seed(0)
data_f32 = torch.randn((1024, 1024), dtype=torch.float32)
data_f64 = data_f32.to(torch.float64)

for row_idx in range(100):
    row_f64 = data_f64[row_idx]
    k = (1024 + 1) // 2
    kth_val, kth_idx = torch.kthvalue(row_f64, k)
    mask = row_f64 == kth_val
    positions = torch.where(mask)[0]

    if len(positions) >= 2:
        print(f"\n行 {row_idx}: 找到重复中位数值")
        print(f"  中位数值 {kth_val.item()} 出现在位置: {positions.tolist()}")

        # CPU
        vals_cpu, idx_cpu = torch.median(data_f64.cpu(), dim=1)
        median_from_full_cpu = idx_cpu[row_idx].item()

        # CUDA
        vals_cuda, idx_cuda = torch.median(data_f64.cuda(), dim=1)
        median_from_full_cuda = idx_cuda[row_idx].item()

        print(f"\nfloat64 CPU vs CUDA 比较:")
        print(f"  CPU 整个张量 median: {median_from_full_cpu}")
        print(f"  CUDA 整个张量 median: {median_from_full_cuda}")
        print(f"  一致: {median_from_full_cpu == median_from_full_cuda}")

        if median_from_full_cpu != median_from_full_cuda:
            print(f"  CPU 选择位置: {positions.tolist().index(median_from_full_cpu) if median_from_full_cpu in positions else 'N/A'}")
            print(f"  CUDA 选择位置: {positions.tolist().index(median_from_full_cuda) if median_from_full_cuda in positions else 'N/A'}")

        break

print("\n" + "="*80)
print("结论")
print("="*80)

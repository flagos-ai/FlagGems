"""
测试 CUDA float64 上 torch.median 和 torch.kthvalue 的一致性
"""
import torch

print("="*80)
print("测试 CUDA float64 上 torch.median 和 torch.kthvalue 的一致性")
print("="*80)

torch.manual_seed(0)

# 生成数据
data_f32 = torch.randn((1024, 1024), dtype=torch.float32)
data_f64_cuda = data_f32.to(torch.float64).cuda()

print(f"数据类型: {data_f64_cuda.dtype}")
print(f"数据设备: {data_f64_cuda.device}")

# 测试 100 行
consistent_count = 0
inconsistent_count = 0
total_duplicate_count = 0

for row_idx in range(1000):
    row = data_f64_cuda[row_idx]
    k = (1024 + 1) // 2
    kth_val, kth_idx = torch.kthvalue(row, k)
    mask = row == kth_val
    positions = torch.where(mask)[0]

    if len(positions) >= 2:
        total_duplicate_count += 1

        # 整个张量的 median
        vals, idx = torch.median(data_f64_cuda, dim=1)
        median_from_full = idx[row_idx].item()

        # 单行的 median
        val, idx_val = torch.median(row, dim=0)

        # 检查一致性
        if median_from_full == kth_idx.item() and idx_val.item() == kth_idx.item():
            consistent_count += 1
        else:
            inconsistent_count += 1
            if inconsistent_count <= 5:
                print(f"\n行 {row_idx}: 不一致")
                print(f"  中位数值 {kth_val.item()} 出现在位置: {positions.tolist()}")
                print(f"  整个张量 median: {median_from_full} (位置 {positions.tolist().index(median_from_full) if median_from_full in positions else 'N/A'})")
                print(f"  单行 median: {idx_val.item()} (位置 {positions.tolist().index(idx_val.item()) if idx_val.item() in positions else 'N/A'})")
                print(f"  kthvalue: {kth_idx.item()} (位置 {positions.tolist().index(kth_idx.item())})")

print(f"\n统计:")
print(f"  有重复中位数值的行数: {total_duplicate_count}")
print(f"  一致的行数: {consistent_count}")
print(f"  不一致的行数: {inconsistent_count}")

if total_duplicate_count > 0:
    print(f"  一致率: {consistent_count / total_duplicate_count * 100:.1f}%")

print("\n" + "="*80)
print("结论")
print("="*80)
print("""
如果 torch.median 和 torch.kthvalue 在 CUDA float64 上不一致，
那么使用 kthvalue 来模拟 median 可能会导致不匹配。

在这种情况下，我们可能需要：
1. 直接使用 torch.median 而不是 kthvalue
2. 或者接受一定的差异率
""")

"""
测试 CUDA float64 上 torch.median 和 torch.kthvalue 的一致性（扩展搜索）
"""
import torch

print("="*80)
print("测试 CUDA float64 上 torch.median 和 torch.kthvalue 的一致性（扩展搜索）")
print("="*80)

# 尝试多个种子
for seed in range(100):
    torch.manual_seed(seed)
    data_f32 = torch.randn((1024, 1024), dtype=torch.float32)
    data_f64_cuda = data_f32.to(torch.float64).cuda()

    for row_idx in range(1000):
        row = data_f64_cuda[row_idx]
        k = (1024 + 1) // 2
        kth_val, kth_idx = torch.kthvalue(row, k)
        mask = row == kth_val
        positions = torch.where(mask)[0]

        if len(positions) >= 2:
            print(f"\n种子 {seed}, 行 {row_idx}: 找到重复中位数值")
            print(f"  中位数值 {kth_val.item()} 出现在位置: {positions.tolist()}")

            # 整个张量的 median
            vals, idx = torch.median(data_f64_cuda, dim=1)
            median_from_full = idx[row_idx].item()

            # 单行的 median
            val, idx_val = torch.median(row, dim=0)

            # kthvalue
            kth_val_row, kth_idx_row = torch.kthvalue(row, k)

            print(f"\n结果:")
            print(f"  整个张量 median: {median_from_full} (位置 {positions.tolist().index(median_from_full) if median_from_full in positions else 'N/A'})")
            print(f"  单行 median: {idx_val.item()} (位置 {positions.tolist().index(idx_val.item()) if idx_val.item() in positions else 'N/A'})")
            print(f"  kthvalue: {kth_idx_row.item()} (位置 {positions.tolist().index(kth_idx_row.item())})")

            # 检查一致性
            if median_from_full == kth_idx_row.item() and idx_val.item() == kth_idx_row.item():
                print(f"\n  >>> 一致！都选择位置 {positions.tolist().index(kth_idx_row.item())}")
            else:
                print(f"\n  >>> 不一致！")

            # 只显示第一个例子
            exit(0)

print("没有找到重复中位数值的行")

"""
测试 bfloat16 数据的 PyTorch median 行为
"""
import torch

print("="*80)
print("测试 bfloat16 数据的 PyTorch median 行为")
print("="*80)

torch.manual_seed(0)

# 生成 bfloat16 数据
data_bf16 = torch.randn((1024, 1024), dtype=torch.bfloat16)
data_f64 = data_bf16.to(torch.float64)

print(f"数据形状: {data_f64.shape}")
print(f"数据类型: {data_f64.dtype}")

# 找一个有重复中位数值的行
for row_idx in range(100):
    row = data_f64[row_idx]
    k = (1024 + 1) // 2  # 513
    kth_val, kth_idx = torch.kthvalue(row, k)
    mask = row == kth_val
    positions = torch.where(mask)[0]

    if len(positions) >= 2:
        print(f"\n行 {row_idx}: 找到重复中位数值")
        print(f"  中位数值 {kth_val.item()} 出现在位置: {positions.tolist()}")

        # 方法 1: 在整个张量上计算 median
        vals, idx = torch.median(data_f64, dim=1)
        median_from_full = idx[row_idx].item()

        # 方法 2: 在单行上计算 median
        val, idx_val = torch.median(row, dim=0)

        # 方法 3: kthvalue
        kth_val_row, kth_idx_row = torch.kthvalue(row, k)

        print(f"\n结果比较:")
        print(f"  整个张量 median: {median_from_full}")
        print(f"  单行 median: {idx_val.item()}")
        print(f"  kthvalue: {kth_idx_row.item()}")

        print(f"\n选择位置:")
        print(f"  整个张量选择: {positions.tolist().index(median_from_full) if median_from_full in positions else 'N/A'} ({'last' if median_from_full == positions[-1].item() else 'first' if median_from_full == positions[0].item() else 'middle'})")
        print(f"  单行选择: {positions.tolist().index(idx_val.item()) if idx_val.item() in positions else 'N/A'} ({'last' if idx_val.item() == positions[-1].item() else 'first' if idx_val.item() == positions[0].item() else 'middle'})")
        print(f"  kthvalue选择: {positions.tolist().index(kth_idx_row.item()) if kth_idx_row.item() in positions else 'N/A'} ({'last' if kth_idx_row.item() == positions[-1].item() else 'first' if kth_idx_row.item() == positions[0].item() else 'middle'})")

        # 检查是否一致
        if median_from_full != idx_val.item():
            print(f"\n>>> 发现不一致！整个张量选择 {median_from_full}，单行选择 {idx_val.item()}")
        else:
            print(f"\n>>> 一致！都选择 {median_from_full}")

        break

# 测试多行
print("\n" + "="*80)
print("测试多行以确认模式")
print("="*80)

inconsistent_count = 0
total_count = 0

for row_idx in range(500):
    row = data_f64[row_idx]
    k = (1024 + 1) // 2
    kth_val, kth_idx = torch.kthvalue(row, k)
    mask = row == kth_val
    positions = torch.where(mask)[0]

    if len(positions) >= 2:
        total_count += 1

        vals, idx = torch.median(data_f64, dim=1)
        median_from_full = idx[row_idx].item()

        val, idx_val = torch.median(row, dim=0)

        if median_from_full != idx_val.item():
            inconsistent_count += 1
            if inconsistent_count <= 3:
                print(f"行 {row_idx}: 整个张量={median_from_full}, 单行={idx_val.item()}, 位置={positions.tolist()}")

print(f"\n统计: 在 {total_count} 个有重复值的行中，{inconsistent_count} 行不一致")

print("\n" + "="*80)
print("结论")
print("="*80)

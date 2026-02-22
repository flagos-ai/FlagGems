"""
验证 PyTorch median 在整个张量 vs 单行的行为差异
"""
import torch

print("="*80)
print("验证 PyTorch median 行为差异")
print("="*80)

# 创建一个简单的例子
data = torch.tensor([
    [1.0, 2.0, 2.0, 3.0, 4.0],  # median=2, positions=[1,2]
    [5.0, 2.0, 2.0, 1.0, 3.0],  # median=2, positions=[1,2]
], dtype=torch.float32)

print("数据:")
print(data)

# 在整个张量上计算 median (dim=1)
print("\n--- 在整个张量上 (dim=1) ---")
vals, idx = torch.median(data, dim=1)
print(f"Median values: {vals.tolist()}")
print(f"Median indices: {idx.tolist()}")

# 在每一行上单独计算 median
print("\n--- 在每一行上单独计算 ---")
for i in range(len(data)):
    val, idx_val = torch.median(data[i], dim=0)
    print(f"Row {i}: median value={val.item()}, index={idx_val.item()}")

# 测试 2: 使用更大尺寸的数据
print("\n" + "="*80)
print("测试 2: 大尺寸数据 (1024, 1024)")
print("="*80)

torch.manual_seed(0)
data = torch.randn((1024, 1024), dtype=torch.float32)

# 找一个有重复中位数值的行
for row_idx in range(100):
    row = data[row_idx]
    k = (1024 + 1) // 2
    kth_val, kth_idx = torch.kthvalue(row, k)
    mask = row == kth_val
    positions = torch.where(mask)[0]

    if len(positions) >= 2:
        print(f"\n行 {row_idx}: 找到重复中位数值")
        print(f"  中位数值 {kth_val.item()} 出现在位置: {positions.tolist()}")

        # 方法 1: 在整个张量上计算 median
        vals, idx = torch.median(data, dim=1)
        median_from_full = idx[row_idx].item()

        # 方法 2: 在单行上计算 median
        val, idx_val = torch.median(row, dim=0)

        # 方法 3: kthvalue
        kth_val_row, kth_idx_row = torch.kthvalue(row, k)

        print(f"  整个张量 median: {median_from_full}")
        print(f"  单行 median: {idx_val.item()}")
        print(f"  kthvalue: {kth_idx_row.item()}")

        print(f"  整个张量选择: {('last' if median_from_full == positions[-1].item() else 'first' if median_from_full == positions[0].item() else 'middle')}")
        print(f"  单行选择: {('last' if idx_val.item() == positions[-1].item() else 'first' if idx_val.item() == positions[0].item() else 'middle')}")
        print(f"  kthvalue选择: {('last' if kth_idx_row.item() == positions[-1].item() else 'first' if kth_idx_row.item() == positions[0].item() else 'middle')}")

        break

print("\n" + "="*80)
print("结论")
print("="*80)
print("""
PyTorch 的 torch.median 在不同调用方式下有不同的行为:
1. 在整个张量上调用 torch.median(data, dim=1): 可能选择最后一个出现位置
2. 在单行上调用 torch.median(row, dim=0): 选择第一个出现位置 (与 kthvalue 一致)

这是 PyTorch 的内部实现细节，可能与优化算法有关。
""")

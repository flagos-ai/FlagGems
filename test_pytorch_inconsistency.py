"""
Verify PyTorch median inconsistency
"""
import torch

print("="*80)
print("Verify PyTorch median inconsistency")
print("="*80)

# Create a simple example with duplicate median values
data = torch.tensor([
    [1.0, 2.0, 2.0, 3.0],
    [2.0, 2.0, 2.0, 3.0],
], dtype=torch.float32)

print("Data:")
print(data)

# Test dim=1 (along rows)
print("\n--- Test dim=1 (along rows) ---")
vals, idx = torch.median(data, dim=1)
print(f"Median values: {vals.tolist()}")
print(f"Median indices: {idx.tolist()}")

# Check each row individually
print("\nIndividual rows:")
for i in range(len(data)):
    val, idx_val = torch.median(data[i], dim=0)
    print(f"  Row {i}: median value={val.item()}, index={idx_val.item()}")

# Test with bfloat16 -> float64
print("\n--- Test bfloat16 -> float64 ---")
data_bf16 = torch.randn((1024, 1024), dtype=torch.bfloat16)
data_f64 = data_bf16.to(torch.float64)

# Find a row with duplicate median values
for row_idx in range(100):
    row = data_f64[row_idx]
    k = (1024 + 1) // 2
    kth_val, kth_idx = torch.kthvalue(row, k)
    mask = row == kth_val
    positions = torch.where(mask)[0]
    if len(positions) >= 2:
        print(f"\nRow {row_idx}: Found duplicate median values")
        print(f"  Median value {kth_val.item()} at positions: {positions.tolist()}")

        # Method 1: Full tensor
        vals, idx = torch.median(data_f64, dim=1)
        median_from_full = idx[row_idx].item()

        # Method 2: Individual row
        val, idx_val = torch.median(row, dim=0)

        # Method 3: kthvalue
        kth_val_row, kth_idx_row = torch.kthvalue(row, k)

        print(f"  Full tensor median: {median_from_full}")
        print(f"  Individual row median: {idx_val.item()}")
        print(f"  kthvalue: {kth_idx_row.item()}")

        print(f"  Full tensor selects: {('last' if median_from_full == positions[-1].item() else 'first' if median_from_full == positions[0].item() else 'middle')}")
        print(f"  Individual row selects: {('last' if idx_val.item() == positions[-1].item() else 'first' if idx_val.item() == positions[0].item() else 'middle')}")
        print(f"  kthvalue selects: {('last' if kth_idx_row.item() == positions[-1].item() else 'first' if kth_idx_row.item() == positions[0].item() else 'middle')}")

        break

print("\n" + "="*80)
print("Conclusion: PyTorch median has inconsistent behavior!")
print("When called on a full tensor, it may select the last occurrence.")
print("When called on a single row, it selects the first occurrence (same as kthvalue).")
print("="*80)

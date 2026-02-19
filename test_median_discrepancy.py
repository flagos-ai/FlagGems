"""
Debug the median discrepancy
"""
import torch

def to_reference_simulated(inp, upcast=False):
    """Simulate to_reference when TO_CPU is False (default)"""
    ref_inp = inp
    if upcast:
        if ref_inp.is_complex():
            ref_inp = ref_inp.to(torch.complex128)
        else:
            ref_inp = ref_inp.to(torch.float64)
    return ref_inp

print("="*80)
print("Debug the median discrepancy")
print("="*80)

# Use the same seed as the failing test
torch.manual_seed(0)
inp = torch.randn((1024, 1024), dtype=torch.bfloat16, device='cuda')
ref_inp = to_reference_simulated(inp, True)

# Get row 416
row_idx = 416

# Method 1: Get median from the full tensor
ref_vals, ref_idx = torch.median(ref_inp, dim=1, keepdim=False)
median_from_full = ref_idx[row_idx].item()

# Method 2: Get median from the row directly
row_data = ref_inp[row_idx]
median_val_from_row, median_idx_from_row = torch.median(row_data, dim=0)

print(f"Row {row_idx}:")
print(f"  Median index from full tensor: {median_from_full}")
print(f"  Median index from row: {median_idx_from_row.item()}")
print(f"  Match: {median_from_full == median_idx_from_row.item()}")

# Check the row data
print(f"\nRow data (first 20):")
print(f"  {row_data[:20].tolist()}")

# Let's manually compute the median
k = (1024 + 1) // 2  # 513
kth_val, kth_idx = torch.kthvalue(row_data, k)
print(f"\nkthvalue:")
print(f"  Value: {kth_val.item()}, Index: {kth_idx.item()}")

# Check if there are duplicate median values
mask = row_data == kth_val
positions = torch.where(mask)[0]
if len(positions) >= 2:
    print(f"\nMedian value appears at positions: {positions.tolist()}")

print("\n" + "="*80)
print("Analysis complete")
print("="*80)

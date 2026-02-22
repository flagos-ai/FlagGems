"""
Debug PyTorch median behavior vs stable sort
"""
import torch

# Set seed for reproducibility
torch.manual_seed(0)

print("="*80)
print("Debug PyTorch median behavior vs stable sort")
print("="*80)

# Generate random data
torch.manual_seed(0)
row_f32 = torch.randn(1024, dtype=torch.float32)
row_bf16 = row_f32.to(torch.bfloat16)
row_f64 = row_f32.to(torch.float64)

# Find a case where stable sort and median differ
for attempt in range(10000):
    row_f32 = torch.randn(1024, dtype=torch.float32)
    row_bf16 = row_f32.to(torch.bfloat16)
    row_f64 = row_f32.to(torch.float64)

    k = (1024 + 1) // 2  # 513
    median_pos = k - 1  # 512

    # Use stable sort
    sorted_idx_bf16 = torch.argsort(row_bf16, stable=True)
    sorted_idx_f64 = torch.argsort(row_f64, stable=True)

    median_idx_sort_bf16 = sorted_idx_bf16[median_pos].item()
    median_idx_sort_f64 = sorted_idx_f64[median_pos].item()

    # Use median
    median_val_bf16, median_idx_bf16 = torch.median(row_bf16, dim=0)
    median_val_f64, median_idx_f64 = torch.median(row_f64, dim=0)

    # Check if they differ
    if median_idx_sort_f64 != median_idx_f64.item():
        print(f"\n--- Found mismatch at attempt {attempt} ---")
        print(f"Stable sort (float64): {median_idx_sort_f64}")
        print(f"Median (float64): {median_idx_f64.item()}")
        print(f"Difference: {abs(median_idx_sort_f64 - median_idx_f64.item())}")

        # Get the median value
        kth_val_f64, kth_idx_f64 = torch.kthvalue(row_f64, k)
        print(f"\nkthvalue (float64): value={kth_val_f64.item()}, index={kth_idx_f64.item()}")

        # Find positions with median value
        mask_f64 = row_f64 == kth_val_f64
        positions_f64 = torch.where(mask_f64)[0]

        if len(positions_f64) >= 2:
            print(f"Median value appears at positions: {positions_f64.tolist()}")
            print(f"  First: {positions_f64[0].item()}, Last: {positions_f64[-1].item()}")

        # Check bfloat16
        median_idx_sort_bf16 = sorted_idx_bf16[median_pos].item()
        median_val_bf16, median_idx_bf16 = torch.median(row_bf16, dim=0)

        print(f"\nStable sort (bfloat16): {median_idx_sort_bf16}")
        print(f"Median (bfloat16): {median_idx_bf16.item()}")

        # Check if they match
        if median_idx_sort_f64 == median_idx_sort_bf16:
            print(f"\n>>> Stable sort matches for bfloat16 and float64")
        else:
            print(f"\n>>> Stable sort differs for bfloat16 and float64")

        break

print("\n" + "="*80)
print("Analysis complete")
print("="*80)

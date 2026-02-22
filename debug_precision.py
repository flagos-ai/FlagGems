"""
Analyze precision impact on median indices
"""
import torch

# Set seed for reproducibility
torch.manual_seed(42)

print("="*80)
print("Analyzing precision impact on median indices")
print("="*80)

# Generate random data
data_f32 = torch.randn(1024, dtype=torch.float32)
data_bf16 = data_f32.to(torch.bfloat16)
data_f64 = data_f32.to(torch.float64)

# Find cases where float64 and bfloat16 give different median indices
for test_row in range(100):
    row_f32 = torch.randn(1024, dtype=torch.float32)
    row_bf16 = row_f32.to(torch.bfloat16)
    row_f64 = row_f32.to(torch.float64)

    # Calculate median for both
    k = (1024 + 1) // 2  # 513

    # Use stable sort to get median index
    sorted_idx_bf16 = torch.argsort(row_bf16, stable=True)
    sorted_idx_f64 = torch.argsort(row_f64, stable=True)

    median_pos = k - 1  # 512
    median_idx_bf16 = sorted_idx_bf16[median_pos].item()
    median_idx_f64 = sorted_idx_f64[median_pos].item()

    if median_idx_bf16 != median_idx_f64:
        print(f"\nTest row {test_row}: Found different median indices")
        print(f"  bfloat16 median index: {median_idx_bf16}")
        print(f"  float64 median index: {median_idx_f64}")

        # Get median values
        median_val_bf16 = row_bf16[median_idx_bf16].item()
        median_val_f64 = row_f64[median_idx_f64].item()

        print(f"  bfloat16 median value: {median_val_bf16}")
        print(f"  float64 median value: {median_val_f64}")

        # Check if values are equal when converted
        median_val_f64_as_bf16 = torch.tensor(median_val_f64).to(torch.bfloat16).item()
        print(f"  float64 value as bfloat16: {median_val_f64_as_bf16}")
        print(f"  Values match in bfloat16: {median_val_bf16 == median_val_f64_as_bf16}")

        # Find positions with duplicate median values
        mask_bf16 = row_bf16 == median_val_bf16
        positions_bf16 = torch.where(mask_bf16)[0]

        mask_f64_bf16 = row_bf16 == median_val_f64_as_bf16
        positions_f64_bf16 = torch.where(mask_f64_bf16)[0]

        if len(positions_bf16) >= 2:
            print(f"  bfloat16 median value appears at positions: {positions_bf16.tolist()}")

        if len(positions_f64_bf16) >= 2:
            print(f"  float64 median value (as bfloat16) appears at positions: {positions_f64_bf16.tolist()}")

        # Check if the median value changes due to precision
        kth_val_bf16, kth_idx_bf16 = torch.kthvalue(row_bf16, k)
        kth_val_f64, kth_idx_f64 = torch.kthvalue(row_f64, k)

        print(f"\n  kthvalue bfloat16: value={kth_val_bf16.item()}, idx={kth_idx_bf16.item()}")
        print(f"  kthvalue float64:  value={kth_val_f64.item()}, idx={kth_idx_f64.item()}")

        break

print("\n" + "="*80)
print("Analysis complete")
print("="*80)

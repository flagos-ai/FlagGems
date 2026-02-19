"""
Analyze stable sort behavior: first vs last occurrence
"""
import torch

# Set seed for reproducibility
torch.manual_seed(42)

print("="*80)
print("Analyzing stable sort behavior: first vs last occurrence")
print("="*80)

# Generate random data
row_f32 = torch.randn(1024, dtype=torch.float32)
row_bf16 = row_f32.to(torch.bfloat16)

# Find a case with duplicate median values
for test_row in range(1000):
    row_f32 = torch.randn(1024, dtype=torch.float32)
    row_bf16 = row_f32.to(torch.bfloat16)
    row_f64 = row_f32.to(torch.float64)

    k = (1024 + 1) // 2  # 513

    # Use kthvalue to find median value
    kth_val_bf16, kth_idx_bf16 = torch.kthvalue(row_bf16.cpu(), k)
    kth_val_f64, kth_idx_f64 = torch.kthvalue(row_f64, k)

    # Find all positions with median value
    mask_bf16 = row_bf16.cpu() == kth_val_bf16
    positions_bf16 = torch.where(mask_bf16)[0]

    if len(positions_bf16) >= 2:
        print(f"\nTest row {test_row}: Found duplicate median values")
        print(f"Median value {kth_val_bf16.item()} appears at positions: {positions_bf16.tolist()}")
        print(f"  First: {positions_bf16[0].item()}, Last: {positions_bf16[-1].item()}")

        # CPU stable sort
        sorted_idx_bf16_cpu = torch.argsort(row_bf16.cpu(), stable=True)
        median_pos = k - 1  # 512
        median_idx_bf16_cpu = sorted_idx_bf16_cpu[median_pos].item()
        print(f"\nCPU stable sort (bfloat16):")
        print(f"  Median index: {median_idx_bf16_cpu}")
        print(f"  Selects: {('first' if median_idx_bf16_cpu == positions_bf16[0].item() else 'last' if median_idx_bf16_cpu == positions_bf16[-1].item() else 'middle')}")

        # CUDA stable sort
        sorted_idx_bf16_cuda = torch.argsort(row_bf16.cuda(), stable=True)
        median_idx_bf16_cuda = sorted_idx_bf16_cuda[median_pos].item()
        print(f"\nCUDA stable sort (bfloat16):")
        print(f"  Median index: {median_idx_bf16_cuda}")
        print(f"  Selects: {('first' if median_idx_bf16_cuda == positions_bf16[0].item() else 'last' if median_idx_bf16_cuda == positions_bf16[-1].item() else 'middle')}")

        # CPU stable sort (float64)
        mask_f64 = row_f64 == kth_val_f64
        positions_f64 = torch.where(mask_f64)[0]
        if len(positions_f64) >= 2:
            sorted_idx_f64_cpu = torch.argsort(row_f64, stable=True)
            median_idx_f64_cpu = sorted_idx_f64_cpu[median_pos].item()
            print(f"\nCPU stable sort (float64):")
            print(f"  Median value {kth_val_f64.item()} appears at positions: {positions_f64.tolist()}")
            print(f"  Median index: {median_idx_f64_cpu}")
            print(f"  Selects: {('first' if median_idx_f64_cpu == positions_f64[0].item() else 'last' if median_idx_f64_cpu == positions_f64[-1].item() else 'middle')}")

        break

print("\n" + "="*80)
print("Analysis complete")
print("="*80)

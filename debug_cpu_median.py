"""
Debug script to understand PyTorch CPU median behavior
"""
import torch

# Set seed for reproducibility
torch.manual_seed(0)

print("="*80)
print("Analyzing PyTorch CPU median behavior")
print("="*80)

# Test case from earlier: float16 with duplicate median values
data_f32 = torch.randn(1024, dtype=torch.float32)
data_f16 = data_f32.to(torch.float16)

for test_row in range(1000):
    row_f32 = torch.randn(1024, dtype=torch.float32)
    row_f16 = row_f32.to(torch.float16)

    k = (1024 + 1) // 2  # 513

    # CPU kthvalue
    kth_val_cpu, kth_idx_cpu = torch.kthvalue(row_f16.cpu(), k)

    # Find all positions with median value
    mask = row_f16.cpu() == kth_val_cpu
    positions = torch.where(mask)[0]

    if len(positions) >= 2:
        print(f"\nTest row {test_row}: Found duplicate median values")
        print(f"k={k}")
        print(f"Median value {kth_val_cpu.item()} appears at positions: {positions.tolist()}")
        print(f"  First: {positions[0].item()}, Last: {positions[-1].item()}")

        # CPU kthvalue
        print(f"CPU kthvalue: idx={kth_idx_cpu.item()}")
        print(f"  - Matches first? {kth_idx_cpu.item() == positions[0].item()}")
        print(f"  - Matches last? {kth_idx_cpu.item() == positions[-1].item()}")

        # CPU median
        median_val_cpu, median_idx_cpu = torch.median(row_f16.cpu(), dim=0)
        print(f"CPU median: idx={median_idx_cpu.item()}")
        print(f"  - Matches first? {median_idx_cpu.item() == positions[0].item()}")
        print(f"  - Matches last? {median_idx_cpu.item() == positions[-1].item()}")

        # Try using stable sort
        sorted_idx = torch.argsort(row_f16.cpu(), dim=0, stable=True)
        median_pos = k - 1  # 0-indexed
        print(f"Stable sort median: idx={sorted_idx[median_pos].item()}")
        print(f"  - Matches CPU median? {sorted_idx[median_pos].item() == median_idx_cpu.item()}")

        # Find which position has the median value at the k-th position in sorted order
        print(f"\nDetailed analysis:")
        print(f"  positions: {positions.tolist()}")
        print(f"  kthvalue CPU index: {kth_idx_cpu.item()}")
        print(f"  median CPU index: {median_idx_cpu.item()}")
        print(f"  stable sort index: {sorted_idx[median_pos].item()}")

        # Check if stable sort gives the same result as CPU median
        if sorted_idx[median_pos].item() == median_idx_cpu.item():
            print(f"\n>>> STABLE SORT MATCHES CPU MEDIAN!")
        else:
            print(f"\n>>> STABLE SORT DOES NOT MATCH CPU MEDIAN!")

        break

# Test with bfloat16
print("\n" + "="*80)
print("Analyzing PyTorch CPU median behavior for bfloat16")
print("="*80)

for test_row in range(1000):
    row_f32 = torch.randn(64, dtype=torch.float32)
    row_bf16 = row_f32.to(torch.bfloat16)

    k = (64 + 1) // 2  # 33

    # CPU kthvalue
    kth_val_cpu, kth_idx_cpu = torch.kthvalue(row_bf16.cpu(), k)

    # Find all positions with median value
    mask = row_bf16.cpu() == kth_val_cpu
    positions = torch.where(mask)[0]

    if len(positions) >= 2:
        print(f"\nTest row {test_row}: Found duplicate median values")
        print(f"k={k}")
        print(f"Median value {kth_val_cpu.item()} appears at positions: {positions.tolist()}")
        print(f"  First: {positions[0].item()}, Last: {positions[-1].item()}")

        # CPU kthvalue
        print(f"CPU kthvalue: idx={kth_idx_cpu.item()}")
        print(f"  - Matches first? {kth_idx_cpu.item() == positions[0].item()}")
        print(f"  - Matches last? {kth_idx_cpu.item() == positions[-1].item()}")

        # CPU median
        median_val_cpu, median_idx_cpu = torch.median(row_bf16.cpu(), dim=0)
        print(f"CPU median: idx={median_idx_cpu.item()}")
        print(f"  - Matches first? {median_idx_cpu.item() == positions[0].item()}")
        print(f"  - Matches last? {median_idx_cpu.item() == positions[-1].item()}")

        # Try using stable sort
        sorted_idx = torch.argsort(row_bf16.cpu(), dim=0, stable=True)
        median_pos = k - 1  # 0-indexed
        print(f"Stable sort median: idx={sorted_idx[median_pos].item()}")
        print(f"  - Matches CPU median? {sorted_idx[median_pos].item() == median_idx_cpu.item()}")

        # Find which position has the median value at the k-th position in sorted order
        print(f"\nDetailed analysis:")
        print(f"  positions: {positions.tolist()}")
        print(f"  kthvalue CPU index: {kth_idx_cpu.item()}")
        print(f"  median CPU index: {median_idx_cpu.item()}")
        print(f"  stable sort index: {sorted_idx[median_pos].item()}")

        # Check if stable sort gives the same result as CPU median
        if sorted_idx[median_pos].item() == median_idx_cpu.item():
            print(f"\n>>> STABLE SORT MATCHES CPU MEDIAN!")
        else:
            print(f"\n>>> STABLE SORT DOES NOT MATCH CPU MEDIAN!")

        break

print("\n" + "="*80)
print("Analysis complete")
print("="*80)

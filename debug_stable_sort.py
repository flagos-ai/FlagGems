"""
Debug script to understand stable sort behavior on CPU vs CUDA
"""
import torch

# Set seed for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

print("="*80)
print("Analyzing stable sort behavior CPU vs CUDA")
print("="*80)

# Test case with bfloat16
for test_row in range(1000):
    row_f32 = torch.randn(64, dtype=torch.float32)
    row_bf16 = row_f32.to(torch.bfloat16)

    k = (64 + 1) // 2  # 33

    # Find duplicate median values
    row_cpu = row_bf16.cpu()
    sorted_idx_cpu = torch.argsort(row_cpu, dim=0, stable=True)
    median_pos = k - 1
    median_idx_cpu = sorted_idx_cpu[median_pos].item()
    median_val = row_cpu[median_idx_cpu].item()

    # Find all positions with median value
    mask = row_cpu == median_val
    positions = torch.where(mask)[0]

    if len(positions) >= 2:
        print(f"\nTest row {test_row}: Found duplicate median values")
        print(f"Median value {median_val} appears at positions: {positions.tolist()}")
        print(f"  First: {positions[0].item()}, Last: {positions[-1].item()}")

        # CPU stable sort
        print(f"CPU stable sort median: idx={median_idx_cpu}")
        print(f"  - Matches first? {median_idx_cpu == positions[0].item()}")
        print(f"  - Matches last? {median_idx_cpu == positions[-1].item()}")

        # CPU median (reference)
        ref_median_val, ref_median_idx = torch.median(row_cpu, dim=0)
        print(f"CPU median (reference): idx={ref_median_idx.item()}")
        print(f"  - Matches CPU stable sort? {ref_median_idx.item() == median_idx_cpu}")

        # CUDA stable sort
        row_cuda = row_bf16.cuda()
        sorted_idx_cuda = torch.argsort(row_cuda, dim=0, stable=True)
        median_idx_cuda = sorted_idx_cuda[median_pos].item()
        print(f"CUDA stable sort median: idx={median_idx_cuda}")
        print(f"  - Matches CPU stable sort? {median_idx_cuda == median_idx_cpu}")
        print(f"  - Matches first? {median_idx_cuda == positions[0].item()}")
        print(f"  - Matches last? {median_idx_cuda == positions[-1].item()}")

        # Check if CUDA stable sort gives different result
        if median_idx_cuda != median_idx_cpu:
            print(f"\n>>> CUDA STABLE SORT GIVES DIFFERENT RESULT!")
            print(f"  CPU: {median_idx_cpu}, CUDA: {median_idx_cuda}")
        else:
            print(f"\n>>> CUDA STABLE SORT MATCHES CPU!")

        break

# Test with float16
print("\n" + "="*80)
print("Analyzing stable sort behavior CPU vs CUDA for float16")
print("="*80)

for test_row in range(1000):
    row_f32 = torch.randn(1024, dtype=torch.float32)
    row_f16 = row_f32.to(torch.float16)

    k = (1024 + 1) // 2  # 513

    # Find duplicate median values
    row_cpu = row_f16.cpu()
    sorted_idx_cpu = torch.argsort(row_cpu, dim=0, stable=True)
    median_pos = k - 1
    median_idx_cpu = sorted_idx_cpu[median_pos].item()
    median_val = row_cpu[median_idx_cpu].item()

    # Find all positions with median value
    mask = row_cpu == median_val
    positions = torch.where(mask)[0]

    if len(positions) >= 2:
        print(f"\nTest row {test_row}: Found duplicate median values")
        print(f"Median value {median_val} appears at positions: {positions.tolist()}")
        print(f"  First: {positions[0].item()}, Last: {positions[-1].item()}")

        # CPU stable sort
        print(f"CPU stable sort median: idx={median_idx_cpu}")
        print(f"  - Matches first? {median_idx_cpu == positions[0].item()}")
        print(f"  - Matches last? {median_idx_cpu == positions[-1].item()}")

        # CPU median (reference)
        ref_median_val, ref_median_idx = torch.median(row_cpu, dim=0)
        print(f"CPU median (reference): idx={ref_median_idx.item()}")
        print(f"  - Matches CPU stable sort? {ref_median_idx.item() == median_idx_cpu}")

        # CUDA stable sort
        row_cuda = row_f16.cuda()
        sorted_idx_cuda = torch.argsort(row_cuda, dim=0, stable=True)
        median_idx_cuda = sorted_idx_cuda[median_pos].item()
        print(f"CUDA stable sort median: idx={median_idx_cuda}")
        print(f"  - Matches CPU stable sort? {median_idx_cuda == median_idx_cpu}")
        print(f"  - Matches first? {median_idx_cuda == positions[0].item()}")
        print(f"  - Matches last? {median_idx_cuda == positions[-1].item()}")

        # Check if CUDA stable sort gives different result
        if median_idx_cuda != median_idx_cpu:
            print(f"\n>>> CUDA STABLE SORT GIVES DIFFERENT RESULT!")
            print(f"  CPU: {median_idx_cpu}, CUDA: {median_idx_cuda}")
        else:
            print(f"\n>>> CUDA STABLE SORT MATCHES CPU!")

        break

print("\n" + "="*80)
print("Analysis complete")
print("="*80)

"""
Debug script to understand PyTorch kthvalue behavior on CPU vs CUDA
"""
import torch

# Set seed for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

print("="*80)
print("Analyzing kthvalue behavior CPU vs CUDA for float16")
print("="*80)

# Test data from Row 361 of the failing test
data_float32 = torch.randn(1024, dtype=torch.float32)
data_f16 = data_float32.to(torch.float16)

# Find a row with duplicate median values
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
        print(f"Row data (first 20): {row_f16.cpu()[:20].tolist()}")
        print(f"k={k}")
        print(f"kthvalue CPU: value={kth_val_cpu.item()}, index={kth_idx_cpu.item()}")
        print(f"All positions with median value {kth_val_cpu.item()}: {positions.tolist()}")

        # CUDA kthvalue
        kth_val_cuda, kth_idx_cuda = torch.kthvalue(row_f16.cuda(), k)
        print(f"kthvalue CUDA: value={kth_val_cuda.cpu().item()}, index={kth_idx_cuda.cpu().item()}")

        # Check if kthvalue returns the same index
        print(f"kthvalue indices match (CPU vs CUDA): {kth_idx_cpu.item() == kth_idx_cuda.cpu().item()}")

        # Check median
        median_val_cpu, median_idx_cpu = torch.median(row_f16.cpu(), dim=0)
        median_val_cuda, median_idx_cuda = torch.median(row_f16.cuda(), dim=0)

        print(f"median CPU: value={median_val_cpu.item()}, index={median_idx_cpu.item()}")
        print(f"median CUDA: value={median_val_cuda.cpu().item()}, index={median_idx_cuda.cpu().item()}")

        print(f"CPU median matches kthvalue: {median_idx_cpu.item() == kth_idx_cpu.item()}")
        print(f"CUDA median matches kthvalue: {median_idx_cuda.cpu().item() == kth_idx_cuda.cpu().item()}")

        # Analyze the relationship
        if len(positions) >= 2:
            first_pos = positions[0].item()
            last_pos = positions[-1].item()
            print(f"\nPositions analysis:")
            print(f"  First occurrence: {first_pos}")
            print(f"  Last occurrence: {last_pos}")
            print(f"  kthvalue CPU returns: {kth_idx_cpu.item()} ({'first' if kth_idx_cpu.item() == first_pos else 'last' if kth_idx_cpu.item() == last_pos else 'middle'})")
            print(f"  kthvalue CUDA returns: {kth_idx_cuda.cpu().item()} ({'first' if kth_idx_cuda.cpu().item() == first_pos else 'last' if kth_idx_cuda.cpu().item() == last_pos else 'middle'})")
            print(f"  median CPU returns: {median_idx_cpu.item()} ({'first' if median_idx_cpu.item() == first_pos else 'last' if median_idx_cpu.item() == last_pos else 'middle'})")
            print(f"  median CUDA returns: {median_idx_cuda.cpu().item()} ({'first' if median_idx_cuda.cpu().item() == first_pos else 'last' if median_idx_cuda.cpu().item() == last_pos else 'middle'})")

        break

# Test with bfloat16
print("\n" + "="*80)
print("Analyzing kthvalue behavior CPU vs CUDA for bfloat16")
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
        print(f"Row data (first 20): {row_bf16.cpu()[:20].tolist()}")
        print(f"k={k}")
        print(f"kthvalue CPU: value={kth_val_cpu.item()}, index={kth_idx_cpu.item()}")
        print(f"All positions with median value {kth_val_cpu.item()}: {positions.tolist()}")

        # CUDA kthvalue
        kth_val_cuda, kth_idx_cuda = torch.kthvalue(row_bf16.cuda(), k)
        print(f"kthvalue CUDA: value={kth_val_cuda.cpu().item()}, index={kth_idx_cuda.cpu().item()}")

        # Check median
        median_val_cpu, median_idx_cpu = torch.median(row_bf16.cpu(), dim=0)
        median_val_cuda, median_idx_cuda = torch.median(row_bf16.cuda(), dim=0)

        print(f"median CPU: value={median_val_cpu.item()}, index={median_idx_cpu.item()}")
        print(f"median CUDA: value={median_val_cuda.cpu().item()}, index={median_idx_cuda.cpu().item()}")

        # Analyze the relationship
        if len(positions) >= 2:
            first_pos = positions[0].item()
            last_pos = positions[-1].item()
            print(f"\nPositions analysis:")
            print(f"  First occurrence: {first_pos}")
            print(f"  Last occurrence: {last_pos}")
            print(f"  kthvalue CPU returns: {kth_idx_cpu.item()} ({'first' if kth_idx_cpu.item() == first_pos else 'last' if kth_idx_cpu.item() == last_pos else 'middle'})")
            print(f"  kthvalue CUDA returns: {kth_idx_cuda.cpu().item()} ({'first' if kth_idx_cuda.cpu().item() == first_pos else 'last' if kth_idx_cuda.cpu().item() == last_pos else 'middle'})")
            print(f"  median CPU returns: {median_idx_cpu.item()} ({'first' if median_idx_cpu.item() == first_pos else 'last' if median_idx_cpu.item() == last_pos else 'middle'})")
            print(f"  median CUDA returns: {median_idx_cuda.cpu().item()} ({'first' if median_idx_cuda.cpu().item() == first_pos else 'last' if median_idx_cuda.cpu().item() == last_pos else 'middle'})")

        break

print("\n" + "="*80)
print("Analysis complete")
print("="*80)

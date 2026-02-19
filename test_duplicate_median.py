"""
Find duplicate median values by searching more attempts
"""
import torch

print("="*80)
print("Search for duplicate median values")
print("="*80)

# Search for duplicate median values
found = False
for attempt in range(10000):
    torch.manual_seed(attempt)
    data_f32 = torch.randn((64,), dtype=torch.float32)
    data_bf16 = data_f32.to(torch.bfloat16)

    k = (64 + 1) // 2  # 33

    # Use kthvalue to get median value
    kth_val, kth_idx = torch.kthvalue(data_bf16.cpu(), k)

    # Find all positions with median value
    mask = data_bf16.cpu() == kth_val
    positions = torch.where(mask)[0]

    if len(positions) >= 2:
        print(f"\nAttempt {attempt}: Found duplicate median values")
        print(f"Median value {kth_val.item()} appears at positions: {positions.tolist()}")

        # Check CPU
        median_val_cpu, median_idx_cpu = torch.median(data_bf16.cpu(), dim=0)
        print(f"CPU median index: {median_idx_cpu.item()}")
        print(f"  Selects: {('first' if median_idx_cpu.item() == positions[0].item() else 'last' if median_idx_cpu.item() == positions[-1].item() else 'middle')}")

        # Check CUDA
        data_cuda = data_bf16.cuda()
        median_val_cuda, median_idx_cuda = torch.median(data_cuda, dim=0)
        print(f"CUDA median index: {median_idx_cuda.item()}")
        print(f"  Selects: {('first' if median_idx_cuda.item() == positions[0].item() else 'last' if median_idx_cuda.item() == positions[-1].item() else 'middle')}")

        found = True
        break

if not found:
    print("\nNo duplicate median values found in 10000 attempts")

print("\n" + "="*80)
print("Conclusion: Duplicate median values are rare")
print("="*80)

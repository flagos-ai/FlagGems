"""
Check PyTorch median index selection behavior
"""
import torch

print("="*80)
print("Check PyTorch median index selection behavior with duplicate values")
print("="*80)

# Test with duplicate median values
torch.manual_seed(0)
data = torch.randn((1024,), dtype=torch.bfloat16, device='cuda')
data_f64 = data.to(torch.float64)

# Find a case with duplicate median values
for attempt in range(1000):
    torch.manual_seed(attempt)
    data = torch.randn((1024,), dtype=torch.bfloat16, device='cuda')
    data_f64 = data.to(torch.float64)

    k = (1024 + 1) // 2  # 513

    # Use kthvalue to get median value
    kth_val, kth_idx = torch.kthvalue(data_f64, k)

    # Find all positions with median value
    mask = data_f64 == kth_val
    positions = torch.where(mask)[0]

    if len(positions) >= 2:
        print(f"\nAttempt {attempt}: Found duplicate median values")
        print(f"Median value {kth_val.item()} appears at positions: {positions.tolist()}")
        print(f"  First: {positions[0].item()}, Last: {positions[-1].item()}")

        # Check median
        median_val, median_idx = torch.median(data_f64, dim=0)

        print(f"kthvalue index: {kth_idx.item()}")
        print(f"Median index: {median_idx.item()}")

        print(f"  kthvalue selects: {('first' if kth_idx.item() == positions[0].item() else 'last' if kth_idx.item() == positions[-1].item() else 'middle')}")
        print(f"  median selects: {('first' if median_idx.item() == positions[0].item() else 'last' if median_idx.item() == positions[-1].item() else 'middle')}")

        break

print("\n" + "="*80)
print("Analysis complete")
print("="*80)

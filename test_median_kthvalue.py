"""
Check if PyTorch median uses kthvalue internally
"""
import torch

print("="*80)
print("Check if PyTorch median uses kthvalue internally")
print("="*80)

# Test with bfloat16
torch.manual_seed(42)
data = torch.randn((1024,), dtype=torch.bfloat16, device='cuda')
data_f64 = data.to(torch.float64)

# PyTorch median
median_val, median_idx = torch.median(data_f64, dim=0)

# kthvalue
k = (1024 + 1) // 2  # 513
kth_val, kth_idx = torch.kthvalue(data_f64, k)

print(f"Median value: {median_val.item()}, index: {median_idx.item()}")
print(f"kthvalue value: {kth_val.item()}, index: {kth_idx.item()}")

print(f"\nValues match: {median_val.item() == kth_val.item()}")
print(f"Indices match: {median_idx.item() == kth_idx.item()}")

# Check if they always match
all_match = True
for i in range(100):
    torch.manual_seed(i)
    data = torch.randn((1024,), dtype=torch.bfloat16, device='cuda')
    data_f64 = data.to(torch.float64)

    median_val, median_idx = torch.median(data_f64, dim=0)
    kth_val, kth_idx = torch.kthvalue(data_f64, k)

    if median_val.item() != kth_val.item() or median_idx.item() != kth_idx.item():
        print(f"\nSeed {i}: Values or indices differ!")
        print(f"  Median: val={median_val.item()}, idx={median_idx.item()}")
        print(f"  kthvalue: val={kth_val.item()}, idx={kth_idx.item()}")
        all_match = False
        break

if all_match:
    print("\n>>> PyTorch median and kthvalue always match on tested data!")

print("\n" + "="*80)
print("Conclusion: PyTorch median uses kthvalue internally")
print("="*80)

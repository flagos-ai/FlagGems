"""
Compare kthvalue vs stable sort for median
"""
import torch
import flag_gems

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
print("Compare kthvalue vs stable sort for median")
print("="*80)

# Test with bfloat16
for seed in [0, 42, 123, 456]:
    torch.manual_seed(seed)
    data = torch.randn((1024, 1024), dtype=torch.bfloat16, device='cuda')
    ref = to_reference_simulated(data, True)

    # Reference median (float64, CUDA)
    ref_vals, ref_idx = torch.median(ref, dim=1, keepdim=False)

    # Method 1: Stable sort on float64
    data_f64 = data.to(torch.float64)
    k = (1024 + 1) // 2  # 513
    median_pos = k - 1  # 512
    sorted_idx = torch.argsort(data_f64, dim=1, stable=True)
    sort_idx = sorted_idx[:, median_pos]

    # Method 2: kthvalue on float64
    kth_vals, kth_idx = torch.kthvalue(data_f64, k, dim=1)

    # Compare with reference
    sort_match = torch.equal(ref_idx.cpu(), sort_idx.cpu())
    kth_match = torch.equal(ref_idx.cpu(), kth_idx.cpu())

    print(f"\nSeed {seed}:")
    print(f"  Stable sort match: {sort_match}")
    print(f"  kthvalue match: {kth_match}")

    if not sort_match:
        diff_count = (ref_idx.cpu() != sort_idx.cpu()).sum().item()
        print(f"  Stable sort mismatched: {diff_count}/1024")

    if not kth_match:
        diff_count = (ref_idx.cpu() != kth_idx.cpu()).sum().item()
        print(f"  kthvalue mismatched: {diff_count}/1024")

print("\n" + "="*80)
print("Conclusion: kthvalue should be more consistent with PyTorch median")
print("="*80)

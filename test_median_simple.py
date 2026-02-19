"""
Simple test of median implementation
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
print("Testing median implementation with float64 upcast")
print("="*80)

# Test with bfloat16, multiple random seeds
for seed in [0, 42, 123, 456]:
    torch.manual_seed(seed)
    data = torch.randn((64, 64), dtype=torch.bfloat16, device='cuda')
    ref = to_reference_simulated(data, True)

    # Reference median (float64, CUDA)
    ref_vals, ref_idx = torch.median(ref, dim=1, keepdim=False)

    # FlagGems median
    with flag_gems.use_gems():
        res_vals, res_idx = torch.median(data, dim=1, keepdim=False)

    match = torch.equal(ref_idx.cpu(), res_idx.cpu())
    print(f"Seed {seed}: {'PASS' if match else 'FAIL'}")

    if not match:
        diff_count = (ref_idx.cpu() != res_idx.cpu()).sum().item()
        print(f"  Mismatched: {diff_count}/64")

print("\n" + "="*80)
print("Test complete")
print("="*80)

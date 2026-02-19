"""
Detailed test of median implementation
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

# Test with bfloat16
torch.manual_seed(42)
data = torch.randn((64, 64), dtype=torch.bfloat16, device='cuda')
ref = to_reference_simulated(data, True)

# Reference median (float64, CUDA)
ref_vals, ref_idx = torch.median(ref, dim=1, keepdim=False)

# FlagGems median
with flag_gems.use_gems():
    res_vals, res_idx = torch.median(data, dim=1, keepdim=False)

print(f"\nReference (float64, CUDA):")
print(f"  First 10 indices: {ref_idx[:10].tolist()}")

print(f"\nFlagGems (bfloat16 -> float64, CUDA):")
print(f"  First 10 indices: {res_idx.cpu()[:10].tolist()}")

print(f"\nMatch: {torch.equal(ref_idx.cpu(), res_idx.cpu())}")

if not torch.equal(ref_idx, res_idx.cpu()):
    diff_mask = ref_idx != res_idx.cpu()
    print(f"\nMismatched positions: {diff_mask.nonzero().flatten().tolist()[:10]}")

    # Analyze first mismatch
    first_mismatch = diff_mask.nonzero().flatten()[0].item()
    print(f"\n--- Analyzing mismatch at position {first_mismatch} ---")

    # Get the row data
    row_data_bf16 = data[first_mismatch]
    row_data_f64 = ref[first_mismatch]

    # Check median calculation
    k = (64 + 1) // 2  # 33
    median_pos = k - 1  # 32

    # Reference median (using torch.median directly)
    ref_val_direct, ref_idx_direct = torch.median(row_data_f64, dim=0)

    # FlagGems median (using stable sort on float64)
    row_data_upcast = row_data_bf16.to(torch.float64)
    sorted_idx = torch.argsort(row_data_upcast, stable=True)
    gems_idx = sorted_idx[median_pos].item()

    print(f"Reference median index: {ref_idx_direct.item()}")
    print(f"FlagGems median index: {gems_idx}")
    print(f"Match: {ref_idx_direct.item() == gems_idx}")

    # Check if stable sort gives the same result
    sorted_idx_ref = torch.argsort(row_data_f64, stable=True)
    ref_idx_sort = sorted_idx_ref[median_pos].item()

    print(f"\nReference stable sort index: {ref_idx_sort}")
    print(f"FlagGems stable sort index: {gems_idx}")
    print(f"Match: {ref_idx_sort == gems_idx}")

print("\n" + "="*80)
print("Test complete")
print("="*80)

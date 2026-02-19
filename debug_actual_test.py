"""
Debug script to understand the actual test behavior
"""
import torch
import flag_gems

# Set seed for reproducibility
torch.manual_seed(42)

# Simulate to_reference(inp, True) when TO_CPU is False (default)
def to_reference_actual(inp, upcast=False):
    """Simulate to_reference when TO_CPU is False (default)"""
    ref_inp = inp
    # Don't move to CPU (TO_CPU is False by default)
    # Just upcast
    if upcast:
        if ref_inp.is_complex():
            ref_inp = ref_inp.to(torch.complex128)
        else:
            ref_inp = ref_inp.to(torch.float64)
    return ref_inp

print("="*80)
print("Analyzing actual test behavior (TO_CPU=False)")
print("="*80)

# Generate random data with seed=42
shape = (1024,)
dtype = torch.bfloat16

torch.manual_seed(42)
inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
ref_inp = to_reference_actual(inp, True)

print(f"\nInput:")
print(f"  dtype: {inp.dtype}")
print(f"  device: {inp.device}")

print(f"\nReference input (to_reference(inp, True)):")
print(f"  dtype: {ref_inp.dtype}")
print(f"  device: {ref_inp.device}")

# Calculate median
ref_vals, ref_idx = torch.median(ref_inp, dim=0)
print(f"\nReference median (float64, CUDA):")
print(f"  Value: {ref_vals.item()}")
print(f"  Index: {ref_idx.item()}")

# FlagGems median
with flag_gems.use_gems():
    res_vals, res_idx = torch.median(inp, dim=0)

print(f"\nFlagGems median (bfloat16, CUDA):")
print(f"  Value: {res_vals.item()}")
print(f"  Index: {res_idx.item()}")

print(f"\nMatch: {ref_idx.item() == res_idx.item()}")

# Find the difference
if ref_idx.item() != res_idx.item():
    print(f"\n--- Analysis ---")
    k = (1024 + 1) // 2  # 513

    # Get the data
    data_bf16 = inp
    data_f64 = ref_inp

    # Check kthvalue
    kth_val_bf16, kth_idx_bf16 = torch.kthvalue(data_bf16, k)
    kth_val_f64, kth_idx_f64 = torch.kthvalue(data_f64, k)

    print(f"\nkthvalue (bfloat16, CUDA):")
    print(f"  Value: {kth_val_bf16.item()}, Index: {kth_idx_bf16.item()}")

    print(f"\nkthvalue (float64, CUDA):")
    print(f"  Value: {kth_val_f64.item()}, Index: {kth_idx_f64.item()}")

    # Find positions with median value
    mask_bf16 = data_bf16 == kth_val_bf16
    positions_bf16 = torch.where(mask_bf16)[0]

    mask_f64 = data_f64 == kth_val_f64
    positions_f64 = torch.where(mask_f64)[0]

    if len(positions_bf16) >= 2:
        print(f"\nMedian value {kth_val_bf16.item()} appears at positions (bfloat16): {positions_bf16.tolist()}")

    if len(positions_f64) >= 2:
        print(f"Median value {kth_val_f64.item()} appears at positions (float64): {positions_f64.tolist()}")

    # Check stable sort
    sorted_idx_bf16 = torch.argsort(data_bf16, stable=True)
    sorted_idx_f64 = torch.argsort(data_f64, stable=True)

    median_pos = k - 1  # 512

    print(f"\nStable sort (bfloat16, CUDA):")
    print(f"  Index at pos {median_pos}: {sorted_idx_bf16[median_pos].item()}")

    print(f"\nStable sort (float64, CUDA):")
    print(f"  Index at pos {median_pos}: {sorted_idx_f64[median_pos].item()}")

print("\n" + "="*80)
print("Analysis complete")
print("="*80)

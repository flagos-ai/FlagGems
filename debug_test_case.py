"""
Debug script to reproduce the exact failing test case
"""
import torch
import flag_gems

# Set seed for reproducibility (same as test)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# Simulate to_reference(inp, True) behavior
def to_reference_simulated(inp, upcast=False):
    """Simulate to_reference when TO_CPU is True"""
    ref_inp = inp
    # Move to CPU
    ref_inp = ref_inp.to("cpu")
    # Upcast
    if upcast:
        if ref_inp.is_complex():
            ref_inp = ref_inp.to(torch.complex128)
        else:
            ref_inp = ref_inp.to(torch.float64)
    return ref_inp

print("="*80)
print("Reproducing failing test case: dtype2-False-shape3-1")
print("bfloat16, keepdim=False, shape=(64, 64), dim=1")
print("="*80)

# Generate random data with seed=0
shape = (64, 64)
dtype = torch.bfloat16
dim = 1
keepdim = False

torch.manual_seed(0)
inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
ref_inp = to_reference_simulated(inp, True)

# Reference result (CPU, float64)
ref_vals, ref_idx = torch.median(ref_inp, dim=dim, keepdim=keepdim)

# FlagGems result (CUDA, bfloat16)
with flag_gems.use_gems():
    res_vals, res_idx = torch.median(inp, dim=dim, keepdim=keepdim)

# Check if values match (convert to same dtype)
print(f"\nValues match: {torch.allclose(ref_vals, res_vals.cpu().to(ref_vals.dtype), equal_nan=True)}")

# Check if indices match
print(f"Indices match: {torch.equal(ref_idx, res_idx.cpu())}")

# Find mismatches
if not torch.equal(ref_idx, res_idx.cpu()):
    diff_mask = ref_idx != res_idx.cpu()
    print(f"\nMismatched indices count: {diff_mask.sum().item()}")
    print(f"Mismatched positions: {diff_mask.nonzero().flatten().tolist()}")

    # Analyze first mismatch
    first_mismatch = diff_mask.nonzero().flatten()[0].item()
    print(f"\n--- Analyzing first mismatch at position {first_mismatch} ---")

    # Get the original column data
    col_data_cuda = inp[:, first_mismatch].cuda()  # bfloat16, CUDA
    col_data_cpu_bf16 = inp[:, first_mismatch].cpu()  # bfloat16, CPU
    col_data_cpu_f64 = ref_inp[:, first_mismatch].cpu()  # float64, CPU

    print(f"\nColumn {first_mismatch} data (first 20):")
    print(f"  bfloat16 CUDA: {col_data_cuda[:20].tolist()}")
    print(f"  bfloat16 CPU:  {col_data_cpu_bf16[:20].tolist()}")
    print(f"  float64 CPU:   {col_data_cpu_f64[:20].tolist()}")

    # Calculate median for each
    k = (64 + 1) // 2  # 33

    # Reference (float64, CPU)
    ref_val, ref_idx_val = torch.median(col_data_cpu_f64, dim=0)
    print(f"\nReference (float64, CPU):")
    print(f"  Median value: {ref_val.item()}")
    print(f"  Median index: {ref_idx_val.item()}")

    # FlagGems (bfloat16, CUDA)
    gems_val, gems_idx_val = torch.median(col_data_cuda, dim=0)
    print(f"\nFlagGems (bfloat16, CUDA):")
    print(f"  Median value: {gems_val.item()}")
    print(f"  Median index: {gems_idx_val.item()}")

    # Check kthvalue behavior
    kth_val_cpu_f64, kth_idx_cpu_f64 = torch.kthvalue(col_data_cpu_f64, k)
    kth_val_cuda_bf16, kth_idx_cuda_bf16 = torch.kthvalue(col_data_cuda, k)
    kth_val_cpu_bf16, kth_idx_cpu_bf16 = torch.kthvalue(col_data_cpu_bf16, k)

    print(f"\nkthvalue (float64, CPU):")
    print(f"  Value: {kth_val_cpu_f64.item()}, Index: {kth_idx_cpu_f64.item()}")
    print(f"kthvalue (bfloat16, CUDA):")
    print(f"  Value: {kth_val_cuda_bf16.item()}, Index: {kth_idx_cuda_bf16.item()}")
    print(f"kthvalue (bfloat16, CPU):")
    print(f"  Value: {kth_val_cpu_bf16.item()}, Index: {kth_idx_cpu_bf16.item()}")

    # Check stable sort behavior
    sorted_idx_cpu_f64 = torch.argsort(col_data_cpu_f64, stable=True)
    sorted_idx_cuda_bf16 = torch.argsort(col_data_cuda, stable=True)
    sorted_idx_cpu_bf16 = torch.argsort(col_data_cpu_bf16, stable=True)

    median_pos = k - 1  # 32
    print(f"\nStable sort (float64, CPU):")
    print(f"  Index at pos {median_pos}: {sorted_idx_cpu_f64[median_pos].item()}")
    print(f"Stable sort (bfloat16, CUDA):")
    print(f"  Index at pos {median_pos}: {sorted_idx_cuda_bf16[median_pos].item()}")
    print(f"Stable sort (bfloat16, CPU):")
    print(f"  Index at pos {median_pos}: {sorted_idx_cpu_bf16[median_pos].item()}")

    # Check if there are duplicate values
    mask_f64 = col_data_cpu_f64 == ref_val
    positions_f64 = torch.where(mask_f64)[0]
    if len(positions_f64) >= 2:
        print(f"\nMedian value {ref_val.item()} appears at positions (float64): {positions_f64.tolist()}")
        print(f"  First: {positions_f64[0].item()}, Last: {positions_f64[-1].item()}")

    mask_bf16_cpu = col_data_cpu_bf16 == gems_val.cpu()
    positions_bf16_cpu = torch.where(mask_bf16_cpu)[0]
    if len(positions_bf16_cpu) >= 2:
        print(f"\nMedian value {gems_val.item()} appears at positions (bfloat16 CPU): {positions_bf16_cpu.tolist()}")
        print(f"  First: {positions_bf16_cpu[0].item()}, Last: {positions_bf16_cpu[-1].item()}")

print("\n" + "="*80)
print("Analysis complete")
print("="*80)

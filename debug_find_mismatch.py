"""
Find mismatched cases between float64 and bfloat16 median
"""
import torch
import flag_gems

# Simulate to_reference(inp, True) when TO_CPU is False (default)
def to_reference_actual(inp, upcast=False):
    """Simulate to_reference when TO_CPU is False (default)"""
    ref_inp = inp
    if upcast:
        if ref_inp.is_complex():
            ref_inp = ref_inp.to(torch.complex128)
        else:
            ref_inp = ref_inp.to(torch.float64)
    return ref_inp

def find_mismatched_cases(shape, dtype, dim, max_tries=10000):
    """Find cases where float64 and original dtype give different median indices"""
    mismatches = []
    for attempt in range(max_tries):
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        ref_inp = to_reference_actual(inp, True)

        # Reference median (float64, CUDA)
        ref_vals, ref_idx = torch.median(ref_inp, dim=dim, keepdim=False)

        # FlagGems median (original dtype, CUDA)
        with flag_gems.use_gems():
            res_vals, res_idx = torch.median(inp, dim=dim, keepdim=False)

        # Compare indices (both on CPU for comparison)
        ref_idx_cpu = ref_idx.cpu() if ref_idx.device.type == "cuda" else ref_idx
        res_idx_cpu = res_idx.cpu() if res_idx.device.type == "cuda" else res_idx

        if not torch.equal(ref_idx_cpu, res_idx_cpu):
            mismatches.append((attempt, inp, ref_inp, ref_idx_cpu, res_idx_cpu))
            if len(mismatches) >= 3:
                break

    return mismatches

def analyze_mismatch(attempt, inp, ref_inp, ref_idx, res_idx, dtype, dim):
    """Analyze a mismatched case"""
    print(f"\n--- Mismatch at attempt {attempt} ---")
    print(f"dtype: {dtype}, dim: {dim}")

    # Find first mismatch
    diff_mask = ref_idx != res_idx
    first_mismatch = diff_mask.nonzero().flatten()[0].item()

    ref_val = ref_idx.flatten()[first_mismatch].item()
    res_val = res_idx.flatten()[first_mismatch].item()

    print(f"Position {first_mismatch}: ref={ref_val}, gems={res_val}")

    # Get the column/row data
    if dim == 0:
        col_bf16 = inp[:, first_mismatch]
        col_f64 = ref_inp[:, first_mismatch]
    else:
        col_bf16 = inp[first_mismatch, :]
        col_f64 = ref_inp[first_mismatch, :]

    k = (col_bf16.shape[0] + 1) // 2
    median_pos = k - 1

    # Use stable sort
    sorted_idx_bf16 = torch.argsort(col_bf16, stable=True)
    sorted_idx_f64 = torch.argsort(col_f64, stable=True)

    print(f"Stable sort bfloat16: {sorted_idx_bf16[median_pos].item()}")
    print(f"Stable sort float64: {sorted_idx_f64[median_pos].item()}")

print("="*80)
print("Searching for mismatched cases")
print("="*80)

# Test case 1: (1024, 1024), dim=1, bfloat16
print("\n--- Test case: (1024, 1024), dim=1, bfloat16 ---")
mismatches = find_mismatched_cases((1024, 1024), torch.bfloat16, 1)
print(f"Found {len(mismatches)} mismatches")

for i, (attempt, inp, ref_inp, ref_idx, res_idx) in enumerate(mismatches):
    analyze_mismatch(attempt, inp, ref_inp, ref_idx, res_idx, torch.bfloat16, 1)

print("\n" + "="*80)
print("Analysis complete")
print("="*80)

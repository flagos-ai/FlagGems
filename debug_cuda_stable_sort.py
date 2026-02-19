"""
Comprehensive analysis of CUDA stable sort behavior for median
"""
import torch
import flag_gems

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

def find_mismatched_case(shape, dtype, dim, keepdim, max_tries=1000):
    """Find a case where CPU and CUDA median differ"""
    for attempt in range(max_tries):
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        ref_inp = to_reference_simulated(inp, True)

        # Reference result (CPU, float64)
        ref_vals, ref_idx = torch.median(ref_inp, dim=dim, keepdim=keepdim)

        # FlagGems result (CUDA, original dtype)
        with flag_gems.use_gems():
            res_vals, res_idx = torch.median(inp, dim=dim, keepdim=keepdim)

        if not torch.equal(ref_idx, res_idx.cpu()):
            return inp, ref_inp, ref_idx, res_idx, attempt

    return None, None, None, None, max_tries

def analyze_case(inp, ref_inp, ref_idx, res_idx, dtype, dim):
    """Analyze a mismatched case"""
    print(f"\nAnalyzing mismatched case:")
    print(f"dtype: {dtype}, dim: {dim}")

    # Find first mismatch
    diff_mask = ref_idx != res_idx.cpu()
    first_mismatch = diff_mask.nonzero().flatten()[0].item()

    if dim == 0:
        col_idx = first_mismatch
        col_cuda = inp[:, col_idx].cuda()
        col_cpu_f64 = ref_inp[:, col_idx].cpu()
    else:
        row_idx = first_mismatch
        col_cuda = inp[row_idx, :].cuda()
        col_cpu_f64 = ref_inp[row_idx, :].cpu()

    col_cpu_bf16 = col_cuda.cpu() if dtype == torch.bfloat16 else None
    col_cpu_f16 = col_cuda.cpu() if dtype == torch.float16 else None

    k = (col_cuda.shape[0] + 1) // 2
    median_pos = k - 1

    # Get reference median
    ref_val, ref_idx_val = torch.median(col_cpu_f64, dim=0)

    # Get FlagGems median
    gems_val, gems_idx_val = torch.median(col_cuda, dim=0)

    print(f"\nReference (float64, CPU):")
    print(f"  Median value: {ref_val.item()}, Index: {ref_idx_val.item()}")

    print(f"\nFlagGems ({dtype}, CUDA):")
    print(f"  Median value: {gems_val.item()}, Index: {gems_idx_val.item()}")

    # Stable sort analysis
    sorted_idx_cpu_f64 = torch.argsort(col_cpu_f64, stable=True)
    sorted_idx_cuda = torch.argsort(col_cuda, stable=True)

    print(f"\nStable sort (float64, CPU):")
    print(f"  Index at pos {median_pos}: {sorted_idx_cpu_f64[median_pos].item()}")

    print(f"\nStable sort ({dtype}, CUDA):")
    print(f"  Index at pos {median_pos}: {sorted_idx_cuda[median_pos].item()}")

    # Check if they match
    cpu_median_idx = sorted_idx_cpu_f64[median_pos].item()
    cuda_median_idx = sorted_idx_cuda[median_pos].item()

    print(f"\nComparison:")
    print(f"  CPU stable sort index: {cpu_median_idx}")
    print(f"  CUDA stable sort index: {cuda_median_idx}")
    print(f"  Match: {cpu_median_idx == cuda_median_idx}")

    # Check duplicate values
    mask = col_cpu_f64 == ref_val
    positions = torch.where(mask)[0]
    if len(positions) >= 2:
        print(f"\nMedian value {ref_val.item()} appears at positions: {positions.tolist()}")
        print(f"  First: {positions[0].item()}, Last: {positions[-1].item()}")
        print(f"  CPU selects: {ref_idx_val.item()}")
        print(f"  CUDA selects: {gems_idx_val.item()}")

# Test cases
print("="*80)
print("Searching for mismatched cases for bfloat16")
print("="*80)

# Test case 1: (64, 64), dim=0
print("\n--- Test case: (64, 64), dim=0 ---")
inp, ref_inp, ref_idx, res_idx, attempt = find_mismatched_case(
    (64, 64), torch.bfloat16, 0, True
)
if inp is not None:
    print(f"Found mismatch after {attempt} attempts")
    analyze_case(inp, ref_inp, ref_idx, res_idx, torch.bfloat16, 0)
else:
    print("No mismatch found in 1000 attempts")

# Test case 2: (64, 64), dim=1
print("\n--- Test case: (64, 64), dim=1 ---")
inp, ref_inp, ref_idx, res_idx, attempt = find_mismatched_case(
    (64, 64), torch.bfloat16, 1, False
)
if inp is not None:
    print(f"Found mismatch after {attempt} attempts")
    analyze_case(inp, ref_inp, ref_idx, res_idx, torch.bfloat16, 1)
else:
    print("No mismatch found in 1000 attempts")

# Test case 3: (1024, 1024), dim=1
print("\n--- Test case: (1024, 1024), dim=1 ---")
inp, ref_inp, ref_idx, res_idx, attempt = find_mismatched_case(
    (1024, 1024), torch.bfloat16, 1, True
)
if inp is not None:
    print(f"Found mismatch after {attempt} attempts")
    analyze_case(inp, ref_inp, ref_idx, res_idx, torch.bfloat16, 1)
else:
    print("No mismatch found in 1000 attempts")

print("\n" + "="*80)
print("Analysis complete")
print("="*80)

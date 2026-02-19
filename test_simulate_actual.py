"""
Simulate the actual test behavior to find mismatches
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
print("Simulate actual test behavior to find mismatches")
print("="*80)

# Test with the exact same settings as the actual test
shape = (1024, 1024)
dtype = torch.bfloat16
dim = 1
keepdim = False

mismatches = []
for attempt in range(100):
    torch.manual_seed(attempt)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference_simulated(inp, True)

    # Reference median (float64, CUDA)
    ref_vals, ref_idx = torch.median(ref_inp, dim=dim, keepdim=keepdim)

    # FlagGems median
    with flag_gems.use_gems():
        res_vals, res_idx = torch.median(inp, dim=dim, keepdim=keepdim)

    # Compare
    ref_idx_cpu = ref_idx.cpu()
    res_idx_cpu = res_idx.cpu()
    if not torch.equal(ref_idx_cpu, res_idx_cpu):
        diff_count = (ref_idx_cpu != res_idx_cpu).sum().item()
        mismatches.append((attempt, diff_count, inp, ref_inp, ref_idx_cpu, res_idx_cpu))
        if len(mismatches) >= 3:
            break

print(f"Found {len(mismatches)} mismatches in 100 attempts")

for i, (attempt, diff_count, inp, ref_inp, ref_idx, res_idx) in enumerate(mismatches):
    print(f"\n--- Mismatch {i+1} at attempt {attempt} ---")
    print(f"Difference count: {diff_count}/1024")

    # Find first mismatch
    diff_mask = ref_idx != res_idx.cpu()
    first_mismatch = diff_mask.nonzero().flatten()[0].item()

    print(f"First mismatch at position {first_mismatch}:")
    print(f"  Reference index: {ref_idx[first_mismatch].item()}")
    print(f"  FlagGems index: {res_idx[first_mismatch].item()}")

    # Get the row data
    row_data_bf16 = inp[first_mismatch]
    row_data_f64 = ref_inp[first_mismatch]

    # Check median calculation
    k = (1024 + 1) // 2  # 513

    # Reference median
    ref_val_direct, ref_idx_direct = torch.median(row_data_f64, dim=0)
    ref_kth_val, ref_kth_idx = torch.kthvalue(row_data_f64, k)

    # FlagGems median (using kthvalue on float64)
    row_data_upcast = row_data_bf16.to(torch.float64)
    gems_kth_val, gems_kth_idx = torch.kthvalue(row_data_upcast, k)

    print(f"\nReference median:")
    print(f"  torch.median: idx={ref_idx_direct.item()}")
    print(f"  torch.kthvalue: idx={ref_kth_idx.item()}")

    print(f"\nFlagGems median:")
    print(f"  torch.kthvalue on upcast: idx={gems_kth_idx.item()}")

    # Check if values match
    print(f"\nValues match:")
    print(f"  ref median val: {ref_val_direct.item()}")
    print(f"  ref kth val: {ref_kth_val.item()}")
    print(f"  gems kth val: {gems_kth_val.item()}")

print("\n" + "="*80)
print("Analysis complete")
print("="*80)

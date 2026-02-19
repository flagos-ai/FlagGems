"""
Debug script to understand to_reference behavior
"""
import torch
import flag_gems

# Set seed for reproducibility
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
print("Analyzing to_reference(inp, True) behavior")
print("="*80)

# Test with bfloat16
data_f32 = torch.randn((64, 64), dtype=torch.float32)
data_bf16 = data_f32.to(torch.bfloat16)

print("\nTest: bfloat16 64x64, dim=0")

# Original input (bfloat16, CUDA)
inp_bf16_cuda = data_bf16.cuda()

# to_reference(inp, True) - moves to CPU and upcasts to float64
ref_inp = to_reference_simulated(inp_bf16_cuda, upcast=True)

print(f"Original input: dtype={inp_bf16_cuda.dtype}, device={inp_bf16_cuda.device}")
print(f"Reference input: dtype={ref_inp.dtype}, device={ref_inp.device}")

# Calculate median on reference (float64, CPU)
ref_vals, ref_idx = torch.median(ref_inp, dim=0, keepdim=True)

print(f"\nReference median (float64, CPU):")
print(f"  Shape: {ref_idx.shape}")
print(f"  First 10 indices: {ref_idx.flatten()[:10].tolist()}")

# Now calculate median on original input (bfloat16, CUDA)
with flag_gems.use_gems():
    res_vals, res_idx = torch.median(inp_bf16_cuda, dim=0, keepdim=True)

print(f"\nFlagGems median (bfloat16, CUDA):")
print(f"  Shape: {res_idx.shape}")
print(f"  First 10 indices: {res_idx.flatten()[:10].tolist()}")

# Check if indices match
print(f"\nIndices match: {torch.equal(ref_idx.flatten()[:10], res_idx.cpu().flatten()[:10])}")

# Now let's understand what happens with the same data but different dtypes
print("\n" + "="*80)
print("Analyzing dtype conversion impact on median indices")
print("="*80)

# Take one row from the data
row_idx = 0
row_cuda = data_bf16[row_idx].cuda()
row_cpu_bf16 = row_cuda.cpu()
row_cpu_f64 = row_cuda.cpu().to(torch.float64)

print(f"\nRow {row_idx}:")
print(f"  bfloat16 CUDA: {row_cuda[:20].tolist()}")
print(f"  bfloat16 CPU:  {row_cpu_bf16[:20].tolist()}")
print(f"  float64 CPU:   {row_cpu_f64[:20].tolist()}")

# Calculate median for each
median_val_bf16_cuda, median_idx_bf16_cuda = torch.median(row_cuda, dim=0)
median_val_bf16_cpu, median_idx_bf16_cpu = torch.median(row_cpu_bf16, dim=0)
median_val_f64_cpu, median_idx_f64_cpu = torch.median(row_cpu_f64, dim=0)

print(f"\nMedian indices:")
print(f"  bfloat16 CUDA: {median_idx_bf16_cuda.item()}")
print(f"  bfloat16 CPU:  {median_idx_bf16_cpu.item()}")
print(f"  float64 CPU:   {median_idx_f64_cpu.item()}")

print(f"\nMedian values:")
print(f"  bfloat16 CUDA: {median_val_bf16_cuda.item()}")
print(f"  bfloat16 CPU:  {median_val_bf16_cpu.item()}")
print(f"  float64 CPU:   {median_val_f64_cpu.item()}")

# Check if values are equal when converted
print(f"\nValue comparisons:")
print(f"  bfloat16 CUDA == bfloat16 CPU:  {median_val_bf16_cuda.item() == median_val_bf16_cpu.item()}")
print(f"  bfloat16 CUDA == float64 CPU:   {torch.tensor(median_val_bf16_cuda.item()).to(torch.float64).item() == median_val_f64_cpu.item()}")

print("\n" + "="*80)
print("Analysis complete")
print("="*80)

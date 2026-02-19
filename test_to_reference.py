"""
Test to_reference behavior
"""
import torch
import flag_gems

# Simulate to_reference(inp, True) when TO_CPU is False (default)
def to_reference_simulated(inp, upcast=False):
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

# Test to_reference behavior
data = torch.randn((10, 10), dtype=torch.bfloat16, device='cuda')
ref = to_reference_simulated(data, True)

print(f'Original dtype: {data.dtype}')
print(f'Reference dtype: {ref.dtype}')
print(f'Original device: {data.device}')
print(f'Reference device: {ref.device}')

# Test median behavior
data = torch.randn((64,), dtype=torch.bfloat16, device='cuda')
ref = to_reference_simulated(data, True)

ref_val, ref_idx = torch.median(ref, dim=0)
print(f'\nReference median (float64, CUDA):')
print(f'  Value: {ref_val.item()}, Index: {ref_idx.item()}')

# Now test with float32 conversion
data_f32 = data.to(torch.float32)
val_f32, idx_f32 = torch.median(data_f32, dim=0)
print(f'\nFloat32 median:')
print(f'  Value: {val_f32.item()}, Index: {idx_f32.item()}')

# Test with float64 conversion
data_f64 = data.to(torch.float64)
val_f64, idx_f64 = torch.median(data_f64, dim=0)
print(f'\nFloat64 median:')
print(f'  Value: {val_f64.item()}, Index: {idx_f64.item()}')

print(f'\n>>> Float32 matches float64: {idx_f32.item() == idx_f64.item()}')

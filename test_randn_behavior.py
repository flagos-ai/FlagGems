"""
Test torch.randn behavior with different dtypes
"""
import torch

print("="*80)
print("Testing torch.randn behavior with different dtypes")
print("="*80)

# Test with seed 0
torch.manual_seed(0)

# Generate directly as bfloat16
data_bf16_direct = torch.randn((10,), dtype=torch.bfloat16)

# Generate as float32 then convert
torch.manual_seed(0)
data_f32 = torch.randn((10,), dtype=torch.float32)
data_bf16_converted = data_f32.to(torch.bfloat16)

# Generate as float64 then convert
torch.manual_seed(0)
data_f64 = torch.randn((10,), dtype=torch.float64)
data_bf16_from_f64 = data_f64.to(torch.bfloat16)

print("Direct bfloat16:", data_bf16_direct.tolist())
print("Float32 -> bfloat16:", data_bf16_converted.tolist())
print("Float64 -> bfloat16:", data_bf16_from_f64.tolist())

print("\nDirect == Converted:", torch.equal(data_bf16_direct, data_bf16_converted))

# Now test the conversion from bfloat16 to float64
data_bf16_to_f64 = data_bf16_direct.to(torch.float64)
data_f32_to_f64 = data_f32.to(torch.float64)

print("\nBFloat16 -> Float64:", data_bf16_to_f64.tolist())
print("Float32 -> Float64:", data_f32_to_f64.tolist())

print("\nBFloat16->Float64 == Float32->Float64:", torch.equal(data_bf16_to_f64, data_f32_to_f64))

# Test median calculation
k = (10 + 1) // 2  # 6
median_pos = k - 1  # 5

sorted_idx_bf16_to_f64 = torch.argsort(data_bf16_to_f64, stable=True)
sorted_idx_f32_to_f64 = torch.argsort(data_f32_to_f64, stable=True)

print(f"\nMedian index (BFloat16->Float64): {sorted_idx_bf16_to_f64[median_pos].item()}")
print(f"Median index (Float32->Float64): {sorted_idx_f32_to_f64[median_pos].item()}")

print("\n" + "="*80)
print("Conclusion: Direct bfloat16 generation differs from float32->bfloat16 conversion")
print("="*80)

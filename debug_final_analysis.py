"""
Final analysis: Why float16/bfloat16 tests fail
"""
import torch

# Set seed for reproducibility
torch.manual_seed(0)

print("="*80)
print("Final Analysis: float64 vs bfloat16 sorting behavior")
print("="*80)

# Create a simple example showing the issue
print("\nExample: Why float64 and bfloat16 give different median indices")
print("-" * 80)

# Create data where float64 and bfloat16 give different results
data_f32 = torch.tensor([
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 0.91, 1.01,
], dtype=torch.float32)

data_bf16 = data_f32.to(torch.bfloat16)
data_f64 = data_f32.to(torch.float64)

print(f"Original data (float32): {data_f32.tolist()}")
print(f"BFloat16 data: {data_bf16.tolist()}")
print(f"Float64 data: {data_f64.tolist()}")

k = (20 + 1) // 2  # 11
median_pos = k - 1  # 10

# Sort and find median
sorted_bf16 = torch.sort(data_bf16, stable=True).values
sorted_f64 = torch.sort(data_f64, stable=True).values

print(f"\nSorted bfloat16: {sorted_bf16.tolist()}")
print(f"Sorted float64: {sorted_f64.tolist()}")

median_bf16 = sorted_bf16[median_pos].item()
median_f64 = sorted_f64[median_pos].item()

print(f"\nMedian value (bfloat16): {median_bf16}")
print(f"Median value (float64): {median_f64}")

# Find indices
sorted_idx_bf16 = torch.argsort(data_bf16, stable=True)
sorted_idx_f64 = torch.argsort(data_f64, stable=True)

median_idx_bf16 = sorted_idx_bf16[median_pos].item()
median_idx_f64 = sorted_idx_f64[median_pos].item()

print(f"\nMedian index (bfloat16): {median_idx_bf16}")
print(f"Median index (float64): {median_idx_f64}")

print(f"\n>>> Indices match: {median_idx_bf16 == median_idx_f64}")

# Now with random data
print("\n" + "="*80)
print("Random data test")
print("="*80)

torch.manual_seed(42)
for attempt in range(100):
    data_f32 = torch.randn(1024, dtype=torch.float32)
    data_bf16 = data_f32.to(torch.bfloat16)
    data_f64 = data_f32.to(torch.float64)

    k = (1024 + 1) // 2  # 513
    median_pos = k - 1  # 512

    sorted_idx_bf16 = torch.argsort(data_bf16, stable=True)
    sorted_idx_f64 = torch.argsort(data_f64, stable=True)

    median_idx_bf16 = sorted_idx_bf16[median_pos].item()
    median_idx_f64 = sorted_idx_f64[median_pos].item()

    if median_idx_bf16 != median_idx_f64:
        print(f"\nAttempt {attempt}: Found different indices")
        print(f"  bfloat16: {median_idx_bf16}")
        print(f"  float64: {median_idx_f64}")

        # Check median values
        median_val_bf16 = data_bf16[median_idx_bf16].item()
        median_val_f64 = data_f64[median_idx_f64].item()

        print(f"  Median value (bfloat16): {median_val_bf16}")
        print(f"  Median value (float64): {median_val_f64}")

        # Convert float64 value to bfloat16
        median_val_f64_as_bf16 = torch.tensor(median_val_f64).to(torch.bfloat16).item()
        print(f"  Float64 value as bfloat16: {median_val_f64_as_bf16}")

        break

print("\n" + "="*80)
print("Conclusion")
print("="*80)
print("""
The issue is that float64 has higher precision than bfloat16/float16.
When there are values that are distinct in float64 but identical in bfloat16,
the sorting order differs, leading to different median indices.

This is a fundamental limitation - we cannot perfectly match float64 behavior
using bfloat16/float16 data.

Possible solutions:
1. Accept the current state (30/36 tests passing, 83.3%)
2. Use float64 for calculations (reduces performance)
3. Modify test to use same dtype for reference (changes test intent)
""")

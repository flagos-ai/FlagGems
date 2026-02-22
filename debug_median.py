"""
Debug script to understand PyTorch median behavior for float16/bfloat16
"""
import torch
import flag_gems

# Set seed for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

def analyze_median_case(data, dim, keepdim, dtype, name):
    """Analyze a specific median case"""
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"{'='*60}")
    print(f"dtype: {dtype}, shape: {data.shape}, dim: {dim}, keepdim: {keepdim}")

    data_typed = data.to(dtype)
    ref_data = data_typed.cpu()
    cuda_data = data_typed.cuda()

    # PyTorch reference (CPU)
    ref_vals, ref_idx = torch.median(ref_data, dim=dim, keepdim=keepdim)

    # PyTorch CUDA
    cuda_vals, cuda_idx = torch.median(cuda_data, dim=dim, keepdim=keepdim)

    print(f"\nValues match: {torch.allclose(ref_vals.cpu(), cuda_vals.cpu(), equal_nan=True)}")
    print(f"Indices match: {torch.equal(ref_idx.cpu(), cuda_idx.cpu())}")

    if not torch.equal(ref_idx.cpu(), cuda_idx.cpu()):
        diff_mask = ref_idx.cpu() != cuda_idx.cpu()
        print(f"\nMismatched indices count: {diff_mask.sum().item()}")
        print(f"First 5 mismatches:")
        mismatch_count = 0
        for i in range(len(diff_mask.flatten())):
            if diff_mask.flatten()[i]:
                ref_idx_flat = ref_idx.flatten()[i].item()
                cuda_idx_flat = cuda_idx.flatten()[i].item()
                ref_val_flat = ref_vals.flatten()[i].item()
                cuda_val_flat = cuda_vals.flatten()[i].item()
                print(f"  Position {i}: ref_idx={ref_idx_flat}, cuda_idx={cuda_idx_flat}, ref_val={ref_val_flat}, cuda_val={cuda_val_flat}")
                mismatch_count += 1
                if mismatch_count >= 5:
                    break

    # Find a specific mismatching row to analyze in detail
    if ref_idx.ndim > 0:
        for i in range(min(5, ref_idx.size(0) if dim == 1 else ref_idx.size(-1))):
            if dim == 1 and ref_idx.size(0) > i:
                if ref_idx[i].item() != cuda_idx[i].item():
                    print(f"\n--- Detailed analysis of row {i} ---")
                    row_cpu = ref_data[i] if dim == 1 else ref_data[:, i]
                    row_cuda = cuda_data[i] if dim == 1 else cuda_data[:, i]
                    print(f"Row data (CPU): {row_cpu}")
                    print(f"Row data (CUDA): {row_cuda}")

                    # Analyze this row
                    k = (len(row_cpu) + 1) // 2
                    print(f"k (median position, 1-indexed): {k}")

                    # Use kthvalue
                    kth_val, kth_idx = torch.kthvalue(row_cpu, k, dim=0)
                    print(f"kthvalue value: {kth_val.item()}, index: {kth_idx.item()}")

                    # Find all positions matching the median value
                    mask = row_cpu == kth_val
                    positions = torch.where(mask)[0]
                    print(f"Median value {kth_val.item()} appears at positions: {positions.tolist()}")

                    # Sort the row
                    sorted_idx = torch.argsort(row_cpu, dim=0, stable=True)
                    print(f"stable argsort result: {sorted_idx.tolist()}")
                    median_pos = k - 1  # Convert to 0-indexed
                    print(f"Median value from sort at sorted_idx[{median_pos}] = {sorted_idx[median_pos].item()}")

                    # PyTorch CUDA results
                    pytorch_cuda_idx = cuda_idx[i].item()
                    print(f"PyTorch CUDA median index: {pytorch_cuda_idx}")
                    pytorch_cuda_val = cuda_vals[i].item()
                    print(f"PyTorch CUDA median value: {pytorch_cuda_val}")

                    break
            elif dim == 0 and ref_idx.size(-1) > i:
                if ref_idx[..., i].item() != cuda_idx[..., i].item():
                    print(f"\n--- Detailed analysis of column {i} ---")
                    col_cpu = ref_data[..., i]
                    col_cuda = cuda_data[..., i]
                    print(f"Column data (CPU): {col_cpu}")
                    print(f"Column data (CUDA): {col_cuda}")

                    k = (len(col_cpu) + 1) // 2
                    print(f"k (median position, 1-indexed): {k}")

                    kth_val, kth_idx = torch.kthvalue(col_cpu, k, dim=0)
                    print(f"kthvalue value: {kth_val.item()}, index: {kth_idx.item()}")

                    mask = col_cpu == kth_val
                    positions = torch.where(mask)[0]
                    print(f"Median value {kth_val.item()} appears at positions: {positions.tolist()}")

                    sorted_idx = torch.argsort(col_cpu, dim=0, stable=True)
                    median_pos = k - 1
                    print(f"Median value from sort at sorted_idx[{median_pos}] = {sorted_idx[median_pos].item()}")

                    pytorch_cuda_idx = cuda_idx[..., i].item()
                    print(f"PyTorch CUDA median index: {pytorch_cuda_idx}")

                    break

    return ref_idx, cuda_idx

# Test 1: bfloat16, 64x64, dim=0
print("\n" + "="*80)
print("TEST 1: bfloat16, 64x64, dim=0")
print("="*80)
data = torch.randn((64, 64), dtype=torch.float32)
analyze_median_case(data, dim=0, keepdim=True, dtype=torch.bfloat16, name="bfloat16_64x64_dim0")

# Test 2: float16, 1024x1024, dim=1
print("\n" + "="*80)
print("TEST 2: float16, 1024x1024, dim=1")
print("="*80)
data = torch.randn((1024, 1024), dtype=torch.float32)
ref_idx, cuda_idx = analyze_median_case(data, dim=1, keepdim=True, dtype=torch.float16, name="float16_1024x1024_dim1")

# Check specific mismatched rows
diff_mask = ref_idx.cpu() != cuda_idx.cpu()
if diff_mask.any():
    print(f"\nMismatched rows: {diff_mask.nonzero()[:10].flatten().tolist()}")
    for row_idx in diff_mask.nonzero()[:5].flatten():
        print(f"\nRow {row_idx}:")
        row_data = data[row_idx].to(torch.float16).cpu()
        k = (len(row_data) + 1) // 2
        kth_val, kth_idx = torch.kthvalue(row_data, k)
        mask = row_data == kth_val
        positions = torch.where(mask)[0]
        print(f"  Median value {kth_val.item()} at positions: {positions.tolist()}")
        print(f"  CPU index: {ref_idx[row_idx].item()}, CUDA index: {cuda_idx[row_idx].item()}")

print("\n" + "="*80)
print("Analysis complete")
print("="*80)

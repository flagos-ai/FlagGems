"""
深入分析 median 索引不匹配的根本原因
"""
import torch
import flag_gems

def to_reference_simulated(inp, upcast=False):
    """模拟 to_reference(inp, True) 当 TO_CPU=False 时"""
    ref_inp = inp
    if upcast:
        if ref_inp.is_complex():
            ref_inp = ref_inp.to(torch.complex128)
        else:
            ref_inp = ref_inp.to(torch.float64)
    return ref_inp

print("="*80)
print("分析 median 索引不匹配的根本原因")
print("="*80)

# 找到不匹配的情况
shape = (1024, 1024)
dtype = torch.bfloat16
dim = 1
keepdim = False

for attempt in range(200):
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
        print(f"\n找到不匹配 (种子 {attempt}): {diff_count}/1024")

        # 分析第一个不匹配
        diff_mask = ref_idx_cpu != res_idx_cpu
        first_mismatch = diff_mask.nonzero().flatten()[0].item()

        print(f"第一个不匹配位置 {first_mismatch}:")
        print(f"  参考索引: {ref_idx_cpu[first_mismatch].item()}")
        print(f"  FlagGems索引: {res_idx_cpu[first_mismatch].item()}")

        # 获取该行的数据
        row_f64 = ref_inp[first_mismatch]  # float64
        row_bf16 = inp[first_mismatch]      # bfloat16

        k = (1024 + 1) // 2  # 513

        # 使用 kthvalue
        kth_val_f64, kth_idx_f64 = torch.kthvalue(row_f64, k)
        kth_val_bf16_as_f64, kth_idx_bf16_as_f64 = torch.kthvalue(row_bf16.to(torch.float64), k)

        # 使用 median
        median_val_f64, median_idx_f64 = torch.median(row_f64, dim=0)
        median_val_bf16_as_f64, median_idx_bf16_as_f64 = torch.median(row_bf16.to(torch.float64), dim=0)

        print(f"\n该行分析:")
        print(f"  float64 kthvalue:  索引={kth_idx_f64.item()}")
        print(f"  float64 median:    索引={median_idx_f64.item()}")
        print(f"  bf16->f64 kthvalue: 索引={kth_idx_bf16_as_f64.item()}")
        print(f"  bf16->f64 median:   索引={median_idx_bf16_as_f64.item()}")

        # 检查中位数值
        mask_f64 = row_f64 == kth_val_f64
        positions_f64 = torch.where(mask_f64)[0]

        if len(positions_f64) >= 2:
            print(f"\n  中位数值 {kth_val_f64.item()} 出现在位置: {positions_f64.tolist()}")
            print(f"  首次出现: {positions_f64[0].item()}, 最后出现: {positions_f64[-1].item()}")

        # 检查 PyTorch 在整个张量上的行为
        full_tensor_median_idx = ref_idx_cpu[first_mismatch].item()
        print(f"\n  整个张量的 median 选择: {full_tensor_median_idx}")

        # 检查数据是否一致
        row_bf16_as_f64 = row_bf16.to(torch.float64)
        data_match = torch.allclose(row_f64, row_bf16_as_f64, rtol=0, atol=0)
        print(f"\n  ref_inp[first_mismatch] == inp[first_mismatch].to(float64): {data_match}")

        if not data_match:
            diff_positions = (row_f64 != row_bf16_as_f64).nonzero().flatten()
            print(f"  数据不一致的位置数量: {len(diff_positions)}")
            if len(diff_positions) > 0:
                print(f"  前10个不一致位置: {diff_positions[:10].tolist()}")

        break

print("\n" + "="*80)
print("分析完成")
print("="*80)

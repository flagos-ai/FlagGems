"""
分析大尺寸测试中的索引不匹配问题
"""
import torch
import flag_gems

def to_reference_simulated(inp, upcast=False):
    """模拟 to_reference(inp, True)"""
    ref_inp = inp
    if upcast:
        if ref_inp.is_complex():
            ref_inp = ref_inp.to(torch.complex128)
        else:
            ref_inp = ref_inp.to(torch.float64)
    return ref_inp

print("="*80)
print("分析大尺寸测试中的索引不匹配问题")
print("="*80)

shape = (1024, 1024)
dtype = torch.bfloat16
dim = 1
keepdim = False

# 找到不匹配的情况
for attempt in range(100):
    torch.manual_seed(attempt)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference_simulated(inp, True)

    # Reference median
    ref_vals, ref_idx = torch.median(ref_inp, dim=dim, keepdim=keepdim)

    # FlagGems median
    with flag_gems.use_gems():
        res_vals, res_idx = torch.median(inp, dim=dim, keepdim=keepdim)

    ref_idx_cpu = ref_idx.cpu()
    res_idx_cpu = res_idx.cpu()
    if not torch.equal(ref_idx_cpu, res_idx_cpu):
        diff_count = (ref_idx_cpu != res_idx_cpu).sum().item()
        print(f"\n种子 {attempt}: {diff_count}/1024 不匹配")

        # 分析第一个不匹配
        diff_mask = ref_idx_cpu != res_idx_cpu
        first_mismatch = diff_mask.nonzero().flatten()[0].item()

        print(f"第一个不匹配位置 {first_mismatch}:")
        print(f"  参考索引: {ref_idx_cpu[first_mismatch].item()}")
        print(f"  FlagGems索引: {res_idx_cpu[first_mismatch].item()}")

        # 获取该行数据
        row_f64 = ref_inp[first_mismatch]
        row_bf16 = inp[first_mismatch]
        row_bf16_as_f64 = row_bf16.to(torch.float64)

        k = (1024 + 1) // 2  # 513

        # 比较不同的方法
        kth_val, kth_idx = torch.kthvalue(row_f64, k)
        median_val_row, median_idx_row = torch.median(row_f64, dim=0)

        print(f"\n  float64 kthvalue: {kth_idx.item()}")
        print(f"  float64 median: {median_idx_row.item()}")
        print(f"  参考结果: {ref_idx_cpu[first_mismatch].item()}")

        # 检查是否有重复值
        mask = row_f64 == kth_val
        positions = torch.where(mask)[0]
        if len(positions) >= 2:
            print(f"\n  中位数值 {kth_val.item()} 出现在位置: {positions.tolist()}")

        # 检查整个张量的 median
        vals_full, idx_full = torch.median(ref_inp, dim=1)
        print(f"\n  整个张量的 median: {idx_full[first_mismatch].item()}")
        print(f"  单行的 median: {median_idx_row.item()}")

        # 关键检查：两者是否一致？
        if idx_full[first_mismatch].item() != median_idx_row.item():
            print(f"\n  >>> 关键发现：整个张量 vs 单行不一致！")
            print(f"  整个张量选择: {idx_full[first_mismatch].item()}")
            print(f"  单行选择: {median_idx_row.item()}")

            # 尝试理解原因
            print(f"\n  这可能是 PyTorch 的优化导致的差异")
            print(f"  让我们尝试另一种方法...")

        break

print("\n" + "="*80)
print("分析完成")
print("="*80)

"""
分析之前发现的不匹配情况（种子 0，行 6）
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
print("分析种子 0 的不匹配情况")
print("="*80)

torch.manual_seed(0)
inp = torch.randn((1024, 1024), dtype=torch.bfloat16, device='cuda')
ref_inp = to_reference_simulated(inp, True)

# 检查行 6
row_idx = 6
row_f64 = ref_inp[row_idx]
row_bf16 = inp[row_idx]

k = (1024 + 1) // 2  # 513

# 使用 kthvalue
kth_val_f64, kth_idx_f64 = torch.kthvalue(row_f64, k)

# 找到所有中位数值的位置
mask_f64 = row_f64 == kth_val_f64
positions_f64 = torch.where(mask_f64)[0]

print(f"行 {row_idx} 分析:")
print(f"  中位数值: {kth_val_f64.item()}")
print(f"  出现位置: {positions_f64.tolist()}")
print(f"  kthvalue 索引: {kth_idx_f64.item()}")

# 在整个张量上计算 median
ref_vals, ref_idx = torch.median(ref_inp, dim=1, keepdim=False)
full_median_idx = ref_idx[row_idx].item()

print(f"\n比较:")
print(f"  整个张量 median 索引: {full_median_idx}")
print(f"  kthvalue 索引: {kth_idx_f64.item()}")

# 在单行上计算 median
median_val_row, median_idx_row = torch.median(row_f64, dim=0)
print(f"  单行 median 索引: {median_idx_row.item()}")

# 检查 FlagGems 的结果
with flag_gems.use_gems():
    res_vals, res_idx = torch.median(inp, dim=1, keepdim=False)

gems_idx = res_idx[row_idx].item()
print(f"  FlagGems 索引: {gems_idx}")

print(f"\n结论:")
if full_median_idx == kth_idx_f64.item():
    print("  整个张量 median 与 kthvalue 一致")
else:
    print(f"  整个张量 median 选择位置 {positions_f64.tolist().index(full_median_idx) if full_median_idx in positions_f64 else 'N/A'}")
    print(f"  kthvalue 选择位置 {positions_f64.tolist().index(kth_idx_f64.item()) if kth_idx_f64.item() in positions_f64 else 'N/A'}")

# 测试：直接调用 torch.median 在单行上
print(f"\n验证: 单行上 torch.median 应该与整个张量一致")
print(f"  单行 torch.median: {median_idx_row.item()}")
print(f"  整个张量 torch.median: {full_median_idx}")
print(f"  一致: {median_idx_row.item() == full_median_idx}")

print("\n" + "="*80)
print("分析完成")
print("="*80)

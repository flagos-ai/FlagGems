#!/usr/bin/env python3
"""
修复 ref_out_buf 设备问题：
当 ref_x = to_reference(x) 时，ref_out_buf 应该在 ref_x.device 上而不是 flag_gems.device 上
"""

import re
from pathlib import Path

TEST_DIR = Path(__file__).parent / "tests" / "experimental_ops"

def fix_ref_out_buf(filepath):
    """修复单个文件中的 ref_out_buf 设备问题"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original = content
    changes = []
    
    # 模式：在有 ref_x = to_reference(x) 的函数中
    # ref_out_buf = torch.empty(..., device=flag_gems.device)
    # 应该改为 device=ref_x.device
    
    # 查找所有测试函数
    func_pattern = r'(def test_\w+\([^)]*\):.*?)(?=\ndef |\Z)'
    
    def process_func(match):
        func_code = match.group(1)
        
        # 查找 ref_xxx = to_reference(yyy) 
        ref_match = re.search(r'(ref_\w+)\s*=\s*to_reference\((\w+)\)', func_code)
        if not ref_match:
            return func_code
        
        ref_var = ref_match.group(1)  # e.g., ref_x
        
        # 替换 ref_out_buf = torch.empty(..., device=flag_gems.device)
        # 为 device=ref_var.device
        new_func = re.sub(
            r'(ref_out_buf\s*=\s*torch\.empty\([^)]*),\s*device=flag_gems\.device\)',
            rf'\1, device={ref_var}.device)',
            func_code
        )
        
        if new_func != func_code:
            return new_func
        return func_code
    
    new_content = re.sub(func_pattern, process_func, content, flags=re.DOTALL)
    
    if new_content != original:
        # 检查具体改了什么
        old_lines = original.split('\n')
        new_lines = new_content.split('\n')
        for i, (old, new) in enumerate(zip(old_lines, new_lines)):
            if old != new:
                changes.append(f"行 {i+1}")
        
        with open(filepath, 'w') as f:
            f.write(new_content)
        
        return True, changes
    
    return False, []

def main():
    # 已知有问题的文件列表
    problem_files = [
        "_adaptive_avg_pool2d_test.py",
        "_adaptive_avg_pool3d_test.py", 
        "_upsample_nearest_exact1d_test.py",
        "as_strided_copy_test.py",
        "diag_test.py",
        "erfinv_test.py",
        "fix_test.py",
        "hypot_test.py",
        "logaddexp_test.py",
        "logit_test.py",
        "masked_select_test.py",
        "maximum_test.py",
        "mse_loss_test.py",
        "mv_test.py",
        "new_ones_test.py",
        "permute_copy_test.py",
        "reflection_pad1d_test.py",
        "reflection_pad2d_test.py",
        "reflection_pad3d_test.py",
        "replication_pad1d_test.py",
        "replication_pad2d_test.py",
        "replication_pad3d_test.py",
        "scalar_tensor_test.py",
        "select_backward_test.py",
        "slice_backward_test.py",
        "special_i1_test.py",
        "stack_test.py",
        "t_copy_test.py",
        "transpose_copy_test.py",
        "unsqueeze_copy_test.py",
    ]
    
    print(f"修复 ref_out_buf 设备问题")
    print("=" * 60)
    
    fixed_count = 0
    for fname in problem_files:
        filepath = TEST_DIR / fname
        if filepath.exists():
            fixed, changes = fix_ref_out_buf(filepath)
            if fixed:
                print(f"✅ {fname}: 修复了 {len(changes)} 处")
                fixed_count += 1
            else:
                print(f"⏭️ {fname}: 无需修改")
        else:
            print(f"❌ {fname}: 文件不存在")
    
    print("=" * 60)
    print(f"总计修复: {fixed_count} 个文件")

if __name__ == "__main__":
    main()

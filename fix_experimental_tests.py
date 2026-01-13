#!/usr/bin/env python3
"""
批量修复 experimental_ops 测试文件，添加 to_reference 支持
"""

import os
import re
from pathlib import Path

# to_reference 函数定义
TO_REFERENCE_FUNC = '''
def to_reference(inp, upcast=False):
    if inp is None:
        return None
    ref_inp = inp
    if TO_CPU:
        ref_inp = ref_inp.to("cpu")
    if upcast:
        if ref_inp.is_complex():
            ref_inp = ref_inp.to(torch.complex128)
        else:
            ref_inp = ref_inp.to(torch.float64)
    return ref_inp
'''

def fix_test_file(filepath):
    """修复单个测试文件"""
    print(f"Processing: {filepath.name}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 1. 修改 import 部分，添加 TO_CPU
    # 查找 from tests.accuracy_utils import gems_assert_close
    import_pattern = r'from tests\.accuracy_utils import gems_assert_close'
    if re.search(import_pattern, content):
        # 已经有导入，添加 TO_CPU
        content = re.sub(
            r'from tests\.accuracy_utils import gems_assert_close(\s+#.*)?',
            r'from tests.accuracy_utils import gems_assert_close, TO_CPU\1',
            content
        )
    
    # 2. 在 ImportError 的 except 块中添加 TO_CPU fallback
    except_pattern = r'(except ImportError:\s*#[^\n]*\n)'
    if re.search(except_pattern, content):
        content = re.sub(
            except_pattern,
            r'\1    TO_CPU = False  # fallback\n',
            content
        )
    
    # 3. 在导入之后、第一个 @pytest 之前添加 to_reference 函数
    # 找到第一个 @pytest.mark 的位置
    pytest_match = re.search(r'\n@pytest\.mark\.', content)
    if pytest_match:
        insert_pos = pytest_match.start()
        content = content[:insert_pos] + '\n' + TO_REFERENCE_FUNC + '\n' + content[insert_pos:]
    
    # 4. 替换所有 ref_xxx = xxx.clone() 为 ref_xxx = to_reference(xxx)
    # 需要处理多种情况：
    
    # 4.1 标准情况: ref_x = x.clone()
    content = re.sub(
        r'(\s+)(ref_\w+)\s*=\s*(\w+)\.clone\(\)(?!\s*\.)',  # 后面没有 .requires_grad
        r'\1\2 = to_reference(\3)',
        content
    )
    
    # 4.2 有 requires_grad 的情况: ref_input = input_tensor.clone().requires_grad_(True)
    # 需要保留 requires_grad 调用
    content = re.sub(
        r'(\s+)(ref_\w+)\s*=\s*(\w+)\.clone\(\)(\.requires_grad_\([^)]+\))',
        r'\1\2 = to_reference(\3)\4',
        content
    )
    
    # 4.3 列表推导: ref_tensors = [t.clone() for t in tensors]
    content = re.sub(
        r'(\s+)(ref_\w+)\s*=\s*\[(\w+)\.clone\(\)\s+for\s+\3\s+in\s+(\w+)\]',
        r'\1\2 = [to_reference(\3) for \3 in \4]',
        content
    )
    
    # 4.4 在函数调用中: torch.ops.aten.xxx(x.clone(), ...)
    # 这个比较复杂，先不自动处理，需要手动检查
    
    # 5. 如果内容有变化，写回文件
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ Fixed {filepath.name}")
        return True
    else:
        print(f"  - No changes needed for {filepath.name}")
        return False

def main():
    """主函数：批量处理所有测试文件"""
    test_dir = Path(__file__).parent / 'tests' / 'experimental_ops'
    
    if not test_dir.exists():
        print(f"Error: Directory not found: {test_dir}")
        return
    
    # 获取所有测试文件（排除 __init__.py）
    test_files = sorted([f for f in test_dir.glob('*.py') if f.name != '__init__.py'])
    
    print(f"Found {len(test_files)} test files to process\n")
    
    fixed_count = 0
    for test_file in test_files:
        if fix_test_file(test_file):
            fixed_count += 1
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total files: {len(test_files)}")
    print(f"  Fixed files: {fixed_count}")
    print(f"  Unchanged: {len(test_files) - fixed_count}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()

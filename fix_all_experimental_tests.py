#!/usr/bin/env python3
"""
修复所有 experimental_ops 测试文件的问题：
1. 缺少 to_reference 函数和 TO_CPU 导入
2. ref_out_buf 设备问题（应该使用 to_reference 获取正确设备）
"""

import os
import re
from pathlib import Path

TEST_DIR = Path(__file__).parent / "tests" / "experimental_ops"

# 标准的 to_reference 函数模板
TO_REFERENCE_TEMPLATE = '''
def to_reference(inp, upcast=False):
    if inp is None:
        return None
    if TO_CPU:
        ref_inp = inp.to("cpu")
    else:
        ref_inp = inp.clone()
    if upcast:
        if ref_inp.is_complex():
            ref_inp = ref_inp.to(torch.complex128)
        else:
            ref_inp = ref_inp.to(torch.float64)
    return ref_inp
'''

# 需要添加 TO_CPU 导入的模式
TO_CPU_IMPORT_PATTERN = r'from tests\.accuracy_utils import gems_assert_close(?!\s*,\s*TO_CPU)'
TO_CPU_IMPORT_REPLACEMENT = 'from tests.accuracy_utils import gems_assert_close, TO_CPU'

def fix_missing_to_reference(content, filename):
    """修复缺少 to_reference 函数的文件"""
    changes = []
    
    # 检查是否已有 to_reference
    if 'def to_reference(' in content:
        return content, changes
    
    # 检查是否导入了 TO_CPU
    if 'TO_CPU' not in content:
        # 添加 TO_CPU 导入
        content = re.sub(
            TO_CPU_IMPORT_PATTERN,
            TO_CPU_IMPORT_REPLACEMENT,
            content
        )
        changes.append("添加 TO_CPU 导入")
    
    # 如果仍然没有 TO_CPU（可能原来就没有导入 gems_assert_close），需要完整添加
    if 'TO_CPU' not in content:
        # 找到 try/except ImportError 块后添加
        import_block_pattern = r'(try:\s*\n\s*from tests\.accuracy_utils import gems_assert_close.*?(?:except ImportError:.*?torch\.testing\.assert_close\(res, ref, \*\*kwargs\)))'
        match = re.search(import_block_pattern, content, re.DOTALL)
        if match:
            old_block = match.group(1)
            new_block = old_block.replace(
                'from tests.accuracy_utils import gems_assert_close',
                'from tests.accuracy_utils import gems_assert_close, TO_CPU'
            )
            if 'TO_CPU = False' not in new_block:
                new_block = new_block.replace(
                    'except ImportError:',
                    'except ImportError:\n    # Fallback values when running outside pytest\n    TO_CPU = False  # fallback'
                )
            content = content.replace(old_block, new_block)
            changes.append("添加 TO_CPU 导入和 fallback")
    
    # 在第一个 @pytest.mark 之前添加 to_reference 函数
    pytest_mark_match = re.search(r'\n(@pytest\.mark\.)', content)
    if pytest_mark_match:
        insert_pos = pytest_mark_match.start()
        content = content[:insert_pos] + '\n' + TO_REFERENCE_TEMPLATE + '\n' + content[insert_pos:]
        changes.append("添加 to_reference 函数")
    
    return content, changes

def fix_ref_out_buf_device(content, filename):
    """修复 ref_out_buf 设备问题"""
    changes = []
    
    # 查找 ref_out_buf = torch.empty(..., device=flag_gems.device) 模式
    # 这些应该改为使用 to_reference 获取的设备
    
    # 模式1: ref_out_buf = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    pattern1 = r'ref_out_buf = torch\.empty\(([^)]+),\s*device=flag_gems\.device\)'
    
    def replace_ref_out_buf(match):
        args = match.group(1)
        # 提取参数，需要确定正确的设备
        # 使用 ref_x.device 或类似的引用设备
        return f'ref_out_buf = torch.empty({args}, device=ref_x.device if "ref_x" in dir() else ("cpu" if TO_CPU else flag_gems.device))'
    
    # 这个替换比较复杂，需要更精确的处理
    # 让我们用更简单的方法：在函数开头获取正确的设备
    
    # 查找所有有 ref_out_buf 问题的测试函数
    # 更简单的方法：将 ref_out_buf 的 device 改为条件表达式
    
    # 模式：ref_out_buf = torch.empty(..., device=flag_gems.device)
    # 替换为：ref_device = "cpu" if TO_CPU else flag_gems.device; ref_out_buf = torch.empty(..., device=ref_device)
    
    # 实际上最好的方法是：在已有 ref_x = to_reference(x) 的情况下，使用 ref_x.device
    
    lines = content.split('\n')
    new_lines = []
    in_test_function = False
    current_ref_var = None
    
    for i, line in enumerate(lines):
        # 检测是否进入测试函数
        if line.strip().startswith('def test_'):
            in_test_function = True
            current_ref_var = None
        
        # 检测 ref_xxx = to_reference(xxx) 
        ref_match = re.match(r'\s*(ref_\w+)\s*=\s*to_reference\(', line)
        if ref_match:
            current_ref_var = ref_match.group(1)
        
        # 检测并修复 ref_out_buf 创建
        if in_test_function and 'ref_out_buf = torch.empty' in line and 'device=flag_gems.device' in line:
            # 使用已知的 ref 变量设备，或者 TO_CPU 条件
            if current_ref_var:
                new_line = line.replace('device=flag_gems.device', f'device={current_ref_var}.device')
            else:
                new_line = line.replace('device=flag_gems.device', 'device="cpu" if TO_CPU else flag_gems.device')
            new_lines.append(new_line)
            changes.append(f"修复 ref_out_buf 设备 (行 {i+1})")
        else:
            new_lines.append(line)
    
    return '\n'.join(new_lines), changes

def fix_inplace_ops_missing_to_reference(content, filename):
    """修复 inplace 操作测试中缺少正确引用处理的问题"""
    changes = []
    
    # 对于 inplace 操作如 abs_, transpose_ 等，它们的模式是：
    # ref_input = torch.randn(...); act_input = ref_input.clone()
    # 这种情况下 ref_input 和 act_input 都在 GPU 上
    # 如果要支持 --ref cpu，需要用 to_reference
    
    # 查找模式: ref_input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    #           act_input = ref_input.clone()
    
    # 对于 inplace 操作，正确的模式应该是:
    # base = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # ref_input = to_reference(base)  
    # act_input = base.clone()
    
    # 这需要更复杂的重构，暂时跳过
    
    return content, changes

def process_file(filepath):
    """处理单个文件"""
    print(f"处理: {filepath.name}")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    all_changes = []
    
    # 1. 修复缺少 to_reference 的问题
    content, changes = fix_missing_to_reference(content, filepath.name)
    all_changes.extend(changes)
    
    # 2. 修复 ref_out_buf 设备问题
    content, changes = fix_ref_out_buf_device(content, filepath.name)
    all_changes.extend(changes)
    
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  ✅ 已修复: {', '.join(all_changes)}")
        return True, all_changes
    else:
        print(f"  ⏭️ 无需修改")
        return False, []

def main():
    test_files = sorted(TEST_DIR.glob("*_test.py"))
    print(f"找到 {len(test_files)} 个测试文件")
    print("=" * 60)
    
    modified_count = 0
    all_modifications = {}
    
    for filepath in test_files:
        modified, changes = process_file(filepath)
        if modified:
            modified_count += 1
            all_modifications[filepath.name] = changes
    
    print("=" * 60)
    print(f"总计修改: {modified_count} 个文件")
    
    if all_modifications:
        print("\n修改详情:")
        for fname, changes in all_modifications.items():
            print(f"  {fname}: {', '.join(changes)}")

if __name__ == "__main__":
    main()

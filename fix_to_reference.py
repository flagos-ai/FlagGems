#!/usr/bin/env python3
"""修复 to_reference 函数，确保在 TO_CPU=False 时也返回 clone"""

import os
import re
from pathlib import Path

# 错误的 to_reference 函数（没有 clone）
OLD_TO_REF = '''def to_reference(inp, upcast=False):
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
    return ref_inp'''

# 正确的 to_reference 函数（有 clone）
NEW_TO_REF = '''def to_reference(inp, upcast=False):
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
    return ref_inp'''

def fix_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if OLD_TO_REF in content:
        content = content.replace(OLD_TO_REF, NEW_TO_REF)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    test_dir = Path('tests/experimental_ops')
    fixed = 0
    for f in test_dir.glob('*.py'):
        if fix_file(f):
            print(f"Fixed: {f.name}")
            fixed += 1
    print(f"\nTotal fixed: {fixed}")

if __name__ == '__main__':
    main()

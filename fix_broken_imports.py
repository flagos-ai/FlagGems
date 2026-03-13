#!/usr/bin/env python3
"""
Script to fix broken multi-line imports by removing orphaned lines.
"""

import re
from pathlib import Path


def fix_broken_imports(file_path):
    """Fix broken multi-line imports in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        fixed_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]

            # Check if this is an orphaned import line (starts with spaces and contains 'as gems_')
            if re.match(r'^\s+\w+.*as gems_', line) and i > 0:
                # Check if previous line is not a 'from' or 'import' statement
                prev_line = lines[i-1].strip()
                if not prev_line.startswith('from') and not prev_line.startswith('import'):
                    # This is an orphaned line, skip it
                    i += 1
                    continue

            # Check if this is a closing parenthesis on its own line after we removed imports
            if line.strip() == ')' and i > 0:
                prev_line = lines[i-1].strip()
                # If previous line doesn't look like an import, remove this closing paren
                if not ('import' in prev_line or 'as gems_' in prev_line):
                    i += 1
                    continue

            fixed_lines.append(line)
            i += 1

        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(fixed_lines)

        return True
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False


def main():
    base_dir = Path('/share/project/zpy/FlagGems/experimental_tests')

    # Get all test files
    unit_files = list((base_dir / 'unit').glob('*_test.py'))
    perf_files = list((base_dir / 'performance').glob('*_test.py'))

    all_files = unit_files + perf_files

    print(f"Fixing {len(all_files)} files...")

    success_count = 0
    for file_path in all_files:
        if fix_broken_imports(file_path):
            success_count += 1

    print(f"Done! Processed {success_count}/{len(all_files)} files")


if __name__ == '__main__':
    main()

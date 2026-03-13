#!/usr/bin/env python3
"""
Script to remove unused imports based on flake8 F401 errors.
"""

import re
import subprocess
from pathlib import Path


def get_unused_imports(file_path):
    """Run flake8 on a file and extract F401 errors."""
    try:
        result = subprocess.run(
            ['flake8', '--select=F401', str(file_path)],
            capture_output=True,
            text=True
        )

        unused = []
        for line in result.stdout.strip().split('\n'):
            if 'F401' in line and line.strip():
                # Extract the import statement from the error message
                match = re.search(r"F401 '([^']+)' imported but unused", line)
                if match:
                    import_name = match.group(1)
                    # Extract line number
                    line_match = re.search(r':(\d+):', line)
                    if line_match:
                        line_num = int(line_match.group(1))
                        unused.append((line_num, import_name))

        return unused
    except Exception as e:
        print(f"Error checking {file_path}: {e}")
        return []


def remove_import_lines(file_path, line_numbers):
    """Remove specific lines from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Sort line numbers in reverse order to remove from bottom to top
        line_numbers_sorted = sorted(set(line_numbers), reverse=True)

        for line_num in line_numbers_sorted:
            if 1 <= line_num <= len(lines):
                # Check if this line is an import
                line = lines[line_num - 1]
                if 'import' in line:
                    del lines[line_num - 1]

        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        return True
    except Exception as e:
        print(f"Error modifying {file_path}: {e}")
        return False


def main():
    base_dir = Path('/share/project/zpy/FlagGems/experimental_tests')

    # Get all test files
    unit_files = list((base_dir / 'unit').glob('*_test.py'))
    perf_files = list((base_dir / 'performance').glob('*_test.py'))

    all_files = unit_files + perf_files

    print(f"Processing {len(all_files)} files...")

    success_count = 0
    for file_path in all_files:
        unused = get_unused_imports(file_path)
        if unused:
            line_numbers = [line_num for line_num, _ in unused]
            if remove_import_lines(file_path, line_numbers):
                success_count += 1
                if success_count % 20 == 0:
                    print(f"  Processed {success_count} files...")

    print(f"\nDone! Successfully processed {success_count} files with unused imports")


if __name__ == '__main__':
    main()

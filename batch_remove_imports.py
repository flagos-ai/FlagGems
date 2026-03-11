#!/usr/bin/env python3
"""
Script to batch remove unused imports based on F401 errors.
Only removes single-line imports to avoid breaking multi-line imports.
"""

import re
from pathlib import Path
from collections import defaultdict


def parse_f401_errors(error_file):
    """Parse F401 error file and group by file path."""
    file_errors = defaultdict(list)

    with open(error_file, 'r') as f:
        for line in f:
            # Format: path/to/file.py:line:col: F401 'module' imported but unused
            match = re.match(r'([^:]+):(\d+):(\d+): F401 \'([^\']+)\' imported but unused', line)
            if match:
                file_path = match.group(1)
                line_num = int(match.group(2))
                import_name = match.group(4)

                file_errors[file_path].append({
                    'line': line_num,
                    'import': import_name
                })

    return file_errors


def is_single_line_import(lines, line_num):
    """Check if the import at line_num is a single-line import."""
    if line_num < 1 or line_num > len(lines):
        return False

    line = lines[line_num - 1].strip()

    # Single-line import patterns:
    # - import xxx
    # - from xxx import yyy
    # - from xxx import yyy as zzz

    # NOT single-line if it contains opening parenthesis without closing
    if '(' in line and ')' not in line:
        return False

    # NOT single-line if it's just a closing parenthesis
    if line == ')':
        return False

    # Should start with 'import' or 'from'
    if not (line.startswith('import ') or line.startswith('from ')):
        return False

    return True


def remove_imports_from_file(file_path, errors):
    """Remove unused imports from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Filter to only single-line imports
        single_line_errors = []
        for error in errors:
            if is_single_line_import(lines, error['line']):
                single_line_errors.append(error)

        if not single_line_errors:
            return 0, 0

        # Sort by line number in reverse order to delete from bottom to top
        single_line_errors.sort(key=lambda x: x['line'], reverse=True)

        # Remove the lines
        removed_count = 0
        for error in single_line_errors:
            line_num = error['line']
            if 1 <= line_num <= len(lines):
                del lines[line_num - 1]
                removed_count += 1

        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        skipped_count = len(errors) - removed_count
        return removed_count, skipped_count

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0, 0


def main():
    error_file = '/share/project/zpy/FlagGems/f401_errors_list.txt'

    print("Parsing F401 errors...")
    file_errors = parse_f401_errors(error_file)

    print(f"Found {len(file_errors)} files with unused imports\n")

    total_removed = 0
    total_skipped = 0
    processed_files = 0

    for file_path, errors in sorted(file_errors.items()):
        removed, skipped = remove_imports_from_file(file_path, errors)

        if removed > 0 or skipped > 0:
            processed_files += 1
            total_removed += removed
            total_skipped += skipped

            if processed_files % 20 == 0:
                print(f"  Processed {processed_files}/{len(file_errors)} files...")

    print(f"\nDone!")
    print(f"  Total files processed: {processed_files}")
    print(f"  Total imports removed: {total_removed}")
    print(f"  Total imports skipped (multi-line): {total_skipped}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Script to remove multi-line imports based on F401 errors.
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


def find_multiline_import_range(lines, start_line):
    """Find the range of a multi-line import statement."""
    # start_line is 1-indexed
    if start_line < 1 or start_line > len(lines):
        return None

    # Check if this line starts with 'from'
    line = lines[start_line - 1]
    if not line.strip().startswith('from '):
        return None

    # Find the start of the import block
    import_start = start_line

    # Find the end (look for closing parenthesis)
    import_end = start_line
    for i in range(start_line - 1, len(lines)):
        if ')' in lines[i]:
            import_end = i + 1
            break

    return (import_start, import_end)


def remove_multiline_import(lines, start_line, end_line):
    """Remove a multi-line import block."""
    # Convert to 0-indexed
    start_idx = start_line - 1
    end_idx = end_line

    # Delete the lines
    del lines[start_idx:end_idx]
    return True


def remove_imports_from_file(file_path, errors):
    """Remove unused multi-line imports from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Group errors by line number to handle multi-line imports
        lines_to_remove = set()

        for error in errors:
            line_num = error['line']
            import_range = find_multiline_import_range(lines, line_num)

            if import_range:
                start, end = import_range
                # Mark all lines in this range for removal
                for i in range(start, end + 1):
                    lines_to_remove.add(i)

        if not lines_to_remove:
            return 0

        # Remove lines in reverse order
        for line_num in sorted(lines_to_remove, reverse=True):
            if 1 <= line_num <= len(lines):
                del lines[line_num - 1]

        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        return len(lines_to_remove)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0


def main():
    error_file = '/tmp/f401_remaining.txt'

    print("Parsing remaining F401 errors...")
    file_errors = parse_f401_errors(error_file)

    print(f"Found {len(file_errors)} files with unused imports\n")

    total_removed = 0
    processed_files = 0

    for file_path, errors in sorted(file_errors.items()):
        removed = remove_imports_from_file(file_path, errors)

        if removed > 0:
            processed_files += 1
            total_removed += removed
            print(f"  {file_path}: removed {removed} lines")

    print(f"\nDone!")
    print(f"  Total files processed: {processed_files}")
    print(f"  Total lines removed: {total_removed}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Script to remove unused imports based on pre-commit flake8 output.
"""

import re
from pathlib import Path


def parse_flake8_output(output_file):
    """Parse flake8 output and extract file:line mappings."""
    file_lines = {}

    with open(output_file, 'r') as f:
        for line in f:
            if 'F401' in line:
                # Format: path/to/file.py:line:col: F401 'module' imported but unused
                match = re.match(r'([^:]+):(\d+):', line)
                if match:
                    file_path = match.group(1)
                    line_num = int(match.group(2))

                    if file_path not in file_lines:
                        file_lines[file_path] = []
                    file_lines[file_path].append(line_num)

    return file_lines


def remove_lines_from_file(file_path, line_numbers):
    """Remove specific lines from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Sort in reverse to remove from bottom to top
        for line_num in sorted(set(line_numbers), reverse=True):
            if 1 <= line_num <= len(lines):
                del lines[line_num - 1]

        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    output_file = '/tmp/precommit_output.txt'

    print("Parsing flake8 output...")
    file_lines = parse_flake8_output(output_file)

    print(f"Found {len(file_lines)} files with unused imports")

    success_count = 0
    for file_path, line_numbers in file_lines.items():
        if remove_lines_from_file(file_path, line_numbers):
            success_count += 1
            if success_count % 20 == 0:
                print(f"  Processed {success_count}/{len(file_lines)} files...")

    print(f"\nDone! Successfully processed {success_count}/{len(file_lines)} files")


if __name__ == '__main__':
    main()

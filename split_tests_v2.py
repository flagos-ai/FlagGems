#!/usr/bin/env python3
"""
Script to split test files into unit tests and performance tests.
This version preserves comments and formatting by using string manipulation.
"""

import re
from pathlib import Path


def extract_functions(content):
    """Extract all function definitions with their complete bodies."""
    functions = []
    lines = content.split('\n')

    i = 0
    while i < len(lines):
        line = lines[i]
        # Check if this line starts a function definition
        if re.match(r'^def test_\w+\(', line):
            func_start = i
            func_name = re.match(r'^def (test_\w+)\(', line).group(1)

            # Find the end of this function (next def at same indentation or EOF)
            i += 1
            while i < len(lines):
                next_line = lines[i]
                # Check if we hit another function definition or class at top level
                if re.match(r'^(def |class |\S)', next_line) and next_line.strip():
                    break
                i += 1

            func_end = i
            func_body = '\n'.join(lines[func_start:func_end])

            # Determine if this is a performance test
            is_perf = 'benchmark' in func_name or 'perf' in func_name

            functions.append({
                'name': func_name,
                'body': func_body,
                'is_perf': is_perf,
                'start': func_start,
                'end': func_end
            })
        else:
            i += 1

    return functions


def split_file(file_path, output_path, keep_perf=True):
    """
    Split a test file, keeping either perf or unit tests.

    Args:
        file_path: Source file path
        output_path: Output file path
        keep_perf: If True, keep performance tests; if False, keep unit tests
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    functions = extract_functions(content)

    if not functions:
        # No test functions found, just copy the file as-is
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return

    # Find where the first function starts
    first_func_start = functions[0]['start']

    # Keep everything before the first function (imports, helpers, etc.)
    header = '\n'.join(lines[:first_func_start])

    # Filter functions based on keep_perf flag
    kept_functions = [f for f in functions if f['is_perf'] == keep_perf]

    # Build the output content
    output_lines = [header]

    for func in kept_functions:
        output_lines.append(func['body'])

    # Join and write
    output_content = '\n'.join(output_lines)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output_content)


def main():
    base_dir = Path('/share/project/zpy/FlagGems/experimental_tests')
    unit_dir = base_dir / 'unit'
    perf_dir = base_dir / 'performance'

    # Get all test files
    test_files = list(base_dir.glob('*_test.py'))

    print(f"Found {len(test_files)} test files")
    print(f"\nProcessing files...")

    success_count = 0
    for test_file in test_files:
        try:
            # Process unit version (keep_perf=False means keep unit tests)
            unit_output = unit_dir / test_file.name
            split_file(test_file, unit_output, keep_perf=False)

            # Process performance version (keep_perf=True means keep perf tests)
            perf_output = perf_dir / test_file.name
            split_file(test_file, perf_output, keep_perf=True)

            success_count += 1
            if success_count % 20 == 0:
                print(f"  Processed {success_count}/{len(test_files)} files...")
        except Exception as e:
            print(f"Error processing {test_file.name}: {e}")

    print(f"\nDone! Successfully processed {success_count}/{len(test_files)} files")


if __name__ == '__main__':
    main()

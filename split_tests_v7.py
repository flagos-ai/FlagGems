#!/usr/bin/env python3
"""
Script to split test files into unit and performance tests.
Simple approach: find the first non-decorator, non-empty line before perf function.
"""

import re
from pathlib import Path


def find_first_pytest_line(lines):
    """Find the line number of the first @pytest decorator (end of import area)."""
    for i, line in enumerate(lines):
        if line.strip().startswith('@pytest'):
            return i
    return None


def find_first_perf_function(lines):
    """Find the first function containing 'benchmark' or 'perf' in its name."""
    for i, line in enumerate(lines):
        if re.match(r'^def test_\w*(benchmark|perf)\w*\(', line):
            return i
    return None


def find_perf_block_start(lines, func_line):
    """
    Find the start of the performance test block.
    Go backwards from func_line to find the first line that is:
    - Not empty
    - Not a decorator (@)
    - Not part of a multi-line decorator (contains closing paren or bracket)

    The line after that is the start of the perf block.
    """
    if func_line <= 0:
        return 0

    i = func_line - 1

    # Skip backwards through decorators and empty lines
    while i >= 0:
        line = lines[i].strip()

        # If empty line, continue
        if line == '':
            i -= 1
            continue

        # If it's a decorator line, continue
        if line.startswith('@'):
            i -= 1
            continue

        # If it's part of a multi-line structure (ends with comma, or is just brackets/parens)
        if line.endswith(',') or line in ['[', ']', '(', ')']:
            i -= 1
            continue

        # Found a real content line (end of previous test function)
        # The perf block starts at i+1
        return i + 1

    # Reached the beginning of file
    return 0


def split_performance_file(source_lines):
    """
    For performance files: keep import area + performance tests (delete unit tests in middle).
    """
    # Find import area end (first @pytest)
    first_pytest = find_first_pytest_line(source_lines)
    if first_pytest is None:
        return source_lines

    # Find first performance function
    first_perf_func = find_first_perf_function(source_lines)
    if first_perf_func is None:
        return source_lines[:first_pytest]

    # Find the start of the performance block
    perf_start = find_perf_block_start(source_lines, first_perf_func)

    # Build result: import area + performance tests
    result = source_lines[:first_pytest] + source_lines[perf_start:]
    return result


def split_unit_file(source_lines):
    """
    For unit files: keep import area + unit tests (delete performance tests at end).
    """
    # Find import area end (first @pytest)
    first_pytest = find_first_pytest_line(source_lines)
    if first_pytest is None:
        return source_lines

    # Find first performance function
    first_perf_func = find_first_perf_function(source_lines)
    if first_perf_func is None:
        return source_lines

    # Find the start of the performance block
    perf_start = find_perf_block_start(source_lines, first_perf_func)

    # Build result: import area + unit tests (up to performance start)
    result = source_lines[:perf_start]
    return result


def process_file(source_file, output_file, is_performance):
    """Process a single file and write the split version."""
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if is_performance:
            result_lines = split_performance_file(lines)
        else:
            result_lines = split_unit_file(lines)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(result_lines)

        return True
    except Exception as e:
        print(f"Error processing {source_file}: {e}")
        return False


def main():
    base_dir = Path('/share/project/zpy/FlagGems/experimental_tests')

    # Get all test files from current directory
    test_files = list(base_dir.glob('*_test.py'))

    print(f"Found {len(test_files)} test files")

    # Create output directories
    unit_dir = base_dir / 'unit'
    perf_dir = base_dir / 'performance'
    unit_dir.mkdir(exist_ok=True)
    perf_dir.mkdir(exist_ok=True)

    print(f"\nProcessing files...")

    success_count = 0
    for test_file in test_files:
        # Process performance version
        perf_output = perf_dir / test_file.name
        if process_file(test_file, perf_output, is_performance=True):
            success_count += 1

        # Process unit version
        unit_output = unit_dir / test_file.name
        process_file(test_file, unit_output, is_performance=False)

        if success_count % 20 == 0:
            print(f"  Processed {success_count}/{len(test_files)} files...")

    print(f"\nDone! Successfully processed {success_count}/{len(test_files)} files")


if __name__ == '__main__':
    main()

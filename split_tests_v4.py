#!/usr/bin/env python3
"""
Script to split test files into unit and performance tests.
This version correctly handles decorators by finding continuous blocks.
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


def find_nearest_pytest_before(lines, func_line):
    """Find the nearest @pytest decorator before the given function line."""
    for i in range(func_line - 1, -1, -1):
        if lines[i].strip().startswith('@pytest'):
            return i
    return None


def split_performance_file(source_lines):
    """
    For performance files: keep import area + performance tests (delete unit tests in middle).
    """
    # Find import area end (first @pytest)
    first_pytest = find_first_pytest_line(source_lines)
    if first_pytest is None:
        # No tests found, return as-is
        return source_lines

    # Find first performance function
    first_perf_func = find_first_perf_function(source_lines)
    if first_perf_func is None:
        # No performance tests found, return empty (or just imports)
        return source_lines[:first_pytest]

    # Find the start of performance tests (nearest @pytest before first perf function)
    perf_start = find_nearest_pytest_before(source_lines, first_perf_func)
    if perf_start is None:
        perf_start = first_perf_func

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
        # No tests found, return as-is
        return source_lines

    # Find first performance function
    first_perf_func = find_first_perf_function(source_lines)
    if first_perf_func is None:
        # No performance tests found, keep everything
        return source_lines

    # Find the start of performance tests (nearest @pytest before first perf function)
    perf_start = find_nearest_pytest_before(source_lines, first_perf_func)
    if perf_start is None:
        perf_start = first_perf_func

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
        if process_file(test_file, unit_output, is_performance=False):
            pass  # Already counted

        if success_count % 20 == 0:
            print(f"  Processed {success_count}/{len(test_files)} files...")

    print(f"\nDone! Successfully processed {success_count}/{len(test_files)} files")


if __name__ == '__main__':
    main()

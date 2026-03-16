#!/usr/bin/env python3
"""
Script to split test files into unit tests and performance tests.
- unit/: keeps only correctness tests (removes benchmark/perf functions)
- performance/: keeps only performance tests (removes non-benchmark/perf functions)
"""

import ast
import os
import sys
from pathlib import Path


class TestFunctionRemover(ast.NodeTransformer):
    """AST transformer to remove specific test functions."""

    def __init__(self, remove_perf=False, remove_unit=False):
        self.remove_perf = remove_perf  # Remove performance tests
        self.remove_unit = remove_unit  # Remove unit tests

    def visit_FunctionDef(self, node):
        # Check if this is a test function
        if node.name.startswith('test_'):
            is_perf = 'benchmark' in node.name or 'perf' in node.name

            # Remove performance tests if remove_perf is True
            if self.remove_perf and is_perf:
                return None

            # Remove unit tests if remove_unit is True
            if self.remove_unit and not is_perf:
                return None

        return node


def process_file(file_path, output_path, remove_perf=False, remove_unit=False):
    """Process a single test file and write the filtered version."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()

        # Parse the source code
        tree = ast.parse(source_code)

        # Transform the AST
        transformer = TestFunctionRemover(remove_perf=remove_perf, remove_unit=remove_unit)
        new_tree = transformer.visit(tree)

        # Convert back to source code
        new_code = ast.unparse(new_tree)

        # Write to output file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(new_code)

        return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    base_dir = Path('/share/project/zpy/FlagGems/experimental_tests')
    unit_dir = base_dir / 'unit'
    perf_dir = base_dir / 'performance'

    # Get all test files in unit directory
    unit_files = list(unit_dir.glob('*_test.py'))
    perf_files = list(perf_dir.glob('*_test.py'))

    print(f"Processing {len(unit_files)} files in unit directory...")
    success_count = 0
    for file_path in unit_files:
        if process_file(file_path, file_path, remove_perf=True):
            success_count += 1
    print(f"Unit directory: {success_count}/{len(unit_files)} files processed successfully")

    print(f"\nProcessing {len(perf_files)} files in performance directory...")
    success_count = 0
    for file_path in perf_files:
        if process_file(file_path, file_path, remove_unit=True):
            success_count += 1
    print(f"Performance directory: {success_count}/{len(perf_files)} files processed successfully")

    print("\nDone!")


if __name__ == '__main__':
    main()

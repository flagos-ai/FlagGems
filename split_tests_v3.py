#!/usr/bin/env python3
"""
Script to split test files into unit tests and performance tests.
This version correctly handles multi-line function definitions.
"""

import re
from pathlib import Path


def find_function_end(lines, start_idx):
    """Find the end of a function definition starting at start_idx."""
    i = start_idx + 1

    # First, find the end of the function signature (the line with ':')
    while i < len(lines) and ':' not in lines[i]:
        i += 1

    if i >= len(lines):
        return len(lines)

    # Now find the end of the function body
    i += 1
    while i < len(lines):
        line = lines[i]
        # Check if we hit another top-level definition or non-indented content
        if line and not line[0].isspace() and line.strip():
            break
        i += 1

    return i


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

            # Extract function name from the first line
            match = re.match(r'^def (test_\w+)\(', line)
            func_name = match.group(1)

            # Find the end of this function
            func_end = find_function_end(lines, func_start)
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

            i = func_end
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

    # Get all test files from git
    import subprocess
    result = subprocess.run(
        ['git', 'show', 'HEAD:experimental_tests/'],
        cwd='/share/project/zpy/FlagGems',
        capture_output=True,
        text=True
    )

    # Extract test file names
    test_files = []
    for line in result.stdout.split('\n'):
        if '_test.py' in line:
            filename = line.split()[-1]
            test_files.append(filename)

    print(f"Found {len(test_files)} test files from git")
    print(f"\nProcessing files...")

    unit_dir = base_dir / 'unit'
    perf_dir = base_dir / 'performance'

    success_count = 0
    for filename in test_files:
        try:
            # Get file content from git
            result = subprocess.run(
                ['git', 'show', f'HEAD:experimental_tests/{filename}'],
                cwd='/share/project/zpy/FlagGems',
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                continue

            content = result.stdout

            # Write to temp file
            temp_file = base_dir / f'.temp_{filename}'
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(content)

            # Process unit version
            unit_output = unit_dir / filename
            split_file(temp_file, unit_output, keep_perf=False)

            # Process performance version
            perf_output = perf_dir / filename
            split_file(temp_file, perf_output, keep_perf=True)

            # Remove temp file
            temp_file.unlink()

            success_count += 1
            if success_count % 20 == 0:
                print(f"  Processed {success_count}/{len(test_files)} files...")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"\nDone! Successfully processed {success_count}/{len(test_files)} files")


if __name__ == '__main__':
    main()

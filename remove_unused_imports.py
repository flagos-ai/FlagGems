#!/usr/bin/env python3
"""
Script to remove unused imports from test files using autoflake.
"""

import subprocess
from pathlib import Path


def remove_unused_imports(file_path):
    """Remove unused imports from a file using autoflake."""
    try:
        subprocess.run(
            [
                'autoflake',
                '--in-place',
                '--remove-all-unused-imports',
                '--remove-unused-variables',
                str(file_path)
            ],
            check=True,
            capture_output=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error processing {file_path}: {e}")
        return False
    except FileNotFoundError:
        print("autoflake not found, trying manual approach...")
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
        if remove_unused_imports(file_path):
            success_count += 1
            if success_count % 20 == 0:
                print(f"  Processed {success_count}/{len(all_files)} files...")

    print(f"\nDone! Successfully processed {success_count}/{len(all_files)} files")


if __name__ == '__main__':
    main()

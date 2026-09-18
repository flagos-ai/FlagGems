#!/usr/bin/env python3

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Auto-sort __all__ exports in Python files and operators in operators.yaml.

Usage:
    # Check if files are sorted (exit 0 if sorted, 1 if not)
    python tools/ci_checks/sort_exports.py --check

    # Fix sorting in all files
    python tools/ci_checks/sort_exports.py --fix

    # Fix specific files only
    python tools/ci_checks/sort_exports.py --fix --files src/flag_gems/__init__.py

    # Dry run (show what would be changed)
    python tools/ci_checks/sort_exports.py --fix --dry-run
"""

import argparse
import ast
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


def sort_python_all(file_path: Path, fix: bool = False, dry_run: bool = False) -> bool:
    """
    Sort __all__ list in a Python file by casefold.

    Args:
        file_path: Path to Python file
        fix: If True, modify the file; if False, only check
        dry_run: If True with fix, show changes but don't write

    Returns:
        True if file is already sorted, False otherwise
    """
    if not file_path.exists():
        print(f"Warning: {file_path} not found")
        return True

    source = file_path.read_text()

    # Match __all__ = [ ... ]
    # Use MULTILINE and DOTALL to handle multi-line lists
    pattern = r"^(__all__\s*=\s*\[)(.*?)(^\])"
    match = re.search(pattern, source, re.MULTILINE | re.DOTALL)

    if not match:
        # No __all__ found, that's fine
        return True

    prefix = match.group(1)  # "__all__ = ["
    content = match.group(2)  # the items
    suffix = match.group(3)  # "]"

    # Extract all string items (handles both " and ')
    items = re.findall(r'["\']([^"\']+)["\']', content)

    if not items:
        return True

    # Sort by casefold
    sorted_items = sorted(items, key=str.casefold)

    if items == sorted_items:
        return True  # Already sorted

    if not fix:
        print(f"❌ {file_path}: __all__ is not sorted by casefold")
        # Show first mismatch
        for i, (actual, expected) in enumerate(zip(items, sorted_items)):
            if actual != expected:
                print(f"   Position {i}: got '{actual}', expected '{expected}'")
                break
        return False

    # Detect indent and quote style from existing items
    lines = content.strip().split("\n")
    quote = '"'
    indent = 4  # default
    for line in lines:
        stripped = line.strip()
        if stripped:
            # Use lines with leading whitespace for indent detection
            if line != line.lstrip():
                indent = len(line) - len(line.lstrip())
            quote = '"' if '"' in stripped else "'"
            break

    indent_str = " " * indent

    # Rebuild the __all__ content
    new_items_lines = [f"{indent_str}{quote}{item}{quote}," for item in sorted_items]
    new_content = "\n" + "\n".join(new_items_lines) + "\n"

    new_source = (
        source[: match.start()] + prefix + new_content + suffix + source[match.end() :]
    )

    if dry_run:
        print(f"Would sort {file_path}: {len(items)} items")
        print(f"  First item: '{sorted_items[0]}'")
        print(f"  Last item: '{sorted_items[-1]}'")
        return False

    file_path.write_text(new_source)
    print(f"✅ {file_path}: sorted {len(items)} items in __all__")
    return False


def sort_operators_yaml(
    file_path: Path, fix: bool = False, dry_run: bool = False
) -> bool:
    """
    Sort operators.yaml by id.casefold().

    Args:
        file_path: Path to operators.yaml
        fix: If True, modify the file; if False, only check
        dry_run: If True with fix, show changes but don't write

    Returns:
        True if file is already sorted, False otherwise
    """
    if yaml is None:
        print(
            "Error: pyyaml is required for sorting operators.yaml. Install with: pip install pyyaml",
            file=sys.stderr,
        )
        sys.exit(2)

    if not file_path.exists():
        print(f"Warning: {file_path} not found")
        return True

    with open(file_path, "r") as f:
        data = yaml.safe_load(f)

    ops = data.get("ops", [])
    if not ops:
        return True

    # Extract ids
    ids = [op["id"] for op in ops]
    sorted_ids = sorted(ids, key=str.casefold)

    if ids == sorted_ids:
        return True  # Already sorted

    if not fix:
        print(f"❌ {file_path}: operators not sorted by id.casefold()")
        # Show first mismatch
        for i, (actual, expected) in enumerate(zip(ids, sorted_ids)):
            if actual != expected:
                print(f"   Position {i}: got '{actual}', expected '{expected}'")
                break
        return False

    # Sort the ops list
    sorted_ops = sorted(ops, key=lambda x: x["id"].casefold())

    if dry_run:
        print(f"Would sort {file_path}: {len(ops)} operators")
        print(f"  First: '{sorted_ops[0]['id']}'")
        print(f"  Last: '{sorted_ops[-1]['id']}'")
        return False

    # Read original file to preserve header comments
    original = file_path.read_text()
    lines = original.splitlines()

    # Find the "ops:" line
    ops_line_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "ops:":
            ops_line_idx = i
            break

    if ops_line_idx is None:
        print(f"Error: Cannot find 'ops:' line in {file_path}", file=sys.stderr)
        return False

    # Preserve header (copyright + "ops:")
    header = "\n".join(lines[: ops_line_idx + 1]) + "\n"

    # Write sorted YAML
    data["ops"] = sorted_ops
    yaml_content = yaml.dump(
        data, default_flow_style=False, allow_unicode=True, sort_keys=False
    )

    # Extract only the ops list from dumped YAML (skip "ops:" line)
    yaml_lines = yaml_content.splitlines()
    ops_start = None
    for i, line in enumerate(yaml_lines):
        if line.strip() == "ops:":
            ops_start = i + 1
            break

    if ops_start is None:
        print("Error: Cannot parse dumped YAML", file=sys.stderr)
        return False

    ops_body = "\n".join(yaml_lines[ops_start:])

    new_content = header + ops_body + "\n"
    file_path.write_text(new_content)
    print(f"✅ {file_path}: sorted {len(ops)} operators")
    return False


def sort_full_config(file_path: Path, fix: bool = False, dry_run: bool = False) -> bool:
    """Sort _FULL_CONFIG tuple in __init__.py by key (casefold).

    Args:
        file_path: Path to __init__.py
        fix: If True, modify the file; if False, just check
        dry_run: If True, print what would be done but don't write

    Returns:
        True if sorted or was fixed, False if unsorted and not fixed
    """
    if not file_path.exists():
        return True

    content = file_path.read_text()
    tree = ast.parse(content)

    # Find _FULL_CONFIG = (...) assignment
    config_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_FULL_CONFIG":
                    config_node = node
                    break

    if config_node is None or not isinstance(config_node.value, ast.Tuple):
        return True  # No _FULL_CONFIG tuple

    # Extract entries with full source text (including multi-line tuples)
    entries = []
    lines = content.splitlines()

    for elt in config_node.value.elts:
        if isinstance(elt, ast.Tuple) and len(elt.elts) >= 2:
            key_node = elt.elts[0]
            if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                key = key_node.value
                # Get the full source text for this tuple (may span multiple lines)
                start_line = elt.lineno - 1  # 0-indexed
                end_line = elt.end_lineno - 1  # 0-indexed

                if start_line == end_line:
                    # Single-line entry
                    source_text = lines[start_line].rstrip()
                else:
                    # Multi-line entry - preserve all lines
                    source_lines = [lines[start_line].rstrip()]
                    for i in range(start_line + 1, end_line + 1):
                        source_lines.append(lines[i].rstrip())
                    source_text = "\n".join(source_lines)

                entries.append((key, source_text))

    if not entries:
        return True

    # Check if sorted
    keys = [e[0] for e in entries]
    sorted_keys = sorted(keys, key=str.casefold)

    if keys == sorted_keys:
        return True  # Already sorted

    if not fix:
        print(f"❌ {file_path}: _FULL_CONFIG is not sorted by key.casefold()")
        for i, (actual, expected) in enumerate(zip(keys, sorted_keys)):
            if actual != expected:
                print(f"   Position {i}: got '{actual}', expected '{expected}'")
                break
        return False

    if dry_run:
        print(f"Would sort {file_path}: _FULL_CONFIG with {len(entries)} entries")
        return False

    # Sort entries by key
    sorted_entries = sorted(entries, key=lambda e: e[0].casefold())

    # Find the _FULL_CONFIG block boundaries
    start_line = config_node.lineno - 1  # Line with "_FULL_CONFIG = ("
    end_line = config_node.end_lineno - 1  # Line with closing ")"

    # Reconstruct
    new_lines = lines[:start_line]
    new_lines.append("_FULL_CONFIG = (")
    for entry in sorted_entries:
        # entry[1] may have multiple lines
        source_text = entry[1]
        if "\n" in source_text:
            # Multi-line entry
            new_lines.extend(source_text.split("\n"))
        else:
            new_lines.append(source_text)
    new_lines.append(")")
    new_lines.extend(lines[end_line + 1 :])

    new_content = "\n".join(new_lines) + "\n"
    file_path.write_text(new_content)
    print(f"✅ {file_path}: sorted _FULL_CONFIG")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Sort __all__ exports and operators.yaml by casefold"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if files are sorted (don't modify)",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Fix sorting issues by modifying files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="Specific files to process (default: all known files)",
    )

    args = parser.parse_args()

    if not args.check and not args.fix:
        parser.error("Must specify either --check or --fix")

    if args.check and args.fix:
        parser.error("Cannot use both --check and --fix")

    # Determine files to process
    if args.files:
        files_to_check = [Path(f) for f in args.files]
    else:
        # Default: all known files
        files_to_check = [
            Path("src/flag_gems/__init__.py"),
            Path("src/flag_gems/ops/__init__.py"),
            Path("conf/operators.yaml"),
        ]

    all_sorted = True

    for file_path in files_to_check:
        if file_path.name == "operators.yaml":
            sorted_ok = sort_operators_yaml(
                file_path, fix=args.fix, dry_run=args.dry_run
            )
        else:
            # Sort __all__
            sorted_ok = sort_python_all(file_path, fix=args.fix, dry_run=args.dry_run)
            # Also sort _FULL_CONFIG if this is __init__.py
            if file_path.name == "__init__.py":
                config_sorted = sort_full_config(
                    file_path, fix=args.fix, dry_run=args.dry_run
                )
                sorted_ok = sorted_ok and config_sorted

        all_sorted = all_sorted and sorted_ok

    if not all_sorted:
        if args.check:
            print("\n❌ Some files are not sorted. Run this to fix:")
            print("    python tools/ci_checks/sort_exports.py --fix")
            sys.exit(1)
        elif args.dry_run:
            print("\nRun without --dry-run to apply changes:")
            print("    python tools/ci_checks/sort_exports.py --fix")
            sys.exit(0)
        else:
            print("\n✅ All files have been sorted")
            sys.exit(0)
    else:
        if args.check:
            print("✅ All files are correctly sorted")
        sys.exit(0)


if __name__ == "__main__":
    main()

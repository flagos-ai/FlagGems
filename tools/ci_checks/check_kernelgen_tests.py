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

"""Check KernelGen test files for forbidden use_gems() calls.

Rules:
  1. Test files for KernelGen operators must NOT call flag_gems.use_gems()
     or use_gems() directly, as this bypasses reference implementation comparison.

Exit codes:
  0 - all checks pass
  1 - rule violations found
  2 - script internal error
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import yaml

OPERATORS_YAML = Path("conf/operators.yaml")
TESTS_DIR = Path("tests")


def get_kernelgen_operators() -> set[str]:
    """Get operator IDs that have 'KernelGen' label."""
    if not OPERATORS_YAML.exists():
        print(f"::error::Cannot find {OPERATORS_YAML}", file=sys.stderr)
        sys.exit(2)
    with open(OPERATORS_YAML) as f:
        data = yaml.safe_load(f)
    ops = data.get("ops", [])
    return {op["id"] for op in ops if "KernelGen" in op.get("labels", [])}


def find_use_gems_calls(filepath: Path) -> list[tuple[int, str]]:
    """Find use_gems() calls in a Python file using AST.

    Returns list of (line_number, code_snippet) tuples.
    """
    try:
        source = filepath.read_text()
        tree = ast.parse(source)
    except SyntaxError:
        # If we can't parse, skip gracefully
        return []

    violations = []
    source_lines = source.splitlines()

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Check for use_gems() or flag_gems.use_gems()
            func = node.func
            if isinstance(func, ast.Name) and func.id == "use_gems":
                line = (
                    source_lines[node.lineno - 1].strip()
                    if node.lineno <= len(source_lines)
                    else ""
                )
                violations.append((node.lineno, line))
            elif isinstance(func, ast.Attribute) and func.attr == "use_gems":
                # e.g., flag_gems.use_gems()
                line = (
                    source_lines[node.lineno - 1].strip()
                    if node.lineno <= len(source_lines)
                    else ""
                )
                violations.append((node.lineno, line))

    return violations


def main():
    parser = argparse.ArgumentParser(
        description="Check KernelGen tests for forbidden use_gems() calls"
    )
    parser.add_argument(
        "--operators",
        help="JSON list of operator IDs to check (incremental mode)",
        default="",
    )
    parser.add_argument(
        "--changed-files",
        help="JSON list of changed test file paths (incremental mode)",
        default="",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Check all KernelGen operator tests (full scan, not used in CI)",
    )
    args = parser.parse_args()

    kernelgen_ops = get_kernelgen_operators()
    print(f"KernelGen operators in registry: {len(kernelgen_ops)}")

    # Incremental mode: only check operators from the current PR
    if args.all:
        ops_to_check = kernelgen_ops
    elif args.operators:
        try:
            requested_ops = json.loads(args.operators)
        except json.JSONDecodeError:
            requested_ops = [
                op.strip() for op in args.operators.split(",") if op.strip()
            ]
        # Only check ops that are KernelGen
        ops_to_check = set(requested_ops) & kernelgen_ops
    elif args.changed_files:
        # Derive operators from changed test file paths
        try:
            changed = json.loads(args.changed_files)
        except json.JSONDecodeError:
            changed = [f.strip() for f in args.changed_files.split(",") if f.strip()]
        ops_from_files = set()
        for fp in changed:
            m = re.match(r"tests/test_(.+)\.py$", fp)
            if m:
                ops_from_files.add(m.group(1))
        ops_to_check = ops_from_files & kernelgen_ops
    else:
        # No operators specified and not --all: nothing to check (safe default)
        print("No operators specified. Use --operators or --all.")
        sys.exit(0)

    if not ops_to_check:
        print("No KernelGen operators to check.")
        sys.exit(0)

    print(f"Checking {len(ops_to_check)} operator test(s)...")
    all_violations = []

    for op_id in sorted(ops_to_check):
        # Find corresponding test file(s)
        test_file = TESTS_DIR / f"test_{op_id}.py"
        if not test_file.exists():
            # Try with leading underscore stripped from test name
            continue

        violations = find_use_gems_calls(test_file)
        if violations:
            for lineno, code in violations:
                msg = f"{test_file}:{lineno}: use_gems() call found: {code}"
                all_violations.append(msg)

    if all_violations:
        print(f"\n❌ Found {len(all_violations)} forbidden use_gems() call(s):\n")
        for v in all_violations:
            print(f"::error::{v}")
            print(f"  • {v}")
        sys.exit(1)
    else:
        print("✅ No forbidden use_gems() calls found.")
        sys.exit(0)


if __name__ == "__main__":
    main()

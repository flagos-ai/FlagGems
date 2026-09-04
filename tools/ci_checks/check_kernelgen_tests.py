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

"""Check test files for forbidden use_gems() calls.

Rules:
  1. Test files must NOT call flag_gems.use_gems() or use_gems() directly,
     as this bypasses reference implementation comparison.
  2. In incremental mode (default in CI), only PR-changed test files are checked.

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

TESTS_DIR = Path("tests")


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
        description="Check test files for forbidden use_gems() calls"
    )
    parser.add_argument(
        "--operators",
        help="JSON list of operator IDs to check (incremental mode)",
        default="",
    )
    parser.add_argument(
        "--changed-files",
        help="JSON list of changed file paths (incremental mode)",
        default="",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Check all test files (full scan)",
    )
    args = parser.parse_args()

    # Determine which test files to check
    if args.all:
        test_files = sorted(TESTS_DIR.glob("test_*.py"))
    elif args.changed_files:
        try:
            changed = json.loads(args.changed_files)
        except json.JSONDecodeError:
            changed = [f.strip() for f in args.changed_files.split(",") if f.strip()]
        test_files = [Path(f) for f in changed if re.match(r"tests/test_.+\.py$", f)]
    elif args.operators:
        try:
            requested_ops = json.loads(args.operators)
        except json.JSONDecodeError:
            requested_ops = [
                op.strip() for op in args.operators.split(",") if op.strip()
            ]
        test_files = [TESTS_DIR / f"test_{op_id}.py" for op_id in requested_ops]
    else:
        print(
            "No operators or files specified. Use --operators, --changed-files, or --all."
        )
        sys.exit(0)

    # Filter to files that actually exist
    test_files = [f for f in test_files if f.exists()]

    if not test_files:
        print("No test files to check.")
        sys.exit(0)

    print(f"Checking {len(test_files)} test file(s)...")
    all_violations = []

    for test_file in sorted(test_files):
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
        print(
            "\nTests must not call use_gems(). "
            "The test framework handles operator dispatch automatically."
        )
        sys.exit(1)
    else:
        print("✅ No forbidden use_gems() calls found.")
        sys.exit(0)


if __name__ == "__main__":
    main()

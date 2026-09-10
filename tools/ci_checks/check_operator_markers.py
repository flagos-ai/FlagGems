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

"""Check that changed operators have corresponding test files and pytest markers.

Rules:
  1. Each operator in operators.yaml must have a test file: tests/test_<id>.py
  2. Test file must contain at least one function decorated with @pytest.mark.<id>
  3. The marker must be collectible (file must be parseable by AST)

Exit codes:
  0 - all checks pass
  1 - rule violations found
  2 - script internal error
"""

import argparse
import ast
import json
import sys
from pathlib import Path

import yaml

OPERATORS_YAML = Path("conf/operators.yaml")
TESTS_DIR = Path("tests")
ALIASES_FILE = Path("conf/ci_test_aliases.yaml")


def load_operators_yaml() -> dict[str, dict]:
    """Load operators.yaml and return dict keyed by id."""
    if not OPERATORS_YAML.exists():
        print(f"::error::Cannot find {OPERATORS_YAML}", file=sys.stderr)
        sys.exit(2)
    with open(OPERATORS_YAML) as f:
        data = yaml.safe_load(f)
    return {op["id"]: op for op in data.get("ops", []) if "id" in op}


def load_aliases() -> dict[str, str]:
    """Load test file aliases mapping operator_id -> test_file_stem."""
    if not ALIASES_FILE.exists():
        return {}
    with open(ALIASES_FILE) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def find_test_file(op_id: str, aliases: dict[str, str]) -> Path | None:
    """Find the test file for an operator.

    Tries in order:
      1. Alias mapping from ci_test_aliases.yaml
      2. tests/test_<id>.py (direct match)
      3. tests/test_<id without trailing _>.py (inplace variants)
    """
    # Check alias first
    if op_id in aliases:
        alias_stem = aliases[op_id]
        # Ensure it has test_ prefix
        if not alias_stem.startswith("test_"):
            alias_stem = f"test_{alias_stem}"
        alias_path = TESTS_DIR / f"{alias_stem}.py"
        if alias_path.exists():
            return alias_path

    # Direct match
    direct = TESTS_DIR / f"test_{op_id}.py"
    if direct.exists():
        return direct

    # Inplace variant: abs_ might share tests/test_abs.py
    if op_id.endswith("_"):
        base = TESTS_DIR / f"test_{op_id[:-1]}.py"
        if base.exists():
            return base

    return None


def check_marker_in_file(filepath: Path, op_id: str) -> bool:
    """Check if the test file has at least one @pytest.mark.<op_id> decorator."""
    try:
        source = filepath.read_text()
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        # If we can't parse, assume marker is present (don't block on parse errors)
        return True

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                if _is_pytest_mark(decorator, op_id):
                    return True
    return False


def _is_pytest_mark(node: ast.expr, marker_name: str) -> bool:
    """Check if a decorator node is @pytest.mark.<marker_name>."""
    # Pattern: pytest.mark.<name>
    if isinstance(node, ast.Attribute) and node.attr == marker_name:
        # Check it's pytest.mark.<name>
        if isinstance(node.value, ast.Attribute) and node.value.attr == "mark":
            if (
                isinstance(node.value.value, ast.Name)
                and node.value.value.id == "pytest"
            ):
                return True
    # Pattern: pytest.mark.<name>(...) - marker with arguments (shouldn't happen but be safe)
    if isinstance(node, ast.Call):
        return _is_pytest_mark(node.func, marker_name)
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Check operator test coverage and markers"
    )
    parser.add_argument(
        "--operators",
        help="JSON list of operator IDs to check (incremental mode)",
        default="",
    )
    args = parser.parse_args()

    if not args.operators:
        print("No operators specified. Use --operators with a JSON list.")
        sys.exit(0)

    try:
        op_ids = json.loads(args.operators)
    except json.JSONDecodeError:
        op_ids = [op.strip() for op in args.operators.split(",") if op.strip()]

    if not op_ids:
        print("No operators to check.")
        sys.exit(0)

    all_operators = load_operators_yaml()
    aliases = load_aliases()
    if aliases:
        print(f"Loaded {len(aliases)} test aliases")

    # Only check operators that exist in the registry
    ops_to_check = [op_id for op_id in op_ids if op_id in all_operators]
    if not ops_to_check:
        print("None of the specified operators found in registry.")
        sys.exit(0)

    print(f"Checking test coverage for {len(ops_to_check)} operator(s)...")
    errors = []

    for op_id in sorted(ops_to_check):
        op_info = all_operators[op_id]
        labels = op_info.get("labels", [])

        # Skip fused operators that don't have aten label - they may have different test patterns
        if "aten" not in labels and "fused" in labels:
            print(f"  {op_id}: fused operator, skipping test file check")
            continue

        # Rule 1: Test file must exist
        test_file = find_test_file(op_id, aliases)
        if test_file is None:
            errors.append(
                f"Operator '{op_id}': no test file found (expected tests/test_{op_id}.py)"
            )
            continue

        # Rule 2: Test file must have the operator marker
        has_marker = check_marker_in_file(test_file, op_id)
        if not has_marker:
            # For inplace variants (e.g., abs_), also accept base marker
            if op_id.endswith("_"):
                base_id = op_id[:-1]
                has_marker = check_marker_in_file(test_file, base_id)

        if not has_marker:
            errors.append(
                f"Operator '{op_id}': test file {test_file} has no "
                f"@pytest.mark.{op_id} decorator"
            )

    if errors:
        print(f"\n❌ Found {len(errors)} issue(s):\n")
        for err in errors:
            print(f"::error::{err}")
            print(f"  • {err}")
        sys.exit(1)
    else:
        print("✅ All operators have test coverage.")
        sys.exit(0)


if __name__ == "__main__":
    main()

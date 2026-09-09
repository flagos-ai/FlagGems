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

"""Check __init__.py for export list ordering and registry key duplicates.

Rules:
  1. __all__ entries must be sorted by casefold
  2. __all__ must not contain duplicates
  3. _FULL_CONFIG must not contain duplicate aten op name keys
  4. _FULL_CONFIG must be sorted by key (casefold)

Exit codes:
  0 - all checks pass
  1 - rule violations found
  2 - script internal error
"""

import ast
import sys
from pathlib import Path

INIT_FILE = Path("src/flag_gems/__init__.py")


def extract_all_list(tree: ast.Module) -> list[str] | None:
    """Extract __all__ list from AST."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, ast.List):
                        elements = []
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(
                                elt.value, str
                            ):
                                elements.append(elt.value)
                        return elements
    return None


def extract_full_config_keys(source: str) -> list[tuple[str, int]]:
    """Extract the first element (aten op name) from each tuple in _FULL_CONFIG.

    Returns list of (key, line_number) tuples.

    Uses AST to find the _FULL_CONFIG assignment and extract string keys
    from the tuple-of-tuples structure.
    """
    tree = ast.parse(source)
    keys = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_FULL_CONFIG":
                    # _FULL_CONFIG is a tuple of tuples
                    if isinstance(node.value, ast.Tuple):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Tuple) and len(elt.elts) >= 2:
                                first = elt.elts[0]
                                if isinstance(first, ast.Constant) and isinstance(
                                    first.value, str
                                ):
                                    keys.append((first.value, first.lineno))
    return keys


def check_all_sorted(all_list: list[str]) -> list[str]:
    """Check __all__ is sorted by casefold and has no duplicates."""
    errors = []

    # Check duplicates
    seen = set()
    for item in all_list:
        if item in seen:
            errors.append(f"__all__ contains duplicate entry: '{item}'")
        seen.add(item)

    # Check sort order
    sorted_list = sorted(all_list, key=lambda x: x.casefold())
    if all_list != sorted_list:
        # Find first mismatch
        for i, (actual, expected) in enumerate(zip(all_list, sorted_list)):
            if actual != expected:
                errors.append(
                    f"__all__ is not sorted by casefold. "
                    f"Position {i}: got '{actual}', expected '{expected}'"
                )
                break

    return errors


# Known intentional duplicate keys in _FULL_CONFIG (by design)
KNOWN_DUPLICATE_KEYS = {
    "ne_.Scalar",
    "ne_.Tensor",
}


def check_config_duplicates(keys: list[tuple[str, int]]) -> list[str]:
    """Check _FULL_CONFIG for duplicate aten op name keys.

    Known intentional duplicates are excluded from error reporting.
    """
    errors = []
    seen: dict[str, int] = {}

    for key, lineno in keys:
        if key in seen:
            if key not in KNOWN_DUPLICATE_KEYS:
                errors.append(
                    f"_FULL_CONFIG has duplicate key '{key}': "
                    f"first at line {seen[key]}, duplicate at line {lineno}"
                )
        else:
            seen[key] = lineno

    return errors


def check_config_sorted(keys: list[tuple[str, int]]) -> list[str]:
    """Check _FULL_CONFIG entries are sorted by key (casefold).

    Reports each out-of-order entry (where key < previous key).
    """
    errors = []
    if len(keys) < 2:
        return errors

    for i in range(1, len(keys)):
        prev_key, _ = keys[i - 1]
        cur_key, cur_line = keys[i]
        if cur_key.casefold() < prev_key.casefold():
            errors.append(
                f"_FULL_CONFIG is not sorted: '{cur_key}' (line {cur_line}) "
                f"should come before '{prev_key}'"
            )

    return errors


def main():
    if not INIT_FILE.exists():
        print(f"::error::Cannot find {INIT_FILE}", file=sys.stderr)
        sys.exit(2)

    try:
        source = INIT_FILE.read_text()
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"::error::Failed to parse {INIT_FILE}: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"Checking {INIT_FILE}...")
    all_errors = []

    # Check __all__
    all_list = extract_all_list(tree)
    if all_list is not None:
        print(f"  __all__ has {len(all_list)} entries")
        all_errors.extend(check_all_sorted(all_list))
    else:
        print("  __all__ not found (skipping __all__ checks)")

    # Check _FULL_CONFIG
    config_keys = extract_full_config_keys(source)
    if config_keys:
        print(f"  _FULL_CONFIG has {len(config_keys)} entries")
        all_errors.extend(check_config_duplicates(config_keys))
        all_errors.extend(check_config_sorted(config_keys))
    else:
        print("  _FULL_CONFIG not found or empty (skipping registry checks)")

    if all_errors:
        print(f"\n❌ Found {len(all_errors)} issue(s):\n")
        for err in all_errors:
            print(f"::error file={INIT_FILE}::{err}")
            print(f"  • {err}")
        print("\n💡 To fix sorting issues, run:")
        print("   python tools/ci_checks/sort_exports.py --fix")
        print("   git add src/flag_gems/__init__.py")
        print("   git commit -m 'fix: sort __all__ exports'")
        sys.exit(1)
    else:
        print("✅ All checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()

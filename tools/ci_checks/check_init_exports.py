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


def discover_init_files() -> list[Path]:
    """Discover all __init__.py files that should be checked.

    Returns __init__.py files from:
      - Main package: src/flag_gems/__init__.py
      - Backend ops: src/flag_gems/runtime/backend/*/*/ops/__init__.py

    Excludes utility/helper packages (utils, fused, etc.) since they typically
    don't export operators with the same naming conventions.
    """
    root = Path("src/flag_gems")
    if not root.exists():
        return []

    files = []

    # 1. Main package init (contains _FULL_CONFIG)
    main_init = root / "__init__.py"
    if main_init.exists():
        files.append(main_init)

    # 2. All backend ops __init__.py files
    # Pattern: src/flag_gems/runtime/backend/_<vendor>/ops/__init__.py
    # Pattern: src/flag_gems/runtime/backend/_<vendor>/<arch>/ops/__init__.py
    backend_root = root / "runtime" / "backend"
    if backend_root.exists():
        for vendor_dir in backend_root.iterdir():
            if not vendor_dir.is_dir() or not vendor_dir.name.startswith("_"):
                continue

            # Check vendor-level ops/
            vendor_ops_init = vendor_dir / "ops" / "__init__.py"
            if vendor_ops_init.exists():
                files.append(vendor_ops_init)

            # Check architecture-level ops/ (e.g., _nvidia/hopper/ops/)
            for arch_dir in vendor_dir.iterdir():
                if not arch_dir.is_dir():
                    continue
                arch_ops_init = arch_dir / "ops" / "__init__.py"
                if arch_ops_init.exists():
                    files.append(arch_ops_init)

    return sorted(files)


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


def check_single_file(file_path: Path) -> list[str]:
    """Check a single __init__.py file and return any errors found."""
    errors = []

    try:
        source = file_path.read_text()
        tree = ast.parse(source)
    except SyntaxError as e:
        errors.append(f"Failed to parse {file_path}: {e}")
        return errors

    # Check __all__
    all_list = extract_all_list(tree)
    if all_list is not None:
        for err in check_all_sorted(all_list):
            errors.append(err)

    # Check _FULL_CONFIG (only in main __init__.py)
    if file_path.name == "__init__.py" and "_FULL_CONFIG" in source:
        config_keys = extract_full_config_keys(source)
        if config_keys:
            errors.extend(check_config_duplicates(config_keys))
            errors.extend(check_config_sorted(config_keys))

    return errors


def main():
    init_files = discover_init_files()

    if not init_files:
        print("::error::No __init__.py files found to check", file=sys.stderr)
        sys.exit(2)

    print(f"Discovered {len(init_files)} __init__.py file(s) to check:\n")
    for f in init_files:
        print(f"  • {f}")
    print()

    all_errors_by_file = {}
    total_errors = 0

    for init_file in init_files:
        print(f"Checking {init_file}...")

        if not init_file.exists():
            print("  ⚠️  File not found, skipping")
            continue

        try:
            source = init_file.read_text()
            tree = ast.parse(source)
        except SyntaxError as e:
            error_msg = f"Failed to parse: {e}"
            print(f"  ❌ {error_msg}")
            all_errors_by_file[init_file] = [error_msg]
            total_errors += 1
            continue

        file_errors = []

        # Check __all__
        all_list = extract_all_list(tree)
        if all_list is not None:
            print(f"  __all__ has {len(all_list)} entries")
            all_errors = check_all_sorted(all_list)
            file_errors.extend(all_errors)
        else:
            print("  __all__ not found (skipping __all__ checks)")

        # Check _FULL_CONFIG (only in main package __init__.py)
        if str(init_file) == "src/flag_gems/__init__.py":
            config_keys = extract_full_config_keys(source)
            if config_keys:
                print(f"  _FULL_CONFIG has {len(config_keys)} entries")
                file_errors.extend(check_config_duplicates(config_keys))
                file_errors.extend(check_config_sorted(config_keys))
            else:
                print("  _FULL_CONFIG not found or empty (skipping registry checks)")

        if file_errors:
            all_errors_by_file[init_file] = file_errors
            total_errors += len(file_errors)
            print(f"  ❌ Found {len(file_errors)} issue(s)")
        else:
            print("  ✅ All checks passed")

        print()

    if total_errors > 0:
        print(f"{'=' * 70}")
        print(
            f"❌ Found {total_errors} issue(s) across {len(all_errors_by_file)} file(s):\n"
        )

        for file_path, errors in all_errors_by_file.items():
            print(f"📁 {file_path}:")
            for err in errors:
                print(f"::error file={file_path}::{err}")
                print(f"  • {err}")
            print()

        print("💡 To fix sorting issues, run:")
        print("   python tools/ci_checks/sort_exports.py --fix")
        print("   git add <affected files>")
        print("   git commit -m 'fix: sort __all__ exports'")
        sys.exit(1)
    else:
        print(f"{'=' * 70}")
        print(f"✅ All {len(init_files)} file(s) passed all checks.")
        sys.exit(0)


if __name__ == "__main__":
    main()

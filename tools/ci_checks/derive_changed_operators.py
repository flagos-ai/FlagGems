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

"""Derive which operators are affected by a PR based on git diff.

Outputs:
  - changed_operators: JSON list of operator IDs
  - changed_files: JSON list of changed file paths
  - has_changes: 'true' or 'false'

Exit codes:
  0 - success
  2 - script internal error
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

# Paths relative to repo root
OPERATORS_YAML = "conf/operators.yaml"
OPS_DIR = "src/flag_gems/ops"
TESTS_DIR = "tests"
INIT_FILE = "src/flag_gems/__init__.py"

# Patterns that map file paths to operator IDs
OPS_FILE_RE = re.compile(r"^src/flag_gems/ops/(.+)\.py$")
TEST_FILE_RE = re.compile(r"^tests/test_(.+)\.py$")


def get_diff_files(base_sha: str, head_sha: str) -> list[str]:
    """Get list of changed files between base and head."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=ACMR", base_sha, head_sha],
            capture_output=True,
            text=True,
            check=True,
        )
        return [f.strip() for f in result.stdout.splitlines() if f.strip()]
    except subprocess.CalledProcessError as e:
        print(f"::error::Failed to get git diff: {e.stderr}", file=sys.stderr)
        sys.exit(2)


def load_operators_yaml() -> dict[str, dict]:
    """Load operators.yaml and return a dict keyed by operator id."""
    yaml_path = Path(OPERATORS_YAML)
    if not yaml_path.exists():
        print(f"::error::Cannot find {OPERATORS_YAML}", file=sys.stderr)
        sys.exit(2)
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    ops = data.get("ops", [])
    return {op["id"]: op for op in ops if "id" in op}


def filename_to_operator_id(filename: str) -> str | None:
    """Convert a filename stem to a potential operator id.

    Handles cases like:
      - test_abs.py -> abs
      - test__reshape_alias.py -> _reshape_alias
      - abs.py (in ops/) -> abs
    """
    # Strip leading underscore that is part of the name, not a prefix artifact
    return filename if filename else None


def derive_operators(changed_files: list[str], all_operators: dict) -> list[str]:
    """Given changed files, derive which operator IDs are affected."""
    changed_ops = set()

    for filepath in changed_files:
        # Case 1: operators.yaml itself changed -> check all operators in diff
        if filepath == OPERATORS_YAML:
            # When operators.yaml changes, we flag all operators for full check
            # In practice, individual checks will determine what to validate
            changed_ops.add("__operators_yaml_changed__")
            continue

        # Case 2: ops source file changed
        m = OPS_FILE_RE.match(filepath)
        if m:
            stem = m.group(1)
            # Handle subdirectory ops like ops/sub/file.py -> sub/file
            # But most ops are flat: ops/abs.py -> abs
            op_id = stem.replace("/", "_")
            if op_id == "__init__":
                continue
            # Try exact match first
            if op_id in all_operators:
                changed_ops.add(op_id)
            else:
                # Try without trailing underscore (inplace variants)
                # e.g., ops file might be abs_.py for abs_ operator
                changed_ops.add(op_id)
            continue

        # Case 3: test file changed
        m = TEST_FILE_RE.match(filepath)
        if m:
            stem = m.group(1)
            if stem in all_operators:
                changed_ops.add(stem)
            else:
                # Still track it, checks can use it
                changed_ops.add(stem)
            continue

        # Case 4: __init__.py changed -> full init check needed
        if filepath == INIT_FILE:
            changed_ops.add("__init_changed__")
            continue

    # Remove sentinel markers from operator list for downstream
    sentinel_markers = {"__operators_yaml_changed__", "__init_changed__"}
    real_ops = sorted(changed_ops - sentinel_markers)

    return real_ops


def set_output(name: str, value: str):
    """Set a GitHub Actions output variable."""
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            # Use delimiter for multiline values
            if "\n" in value:
                f.write(f"{name}<<EOF\n{value}\nEOF\n")
            else:
                f.write(f"{name}={value}\n")
    else:
        # Running locally, just print
        print(f"  {name}={value}")


def main():
    parser = argparse.ArgumentParser(
        description="Derive changed operators from PR diff"
    )
    parser.add_argument("--base", required=True, help="Base commit SHA")
    parser.add_argument("--head", required=True, help="Head commit SHA")
    args = parser.parse_args()

    print(f"Comparing {args.base}..{args.head}")

    changed_files = get_diff_files(args.base, args.head)
    print(f"Changed files ({len(changed_files)}):")
    for f in changed_files[:20]:
        print(f"  {f}")
    if len(changed_files) > 20:
        print(f"  ... and {len(changed_files) - 20} more")

    all_operators = load_operators_yaml()
    print(f"Total operators in registry: {len(all_operators)}")

    changed_ops = derive_operators(changed_files, all_operators)
    print(f"Changed operators ({len(changed_ops)}):")
    for op in changed_ops[:20]:
        print(f"  {op}")
    if len(changed_ops) > 20:
        print(f"  ... and {len(changed_ops) - 20} more")

    # Set outputs
    ops_json = json.dumps(changed_ops)
    files_json = json.dumps(changed_files)
    has_changes = "true" if changed_ops else "false"

    set_output("changed_operators", ops_json)
    set_output("changed_files", files_json)
    set_output("has_changes", has_changes)

    print(f"\nhas_changes={has_changes}")


if __name__ == "__main__":
    main()

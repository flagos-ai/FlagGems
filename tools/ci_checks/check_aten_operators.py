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

"""Check that aten operator names in operators.yaml are valid.

Rules:
  1. Operators with 'aten' label must have a non-empty 'for' field
  2. Each name in 'for' must match the aten naming pattern:
     - Simple: <op_name> (e.g., "abs", "_reshape_alias")
     - Overloaded: <op_name>.<overload> (e.g., "div.Scalar_mode")
  3. If a static allowlist is available, validate names against it
  4. Non-aten operators (fused, vLLM, etc.) with 'for' set to 'None' string are acceptable

Exit codes:
  0 - all checks pass
  1 - rule violations found
  2 - script internal error
"""

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

OPERATORS_YAML = Path("conf/operators.yaml")
ATEN_ALLOWLIST_FILE = Path("conf/aten_op_allowlist.txt")

# Pattern for valid aten operator names
# Allows: word chars, dots for overloads, leading underscores
ATEN_NAME_PATTERN = re.compile(r"^_?[a-zA-Z][a-zA-Z0-9_]*(\.[a-zA-Z][a-zA-Z0-9_]*)?$")


def load_operators_yaml() -> dict[str, dict]:
    """Load operators.yaml and return dict keyed by id."""
    if not OPERATORS_YAML.exists():
        print(f"::error::Cannot find {OPERATORS_YAML}", file=sys.stderr)
        sys.exit(2)
    with open(OPERATORS_YAML) as f:
        data = yaml.safe_load(f)
    return {op["id"]: op for op in data.get("ops", []) if "id" in op}


def load_allowlist() -> set[str] | None:
    """Load the aten operator allowlist if available.

    Returns None if the file doesn't exist (allowlist-based validation is optional).
    """
    if not ATEN_ALLOWLIST_FILE.exists():
        return None
    with open(ATEN_ALLOWLIST_FILE) as f:
        return {line.strip() for line in f if line.strip() and not line.startswith("#")}


def check_aten_names(
    op_id: str, op_info: dict, allowlist: set[str] | None
) -> list[str]:
    """Validate aten names for a single operator."""
    errors = []
    labels = op_info.get("labels", [])

    # Only check operators with 'aten' label
    if "aten" not in labels:
        return errors

    for_field = op_info.get("for")

    # Rule 1: aten operators must have a non-empty 'for' field
    if for_field is None:
        errors.append(
            f"Operator '{op_id}': has 'aten' label but 'for' field is None/missing"
        )
        return errors

    if isinstance(for_field, str) and for_field == "None":
        errors.append(
            f"Operator '{op_id}': has 'aten' label but 'for' is string 'None'"
        )
        return errors

    if not isinstance(for_field, list) or len(for_field) == 0:
        errors.append(f"Operator '{op_id}': 'for' must be a non-empty list")
        return errors

    # Rule 2: Each name must match the aten naming pattern
    for name in for_field:
        if not isinstance(name, str):
            errors.append(
                f"Operator '{op_id}': 'for' contains non-string value: {name}"
            )
            continue

        if not ATEN_NAME_PATTERN.match(name):
            errors.append(
                f"Operator '{op_id}': aten name '{name}' does not match expected pattern "
                f"(expected: <op_name> or <op_name>.<overload>)"
            )

        # Rule 3: If allowlist exists, validate against it
        if allowlist is not None:
            # Strip overload for base-name check
            base_name = name.split(".")[0]
            if base_name not in allowlist and name not in allowlist:
                errors.append(
                    f"Operator '{op_id}': aten name '{name}' (base: '{base_name}') "
                    f"not found in allowlist"
                )

    return errors


def main():
    parser = argparse.ArgumentParser(description="Check aten operator name validity")
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
    allowlist = load_allowlist()
    if allowlist:
        print(f"Loaded aten allowlist: {len(allowlist)} entries")
    else:
        print("No aten allowlist found, skipping allowlist validation")

    # Only check operators that exist in the registry
    ops_to_check = [op_id for op_id in op_ids if op_id in all_operators]
    if not ops_to_check:
        print("None of the specified operators found in registry.")
        sys.exit(0)

    print(f"Checking aten names for {len(ops_to_check)} operator(s)...")
    all_errors = []

    for op_id in sorted(ops_to_check):
        all_errors.extend(check_aten_names(op_id, all_operators[op_id], allowlist))

    if all_errors:
        print(f"\n❌ Found {len(all_errors)} issue(s):\n")
        for err in all_errors:
            print(f"::error::{err}")
            print(f"  • {err}")
        sys.exit(1)
    else:
        print("✅ All aten names are valid.")
        sys.exit(0)


if __name__ == "__main__":
    main()

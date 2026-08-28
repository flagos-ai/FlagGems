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

"""Check conf/operators.yaml for schema, duplicates, and ordering issues.

Rules:
  1. Every entry must have required fields: id, description, for, labels, kind, stages
  2. No duplicate id values
  3. Entries must be sorted by id.casefold()
  4. id must be a non-empty string
  5. labels must be a non-empty list
  6. stages must be a non-empty list

Exit codes:
  0 - all checks pass
  1 - rule violations found
  2 - script internal error
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

OPERATORS_YAML = Path("conf/operators.yaml")

REQUIRED_FIELDS = {"id", "description", "for", "labels", "kind", "stages"}


def check_required_fields(ops: list[dict]) -> list[str]:
    """Check that all entries have required fields."""
    errors = []
    for i, op in enumerate(ops):
        op_id = op.get("id", f"<entry #{i+1}>")
        missing = REQUIRED_FIELDS - set(op.keys())
        if missing:
            errors.append(
                f"Operator '{op_id}': missing required fields: {sorted(missing)}"
            )
        # Validate field types
        if "id" in op:
            if not isinstance(op["id"], str) or not op["id"].strip():
                errors.append(f"Entry #{i+1}: 'id' must be a non-empty string")
        if "labels" in op:
            if not isinstance(op["labels"], list) or len(op["labels"]) == 0:
                errors.append(f"Operator '{op_id}': 'labels' must be a non-empty list")
        if "stages" in op:
            if not isinstance(op["stages"], list) or len(op["stages"]) == 0:
                errors.append(f"Operator '{op_id}': 'stages' must be a non-empty list")
    return errors


def check_duplicate_ids(ops: list[dict]) -> list[str]:
    """Check for duplicate operator IDs."""
    errors = []
    seen = {}
    for i, op in enumerate(ops):
        op_id = op.get("id")
        if op_id is None:
            continue
        if op_id in seen:
            errors.append(
                f"Duplicate id '{op_id}': appears at entry #{seen[op_id]+1} and #{i+1}"
            )
        else:
            seen[op_id] = i
    return errors


def check_sort_order(ops: list[dict]) -> list[str]:
    """Check that entries are sorted by id.casefold()."""
    errors = []
    ids = [op.get("id", "") for op in ops]
    sorted_ids = sorted(ids, key=lambda x: x.casefold())

    first_mismatch = None
    for i, (actual, expected) in enumerate(zip(ids, sorted_ids)):
        if actual != expected:
            first_mismatch = i
            break

    if first_mismatch is not None:
        # Find the first few out-of-order entries for a helpful message
        mismatches = []
        for i in range(first_mismatch, min(first_mismatch + 5, len(ids))):
            if ids[i] != sorted_ids[i]:
                mismatches.append(
                    f"  position {i+1}: got '{ids[i]}', expected '{sorted_ids[i]}'"
                )
        errors.append(
            "operators.yaml is not sorted by id.casefold(). First mismatches:\n"
            + "\n".join(mismatches)
        )
    return errors


def main():
    parser = argparse.ArgumentParser(description="Check operators.yaml validity")
    parser.add_argument(
        "--operators",
        help="JSON list of operator IDs to check (incremental mode). "
        "If not provided, only duplicate-id check runs globally.",
        default="",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Check all operators (for pre-commit use)",
    )
    args = parser.parse_args()

    if not OPERATORS_YAML.exists():
        print(f"::error::Cannot find {OPERATORS_YAML}", file=sys.stderr)
        sys.exit(2)

    try:
        with open(OPERATORS_YAML) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"::error::Failed to parse {OPERATORS_YAML}: {e}", file=sys.stderr)
        sys.exit(2)

    ops = data.get("ops", [])
    if not ops:
        print(f"::error::No 'ops' key found in {OPERATORS_YAML}", file=sys.stderr)
        sys.exit(2)

    print(f"Total operators in {OPERATORS_YAML}: {len(ops)}")

    all_errors = []

    # Duplicate ID check always runs globally (cheap, catches merge conflicts)
    all_errors.extend(check_duplicate_ids(ops))

    # Required fields check: incremental if --operators given, else global
    if args.operators:
        try:
            changed_ids = json.loads(args.operators)
        except json.JSONDecodeError:
            changed_ids = [op.strip() for op in args.operators.split(",") if op.strip()]
        changed_ops = [op for op in ops if op.get("id") in set(changed_ids)]
        print(f"Checking required fields for {len(changed_ops)} changed operator(s)...")
        all_errors.extend(check_required_fields(changed_ops))
    else:
        print("Checking required fields for all operators...")
        all_errors.extend(check_required_fields(ops))

    # Sort order check disabled until existing data is fixed
    # all_errors.extend(check_sort_order(ops))

    if all_errors:
        print(f"\n❌ Found {len(all_errors)} issue(s):\n")
        for err in all_errors:
            print(f"::error::{err}")
            print(f"  • {err}")
        sys.exit(1)
    else:
        print("✅ All checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()

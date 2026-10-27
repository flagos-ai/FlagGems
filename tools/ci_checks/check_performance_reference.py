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

"""Check that changed operators have corresponding benchmark files.

Rules (warning-only, does not block PR):
  1. Each operator should have a benchmark file: benchmark/test_<id>.py
  2. Benchmark file should contain at least one benchmark class or run() call

This helps ensure performance tracking is maintained as operators are added.

Exit codes:
  0 - all checks pass (warnings are still exit 0)
  2 - script internal error
"""

import argparse
import ast
import json
import sys
from pathlib import Path

import yaml

OPERATORS_YAML = Path("conf/operators.yaml")
BENCHMARK_DIR = Path("benchmark")


def load_operators_yaml() -> dict[str, dict]:
    """Load operators.yaml and return dict keyed by id."""
    if not OPERATORS_YAML.exists():
        print(f"::error::Cannot find {OPERATORS_YAML}", file=sys.stderr)
        sys.exit(2)
    with open(OPERATORS_YAML) as f:
        data = yaml.safe_load(f)
    return {op["id"]: op for op in data.get("ops", []) if "id" in op}


def find_benchmark_file(op_id: str) -> Path | None:
    """Find the benchmark file for an operator."""
    direct = BENCHMARK_DIR / f"test_{op_id}.py"
    if direct.exists():
        return direct

    # Inplace variant shares benchmark with base
    if op_id.endswith("_"):
        base = BENCHMARK_DIR / f"test_{op_id[:-1]}.py"
        if base.exists():
            return base

    return None


def check_benchmark_content(filepath: Path) -> bool:
    """Check that the benchmark file has meaningful content.

    Looks for:
    - A call to .run() (benchmark execution)
    - Or a class inheriting from a Benchmark base class
    """
    try:
        source = filepath.read_text()
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        # If we can't parse, assume it's fine
        return True

    for node in ast.walk(tree):
        # Look for .run() calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "run":
                return True
        # Look for class definitions (benchmark classes)
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if isinstance(base, ast.Attribute):
                    if "Benchmark" in base.attr:
                        return True
                if isinstance(base, ast.Name):
                    if "Benchmark" in base.id:
                        return True

    return False


def main():
    parser = argparse.ArgumentParser(
        description="Check operator benchmark file coverage"
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
    ops_to_check = [op_id for op_id in op_ids if op_id in all_operators]

    if not ops_to_check:
        print("None of the specified operators found in registry.")
        sys.exit(0)

    print(f"Checking benchmark coverage for {len(ops_to_check)} operator(s)...")
    all_warnings = []

    for op_id in sorted(ops_to_check):
        op_info = all_operators[op_id]
        labels = op_info.get("labels", [])

        # Skip fused/vLLM operators - they often have different benchmark patterns
        if "fused" in labels and "aten" not in labels:
            continue

        benchmark_file = find_benchmark_file(op_id)
        if benchmark_file is None:
            all_warnings.append(
                f"Operator '{op_id}': no benchmark file found "
                f"(expected benchmark/test_{op_id}.py)"
            )
            continue

        if not check_benchmark_content(benchmark_file):
            all_warnings.append(
                f"Operator '{op_id}': benchmark file {benchmark_file} "
                f"has no .run() call or Benchmark class"
            )

    if all_warnings:
        print(f"\n⚠️  Found {len(all_warnings)} warning(s):\n")
        for w in all_warnings:
            print(f"::warning::{w}")
            print(f"  • {w}")
    else:
        print("✅ All operators have benchmark coverage.")

    # Always exit 0 - this is a warning-only check
    sys.exit(0)


if __name__ == "__main__":
    main()

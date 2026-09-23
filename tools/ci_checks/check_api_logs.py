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

"""Check that operator implementations follow the API logging convention.

Convention:
  1. Each operator source file must import logging and create a logger:
       import logging
       logger = logging.getLogger(__name__)

  2. Each public function must contain a logger.debug("GEMS <OP_NAME>") call
     at its entry point. The op name should be uppercase.

This check emits warnings (annotations) rather than blocking the PR.

Exit codes:
  0 - all checks pass (or only warnings)
  1 - hard errors found (reserved for future use)
  2 - script internal error
"""

import argparse
import ast
import json
import sys
from pathlib import Path

import yaml

OPERATORS_YAML = Path("conf/operators.yaml")
OPS_DIR = Path("src/flag_gems/ops")


def load_operators_yaml() -> dict[str, dict]:
    """Load operators.yaml and return dict keyed by id."""
    if not OPERATORS_YAML.exists():
        print(f"::error::Cannot find {OPERATORS_YAML}", file=sys.stderr)
        sys.exit(2)
    with open(OPERATORS_YAML) as f:
        data = yaml.safe_load(f)
    return {op["id"]: op for op in data.get("ops", []) if "id" in op}


def find_op_source(op_id: str) -> Path | None:
    """Find the source file for an operator."""
    # Direct match
    direct = OPS_DIR / f"{op_id}.py"
    if direct.exists():
        return direct

    # Some ops use different file names
    # e.g., abs_ might be in abs.py
    if op_id.endswith("_"):
        base = OPS_DIR / f"{op_id[:-1]}.py"
        if base.exists():
            return base

    return None


def check_logging_setup(tree: ast.Module) -> bool:
    """Check if the file has proper logging setup."""
    has_import = False
    has_logger = False

    for node in ast.walk(tree):
        # Check: import logging
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "logging":
                    has_import = True
        # Check: logger = logging.getLogger(__name__)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "logger":
                    has_logger = True

    return has_import and has_logger


def _has_skip_decorator(node: ast.FunctionDef, skip_names: set[str]) -> bool:
    """Check if a function has any decorator that marks it as a kernel/helper."""
    for dec in node.decorator_list:
        # @triton.jit or @pointwise_dynamic
        if isinstance(dec, ast.Attribute):
            if dec.attr in skip_names:
                return True
            # e.g., triton.jit -> check "jit"
        if isinstance(dec, ast.Name):
            if dec.id in skip_names:
                return True
        # @decorator(...) call form
        if isinstance(dec, ast.Call):
            func = dec.func
            if isinstance(func, ast.Attribute) and func.attr in skip_names:
                return True
            if isinstance(func, ast.Name) and func.id in skip_names:
                return True
    return False


def check_debug_calls(tree: ast.Module, op_id: str) -> list[str]:
    """Check that public functions have logger.debug("GEMS ...") calls.

    Returns list of warning messages.
    """
    warnings = []

    # Decorator names that indicate Triton kernels / internal helpers
    SKIP_DECORATORS = {"triton.jit", "pointwise_dynamic", "jit", "autotune"}

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        # Skip private/helper functions
        if node.name.startswith("_") and node.name != f"_{op_id}":
            continue

        # Skip Triton kernels and decorated kernel functions
        if _has_skip_decorator(node, SKIP_DECORATORS):
            continue

        # Look for logger.debug call in function body
        has_gems_log = False
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func = child.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "debug"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "logger"
                ):
                    # Check if the message starts with "GEMS"
                    if child.args:
                        arg = child.args[0]
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            if arg.value.startswith("GEMS"):
                                has_gems_log = True
                                break

        if not has_gems_log:
            warnings.append(
                f"Function '{node.name}' at line {node.lineno} "
                f'missing logger.debug("GEMS ...") call'
            )

    return warnings


def main():
    parser = argparse.ArgumentParser(
        description="Check operator API logging convention"
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

    print(f"Checking API logging for {len(ops_to_check)} operator(s)...")
    all_warnings = []

    for op_id in sorted(ops_to_check):
        source_file = find_op_source(op_id)
        if source_file is None:
            # Not all operators have a direct source file (backend-specific, etc.)
            continue

        try:
            source = source_file.read_text()
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue

        # Check logging setup
        if not check_logging_setup(tree):
            all_warnings.append(
                f"{source_file}: operator '{op_id}' missing logging setup "
                f"(import logging + logger = logging.getLogger(__name__))"
            )
            continue

        # Check debug calls in public functions
        fn_warnings = check_debug_calls(tree, op_id)
        for w in fn_warnings:
            all_warnings.append(f"{source_file}: {w}")

    if all_warnings:
        print(f"\n⚠️  Found {len(all_warnings)} warning(s):\n")
        for w in all_warnings:
            print(f"::warning::{w}")
            print(f"  • {w}")
        # Exit 0: warnings don't block
        sys.exit(0)
    else:
        print("✅ All operators follow the API logging convention.")
        sys.exit(0)


if __name__ == "__main__":
    main()

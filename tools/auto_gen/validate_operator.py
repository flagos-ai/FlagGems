#!/usr/bin/env python3
"""
Validator for FlagGems operator implementation completeness.

Checks that a generated operator has:
- operators.yaml entries with KernelGen label
- test file with pytest marks
- benchmark file with pytest marks
"""

import argparse
import logging
import os
import re
import sys
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger(__name__)


def parse_variants_from_aten_ops(aten_ops: list[str]) -> dict[str, str]:
    """
    Parse operator variants from registered ATen ops.

    Args:
        aten_ops: List like ['sign', 'sign.out', 'sign_']

    Returns:
        Dict mapping variant type to operator ID:
        {
            'base': 'sign',
            'inplace': 'sign_',
            'out': 'sign_out'
        }
    """
    variants = {}

    for op in aten_ops:
        if op.endswith("_"):
            # In-place variant
            base = op[:-1]
            variants["inplace"] = op
            if "base" not in variants:
                variants["base"] = base
        elif ".out" in op:
            # Out variant
            base = op.replace(".out", "")
            variants["out"] = base + "_out"
            if "base" not in variants:
                variants["base"] = base
        else:
            # Base variant
            variants["base"] = op

    return variants


def infer_aten_ops_from_worktree(worktree_path: str, operator: str) -> list[str]:
    """
    Infer registered ATen ops by parsing src/flag_gems/__init__.py.

    Fallback when aten_ops_registered is missing from CC output.

    Returns:
        List of aten ops like ['sign', 'sign.out', 'sign_']
    """
    init_py = os.path.join(worktree_path, "src", "flag_gems", "__init__.py")
    if not os.path.exists(init_py):
        return []

    try:
        with open(init_py) as f:
            content = f.read()
    except Exception:
        return []

    aten_ops = []
    # Look for lines like: ("sign", sign), ("sign.out", sign_out), ("sign_", sign_)
    # Match operator name or operator_ or operator.out
    patterns = [
        rf'\("{re.escape(operator)}"',  # base: ("sign"
        rf'\("{re.escape(operator)}\.out"',  # out: ("sign.out"
        rf'\("{re.escape(operator)}_"',  # inplace: ("sign_"
    ]

    for pattern in patterns:
        matches = re.findall(pattern, content)
        if pattern == patterns[0] and matches:
            aten_ops.append(operator)
        elif pattern == patterns[1] and matches:
            aten_ops.append(f"{operator}.out")
        elif pattern == patterns[2] and matches:
            aten_ops.append(f"{operator}_")

    return aten_ops


def check_operators_yaml(
    worktree_path: str, variants: dict[str, str]
) -> dict[str, Any]:
    """
    Check if operator is registered in conf/operators.yaml with KernelGen label.

    Returns:
        {
            "valid": bool,
            "missing": list[str],
            "found": list[str]
        }
    """
    if yaml is None:
        return {
            "valid": False,
            "missing": ["pyyaml not installed"],
            "found": [],
        }

    yaml_path = os.path.join(worktree_path, "conf", "operators.yaml")
    if not os.path.exists(yaml_path):
        return {
            "valid": False,
            "missing": ["conf/operators.yaml not found"],
            "found": [],
        }

    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
    except Exception as e:
        return {
            "valid": False,
            "missing": [f"Failed to parse operators.yaml: {e}"],
            "found": [],
        }

    ops = data.get("ops", [])
    op_ids = {op.get("id"): op for op in ops}

    missing = []
    found = []

    for variant_type, op_id in variants.items():
        if op_id not in op_ids:
            missing.append(f"operators.yaml: {op_id} not found")
        else:
            op_entry = op_ids[op_id]
            labels = op_entry.get("labels", [])
            if "KernelGen" not in labels:
                missing.append(f"operators.yaml: {op_id} missing 'KernelGen' label")
            else:
                found.append(op_id)

    return {"valid": len(missing) == 0, "missing": missing, "found": found}


def check_pytest_marks(file_path: str, variants: dict[str, str]) -> dict[str, Any]:
    """
    Check if test/benchmark file has pytest.mark decorators for all variants.

    Returns:
        {
            "valid": bool,
            "missing": list[str],
            "found": list[str]
        }
    """
    if not os.path.exists(file_path):
        return {
            "valid": False,
            "missing": [f"{file_path} not found"],
            "found": [],
        }

    try:
        with open(file_path) as f:
            content = f.read()
    except Exception as e:
        return {
            "valid": False,
            "missing": [f"Failed to read {file_path}: {e}"],
            "found": [],
        }

    missing = []
    found = []

    for variant_type, op_id in variants.items():
        # Look for @pytest.mark.<op_id>
        mark_pattern = rf"@pytest\.mark\.{re.escape(op_id)}\b"
        if not re.search(mark_pattern, content):
            missing.append(f"{file_path}: missing @pytest.mark.{op_id}")
        else:
            found.append(op_id)

    return {"valid": len(missing) == 0, "missing": missing, "found": found}


def validate_operator(
    worktree_path: str, operator: str, aten_ops: list[str]
) -> dict[str, Any]:
    """
    Validate operator implementation completeness.

    Args:
        worktree_path: Path to worktree
        operator: Base operator name (e.g., 'sign')
        aten_ops: Registered ATen ops (e.g., ['sign', 'sign.out', 'sign_'])
                  If empty, will attempt to infer from worktree

    Returns:
        {
            "valid": bool,
            "missing": list[str],  # All missing items
            "checks": {
                "operators_yaml": {...},
                "test_file": {...},
                "benchmark_file": {...}
            }
        }
    """
    # Fallback: if aten_ops is empty, try to infer from worktree
    if not aten_ops:
        logger.debug(
            f"aten_ops_registered missing or empty, inferring from worktree for {operator}"
        )
        aten_ops = infer_aten_ops_from_worktree(worktree_path, operator)
        if not aten_ops:
            logger.warning(
                f"Cannot infer aten_ops for {operator}, validation will be incomplete"
            )
            return {
                "valid": False,
                "missing": [
                    f"Cannot determine registered ATen ops for {operator} "
                    f"(aten_ops_registered missing from CC output and inference failed)"
                ],
                "checks": {},
            }

    variants = parse_variants_from_aten_ops(aten_ops)
    logger.debug(f"Parsed variants for {operator}: {variants}")

    checks = {}
    all_missing = []

    # Check operators.yaml
    checks["operators_yaml"] = check_operators_yaml(worktree_path, variants)
    all_missing.extend(checks["operators_yaml"]["missing"])

    # Check test file
    test_file = os.path.join(worktree_path, "tests", f"test_{operator}.py")
    checks["test_file"] = check_pytest_marks(test_file, variants)
    all_missing.extend(checks["test_file"]["missing"])

    # Check benchmark file
    benchmark_file = os.path.join(worktree_path, "benchmark", f"test_{operator}.py")
    checks["benchmark_file"] = check_pytest_marks(benchmark_file, variants)
    all_missing.extend(checks["benchmark_file"]["missing"])

    return {
        "valid": len(all_missing) == 0,
        "missing": all_missing,
        "checks": checks,
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate FlagGems operator implementation completeness"
    )
    parser.add_argument("worktree_path", help="Path to worktree")
    parser.add_argument("operator", help="Base operator name (e.g., 'sign')")
    parser.add_argument(
        "aten_ops",
        nargs="*",
        help="Registered ATen ops (e.g., sign sign.out sign_). "
        "If omitted, will attempt to infer from worktree.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if yaml is None:
        logger.error("pyyaml is required. Install it with: pip install pyyaml")
        sys.exit(1)

    result = validate_operator(args.worktree_path, args.operator, args.aten_ops)

    if result["valid"]:
        logger.info(f"✅ Operator '{args.operator}' validation PASSED")
        sys.exit(0)
    else:
        logger.error(f"❌ Operator '{args.operator}' validation FAILED")
        logger.error(f"Missing items ({len(result['missing'])}):")
        for item in result["missing"]:
            logger.error(f"  - {item}")
        sys.exit(1)


if __name__ == "__main__":
    main()

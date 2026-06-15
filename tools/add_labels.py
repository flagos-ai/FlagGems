#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone script to add operator labels from operators.yaml into summary.json.

Usage:
    python add_labels.py <result_dir>

Where <result_dir> is the output directory produced by run_tests.py containing
summary.json. The script reads conf/operators.yaml relative to the project root,
builds a mapping of op_id -> labels, and injects the "labels" field into each
operator entry in summary.json.
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent


def pinfo(msg):
    print(f"\033[32m[INFO]\033[0m {msg}")


def perror(msg):
    print(f"\033[31m[ERROR]\033[0m {msg}")


def load_op_labels():
    """Load operator labels from conf/operators.yaml."""
    op_labels = {}
    op_inventory = ROOT / "conf" / "operators.yaml"
    try:
        with open(str(op_inventory), "r") as f:
            data = yaml.safe_load(f)
            catalog = data.get("ops", [])
    except Exception as e:
        perror(f"Failed to load operator inventory: {e}")
        return op_labels

    for op_entry in catalog:
        op_id = op_entry.get("id", "")
        labels = op_entry.get("labels", [])
        op_labels[op_id] = labels

    return op_labels


def main():
    parser = argparse.ArgumentParser(
        description="Add operator labels from operators.yaml into summary.json"
    )
    parser.add_argument(
        "result_dir",
        help="Path to the result directory containing summary.json",
    )
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    summary_path = result_dir / "summary.json"

    if not summary_path.exists():
        perror(f"summary.json not found in {result_dir}")
        sys.exit(1)

    # Load summary
    with summary_path.open("r") as f:
        summary = json.load(f)

    op_data = summary.get("result", {})
    if not op_data:
        perror("No 'result' field found in summary.json")
        sys.exit(1)

    # Load labels from operators.yaml
    op_labels = load_op_labels()
    if not op_labels:
        perror("No labels loaded from operators.yaml")
        sys.exit(1)

    # Inject labels into each operator entry
    updated = 0
    for op_id, op_entry in op_data.items():
        labels = op_labels.get(op_id, [])
        op_entry["labels"] = labels
        updated += 1

    # Write back
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    pinfo(f"Updated {updated} operators with labels in {summary_path}")


if __name__ == "__main__":
    main()

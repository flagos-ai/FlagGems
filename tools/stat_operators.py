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

"""
Analyze operators.yaml and report statistics for each operator, including:
- id, description, for, labels, kind, stages
- Current stage (last stage key)
- Whether it runs under run_tests.py default mode (stable)
- Whether it runs under run_tests.py --stages all mode

Usage:
    Requires: pyyaml (pip install pyyaml)
    python tools/stat_operators.py                      # print to terminal
    python tools/stat_operators.py --output stats.csv   # output to CSV
    python tools/stat_operators.py --stages stable      # filter by stage (default: stable)
    python tools/stat_operators.py --stages all         # report for all mode
    python tools/stat_operators.py --skip-only          # only show non-running ops in stable mode
    python tools/stat_operators.py --skip-only --output skip.csv  # output skip list to CSV
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import yaml


def load_operators(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("ops", [])


def get_current_stage(stages):
    """Get current stage of the operator (last entry's key in stages list)."""
    if not stages:
        return None
    last = stages[-1]
    if isinstance(last, dict):
        return next(iter(last.keys()), None)
    return None


def get_stage_version(stages):
    """Get the version corresponding to the current stage."""
    if not stages:
        return None
    last = stages[-1]
    if isinstance(last, dict):
        return next(iter(last.values()), None)
    return None


def get_all_stages(stages):
    """Get full stage history of the operator."""
    result = []
    for s in stages or []:
        if isinstance(s, dict):
            for k, v in s.items():
                result.append(f"{k}:{v}")
    return " -> ".join(result) if result else ""


def will_run(current_stage, effective_stages):
    """Check if the operator will run under given effective_stages."""
    if current_stage is None:
        return False
    return current_stage in effective_stages


def main():
    parser = argparse.ArgumentParser(
        description="Analyze operators.yaml and report statistics"
    )
    parser.add_argument(
        "--yaml",
        default=str(Path(__file__).parent.parent / "conf" / "operators.yaml"),
        help="Path to operators.yaml",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output file path (CSV format); prints to terminal if omitted",
    )
    parser.add_argument(
        "--stages",
        default="stable",
        help="Simulate run_tests.py --stages argument (default: stable)",
    )
    parser.add_argument(
        "--skip-only",
        action="store_true",
        default=False,
        help="Only show operators that will NOT run in default (stable) mode",
    )
    args = parser.parse_args()

    # Auto-append timestamp to output filename: stats.csv -> stats_20260709_173200.csv
    if args.output:
        p = Path(args.output)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = str(p.parent / f"{p.stem}_{timestamp}{p.suffix}")

    # Parse effective_stages
    effective_stages = []
    for s in args.stages.split(","):
        stage = s.strip()
        if stage == "all":
            effective_stages = ["alpha", "beta", "stable"]
            break
        if stage in ["alpha", "beta", "stable", "removed"]:
            effective_stages.append(stage)
    if not effective_stages:
        effective_stages = ["stable"]

    ops = load_operators(args.yaml)

    # Build statistics data
    rows = []
    for op in ops:
        op_id = op.get("id", "")
        description = op.get("description", "").strip().replace("\n", " ")
        for_list = ", ".join(op.get("for", []))
        labels = ", ".join(op.get("labels", []))
        kind = ", ".join(op.get("kind", []))
        stages = op.get("stages", [])
        current_stage = get_current_stage(stages)
        current_version = get_stage_version(stages)
        stage_history = get_all_stages(stages)
        runs_in_mode = will_run(current_stage, effective_stages)
        runs_in_stable = will_run(current_stage, ["stable"])
        runs_in_all = will_run(current_stage, ["alpha", "beta", "stable"])

        rows.append(
            {
                "id": op_id,
                "for": for_list,
                "labels": labels,
                "kind": kind,
                "current_stage": current_stage or "N/A",
                "current_version": current_version or "N/A",
                "stage_history": stage_history,
                "runs_default(stable)": "YES" if runs_in_stable else "NO",
                "runs_all": "YES" if runs_in_all else "NO",
                "runs_selected": "YES" if runs_in_mode else "NO",
                "description": description[:80],
            }
        )

    # Summary counts
    total = len(rows)
    stage_counts = {}
    for r in rows:
        s = r["current_stage"]
        stage_counts[s] = stage_counts.get(s, 0) + 1
    runs_stable_count = sum(1 for r in rows if r["runs_default(stable)"] == "YES")
    runs_all_count = sum(1 for r in rows if r["runs_all"] == "YES")

    # Field descriptions
    field_descriptions = {
        "id": "Unique operator identifier, used with run_tests.py --ops",
        "for": "Underlying torch operation names",
        "labels": (
            "Labels: aten=ATen native, NoCPU=no CPU ref, "
            "pointwise=elementwise, KernelGen=generated kernel"
        ),
        "kind": "Category: Math, Reduction, BLAS, Distribution, Memory",
        "current_stage": (
            "Current stage (last key in stages list): " "alpha/beta/stable/removed"
        ),
        "current_version": "Version when entering current stage",
        "stage_history": "Full stage progression, e.g. beta:5.0 -> stable:5.3",
        "runs_default(stable)": (
            "Whether this op runs in default mode (--stages stable)"
        ),
        "runs_all": "Whether this op runs in --stages all (excludes removed)",
        "runs_selected": f"Whether this op runs in current mode (--stages {args.stages})",
        "description": "Operator description (first 80 chars)",
    }

    # --skip-only mode: only output non-running operators
    if args.skip_only:
        not_running = [r for r in rows if r["runs_default(stable)"] == "NO"]

        # Output CSV if --output specified
        if args.output:
            fieldnames = [
                "#",
                "id",
                "current_stage",
                "current_version",
                "stage_history",
            ]
            with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for i, r in enumerate(not_running, 1):
                    writer.writerow(
                        {
                            "#": i,
                            "id": r["id"],
                            "current_stage": r["current_stage"],
                            "current_version": r["current_version"],
                            "stage_history": r["stage_history"],
                        }
                    )
            print(f"Exported {len(not_running)} non-running ops to: " f"{args.output}")
            return

        # Terminal output
        print("=" * 100)
        print("Operators that will NOT run in default mode (--stages stable)")
        print(
            "Rule: last key in operators.yaml stages list != 'stable' " "=> won't run"
        )
        print("=" * 100)
        print()
        header = (
            f"{'#':<4} {'ID':<45} {'Current Stage':<15} "
            f"{'Version':<10} {'Stage History'}"
        )
        print(header)
        print("-" * 120)
        for i, r in enumerate(not_running, 1):
            print(
                f"{i:<4} "
                f"{r['id']:<45} "
                f"{r['current_stage']:<15} "
                f"{r['current_version']:<10} "
                f"{r['stage_history']}"
            )
        print()
        print(
            f"Total {len(not_running)} ops won't run in stable mode "
            f"(total ops: {total})"
        )
        print()
        # Group by stage
        skip_by_stage = {}
        for r in not_running:
            s = r["current_stage"]
            skip_by_stage.setdefault(s, []).append(r["id"])
        print("Grouped by stage:")
        for stage, ids in sorted(skip_by_stage.items()):
            print(f"  {stage} ({len(ids)}):")
            for op_id in ids:
                print(f"    - {op_id}")
        return

    # Output
    if args.output:
        fieldnames = [
            "id",
            "for",
            "labels",
            "kind",
            "current_stage",
            "current_version",
            "stage_history",
            "runs_default(stable)",
            "runs_all",
            "runs_selected",
            "description",
        ]
        with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            empty_row = {k: "" for k in fieldnames}

            # Write summary info
            def write_info(text):
                r = empty_row.copy()
                r[fieldnames[0]] = text
                writer.writerow(r)

            write_info("=== Operator Statistics (operators.yaml) ===")
            write_info("")
            write_info("[Field Descriptions]")
            for field, desc in field_descriptions.items():
                write_info(f"  {field}: {desc}")
            write_info("")
            write_info("[Summary]")
            write_info(f"  Total operators: {total}")
            write_info("  Stage distribution:")
            for stage, count in sorted(stage_counts.items()):
                write_info(f"    {stage}: {count}")
            write_info(f"  Runnable in default (stable): {runs_stable_count}")
            write_info(f"  Runnable in all mode: {runs_all_count}")
            write_info("")

            # Write header and data
            writer.writeheader()
            writer.writerows(rows)
        print(f"Exported to: {args.output}")
    else:
        # Terminal table output
        print("=" * 100)
        print("Operator Statistics (operators.yaml)")
        print("=" * 100)
        print()
        print("Field descriptions:")
        for field, desc in field_descriptions.items():
            print(f"  {field:<25s} - {desc}")
        print()
        print(f"Total operators: {total}")
        print("Stage distribution:")
        for stage, count in sorted(stage_counts.items()):
            print(f"  {stage:10s}: {count}")
        print(f"Runnable in default (stable): {runs_stable_count}")
        print(f"Runnable in all mode: {runs_all_count}")
        print("=" * 100)
        print()

        # Print detail table
        header = (
            f"{'#':<5} {'ID':<45} {'Stage':<10} {'Version':<8} "
            f"{'Run(stable)':<12} {'Run(all)':<10} {'Labels'}"
        )
        print(header)
        print("-" * len(header))
        for i, r in enumerate(rows, 1):
            print(
                f"{i:<5} "
                f"{r['id']:<45} "
                f"{r['current_stage']:<10} "
                f"{r['current_version']:<8} "
                f"{r['runs_default(stable)']:<12} "
                f"{r['runs_all']:<10} "
                f"{r['labels']}"
            )

    # Print non-running operators for current --stages
    not_running = [r for r in rows if r["runs_selected"] == "NO"]
    stages_label = args.stages
    print(f"\n{'=' * 100}")
    print(
        f"Operators NOT running in current mode "
        f"(--stages {stages_label}): {len(not_running)}"
    )
    print("=" * 100)
    for i, r in enumerate(not_running, 1):
        print(
            f"  {i:<4} {r['id']:<45} "
            f"stage={r['current_stage']:<10} "
            f"version={r['current_version']:<8} "
            f"stages={r['stage_history']}"
        )
    print(
        f"\nTotal {len(not_running)} ops won't run in --stages "
        f"{stages_label} (total: {total})"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Competition Score Calculator

Parses benchmark logs (from benchmark/conftest.py record logger, line format: [INFO] {json}).
Uses each metric's latency_base as the baseline by default, no dependency on baseline_scores.json.

Scoring dimensions (max 100):
  functional_correctness  30  Functional correctness (based on correctness test pass rate)
  performance             20  Performance competitiveness (based on geometric mean of speedup)
  test_coverage           20  Test completeness (based on number of test cases)
  adaptability            10  Open-source adaptability (default 0, manual override)
  compatibility           10  Cross-platform compatibility (default 0, manual override)
  readability             10  Code readability (default 0, manual override)
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Dimension max score definitions
# ---------------------------------------------------------------------------
DIMENSION_MAX = {
    "functional_correctness": 30,
    "performance": 20,
    "test_coverage": 20,
    "adaptability": 10,
    "compatibility": 10,
    "readability": 10,
}

# ---------------------------------------------------------------------------
# Performance scoring parameters
# ---------------------------------------------------------------------------
PERF_SPEEDUP_FLOOR = 0.9  # speedup < this value → 0 points
PERF_SPEEDUP_CEIL = 1.5  # speedup >= this value → full score
FAILURE_PENALTY_SPEEDUP = 0.5

# ---------------------------------------------------------------------------
# Test coverage parameters
# ---------------------------------------------------------------------------
DEFAULT_EXPECTED_CASES = 10  # Default expected number of test cases


# ---------------------------------------------------------------------------
# Benchmark log parsing
# ---------------------------------------------------------------------------
def parse_benchmark_log(log_file: Path) -> List[Dict]:
    results: List[Dict] = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("[INFO]"):
                continue
            json_str = line[len("[INFO]") :].strip()
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line: {e}", file=sys.stderr)
                continue
            if "op_name" in data:
                results.append(data)
    return results


# ---------------------------------------------------------------------------
# Speedup calculation (equal-weight geometric mean across shapes)
# ---------------------------------------------------------------------------
def calculate_speedup(pr_result: Dict) -> Optional[float]:
    """Calculate speedup as the geometric mean of per-shape speedups.

    Each shape contributes equally to the result, preventing large shapes
    (with higher absolute latency) from dominating the average.
    """
    pr_metrics = pr_result.get("result", [])
    if not pr_metrics:
        return None

    shape_speedups: List[float] = []
    for m in pr_metrics:
        latency = m.get("latency")
        latency_base = m.get("latency_base")
        if latency is None or latency_base is None:
            continue
        latency = float(latency)
        latency_base = float(latency_base)
        if latency <= 0 or latency_base <= 0:
            continue
        shape_speedups.append(latency_base / latency)

    if not shape_speedups:
        return None

    return math.exp(sum(math.log(s) for s in shape_speedups) / len(shape_speedups))


# ---------------------------------------------------------------------------
# Performance summary (preserving original detail structure)
# ---------------------------------------------------------------------------
def calculate_normalized_score(benchmark_results: List[Dict]) -> Dict:
    speedups: List[float] = []
    details: List[Dict] = []

    for result in benchmark_results:
        op_name = result.get("op_name")
        dtype = result.get("dtype")
        speedup = calculate_speedup(result)

        if speedup is not None:
            speedups.append(speedup)
            details.append(
                {
                    "op_name": op_name,
                    "dtype": dtype,
                    "speedup": round(speedup, 4),
                    "num_shapes": len(result.get("result", [])),
                }
            )
        else:
            speedups.append(FAILURE_PENALTY_SPEEDUP)
            details.append(
                {
                    "op_name": op_name,
                    "dtype": dtype,
                    "speedup": FAILURE_PENALTY_SPEEDUP,
                    "penalized": True,
                    "error": "No valid latency/latency_base",
                }
            )

    num_penalized = sum(1 for d in details if d.get("penalized"))

    if speedups:
        geometric_mean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
    else:
        geometric_mean = 0.0

    return {
        "geometric_mean_speedup": round(geometric_mean, 4),
        "num_tests": len(speedups),
        "num_failed": num_penalized,
        "details": details,
    }


# ---------------------------------------------------------------------------
# 6-dimension scoring
# ---------------------------------------------------------------------------
def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def score_functional_correctness(passed: int, total: int) -> float:
    """30 × (passed / total)"""
    if total <= 0:
        return 0.0
    return round(DIMENSION_MAX["functional_correctness"] * (passed / total), 2)


def score_performance(geometric_mean_speedup: float) -> float:
    """Linear mapping: speedup 0.9->0, 1.5->20, <0.9->0, >1.5->20"""
    if geometric_mean_speedup < PERF_SPEEDUP_FLOOR:
        return 0.0
    ratio = (geometric_mean_speedup - PERF_SPEEDUP_FLOOR) / (
        PERF_SPEEDUP_CEIL - PERF_SPEEDUP_FLOOR
    )
    return round(DIMENSION_MAX["performance"] * _clamp(ratio, 0.0, 1.0), 2)


def score_test_coverage(total_cases: int, expected_cases: int) -> float:
    """20 × min(1.0, total_cases / expected_cases)"""
    if expected_cases <= 0:
        return 0.0
    ratio = min(1.0, total_cases / expected_cases)
    return round(DIMENSION_MAX["test_coverage"] * ratio, 2)


def calculate_dimension_scores(
    *,
    correctness_passed: int,
    correctness_total: int,
    geometric_mean_speedup: float,
    expected_cases: int = DEFAULT_EXPECTED_CASES,
    override_adaptability: Optional[float] = None,
    override_compatibility: Optional[float] = None,
    override_readability: Optional[float] = None,
) -> Dict[str, float]:
    scores = {
        "functional_correctness": score_functional_correctness(
            correctness_passed, correctness_total
        ),
        "performance": score_performance(geometric_mean_speedup),
        "test_coverage": score_test_coverage(correctness_total, expected_cases),
        "adaptability": _clamp(
            override_adaptability if override_adaptability is not None else 0.0,
            0.0,
            DIMENSION_MAX["adaptability"],
        ),
        "compatibility": _clamp(
            override_compatibility if override_compatibility is not None else 0.0,
            0.0,
            DIMENSION_MAX["compatibility"],
        ),
        "readability": _clamp(
            override_readability if override_readability is not None else 0.0,
            0.0,
            DIMENSION_MAX["readability"],
        ),
    }
    return scores


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calculate competition score from benchmark log"
    )
    parser.add_argument("--log", type=Path, default=None, help="Benchmark log file")
    parser.add_argument(
        "--output", type=Path, default=Path("score.json"), help="Output score JSON file"
    )
    parser.add_argument("--pr-id", type=str, default="unknown", help="PR ID")
    parser.add_argument("--pr-title", type=str, default="", help="PR title")
    parser.add_argument("--pr-author", type=str, default="", help="PR author")
    parser.add_argument("--commit-sha", type=str, default="", help="Commit SHA")
    parser.add_argument("--correctness-failed", action="store_true")

    # Correctness test statistics
    parser.add_argument(
        "--correctness-passed",
        type=int,
        default=0,
        help="Number of correctness tests passed",
    )
    parser.add_argument(
        "--correctness-total",
        type=int,
        default=0,
        help="Total number of correctness tests",
    )

    # Expected number of test cases for coverage
    parser.add_argument(
        "--expected-cases",
        type=int,
        default=DEFAULT_EXPECTED_CASES,
        help="Expected number of test cases for test_coverage scoring",
    )

    # Manual override dimensions
    parser.add_argument(
        "--override-adaptability",
        type=float,
        default=None,
        help="Manual override for adaptability score (0-10)",
    )
    parser.add_argument(
        "--override-compatibility",
        type=float,
        default=None,
        help="Manual override for compatibility score (0-10)",
    )
    parser.add_argument(
        "--override-readability",
        type=float,
        default=None,
        help="Manual override for readability score (0-10)",
    )

    args = parser.parse_args()

    if args.correctness_failed:
        dim_scores = calculate_dimension_scores(
            correctness_passed=0,
            correctness_total=max(args.correctness_total, 1),
            geometric_mean_speedup=0.0,
            expected_cases=args.expected_cases,
            override_adaptability=args.override_adaptability,
            override_compatibility=args.override_compatibility,
            override_readability=args.override_readability,
        )
        total_score = round(sum(dim_scores.values()), 2)

        output_data = {
            "version": "2.0",
            "pr_id": args.pr_id,
            "pr_title": args.pr_title,
            "pr_author": args.pr_author,
            "commit_sha": args.commit_sha,
            "correctness": {
                "passed": False,
                "total": args.correctness_total,
                "num_passed": 0,
                "reason": "Correctness tests failed - performance tests were not executed",
            },
            "performance": {
                "geometric_mean_speedup": 0.0,
                "num_tests": 0,
                "num_failed": 0,
                "details": [],
            },
            "score_details": dim_scores,
            "total_score": total_score,
            "status": "correctness_failed",
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        return 0

    if args.log is None:
        print(
            "Error: --log is required when --correctness-failed is not set",
            file=sys.stderr,
        )
        return 1

    benchmark_results = parse_benchmark_log(args.log)
    perf_data = calculate_normalized_score(benchmark_results)

    dim_scores = calculate_dimension_scores(
        correctness_passed=args.correctness_passed,
        correctness_total=args.correctness_total,
        geometric_mean_speedup=perf_data["geometric_mean_speedup"],
        expected_cases=args.expected_cases,
        override_adaptability=args.override_adaptability,
        override_compatibility=args.override_compatibility,
        override_readability=args.override_readability,
    )
    total_score = round(sum(dim_scores.values()), 2)

    output_data = {
        "version": "2.0",
        "pr_id": args.pr_id,
        "pr_title": args.pr_title,
        "pr_author": args.pr_author,
        "commit_sha": args.commit_sha,
        "correctness": {
            "passed": True,
            "total": args.correctness_total,
            "num_passed": args.correctness_passed,
            "reason": "All correctness tests passed",
        },
        "performance": perf_data,
        "score_details": dim_scores,
        "total_score": total_score,
        "status": "success" if perf_data["num_failed"] == 0 else "partial",
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

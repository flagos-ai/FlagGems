#!/usr/bin/env python3
"""
Competition Score Calculator

解析 benchmark log（来自 benchmark/conftest.py 的 record logger，行格式: [INFO] {json}）。
默认直接使用每条 metric 的 latency_base 作为 baseline，不依赖 baseline_scores.json。
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional


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


def _avg(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def calculate_speedup(pr_result: Dict) -> Optional[float]:
    pr_metrics = pr_result.get("result", [])
    if not pr_metrics:
        return None

    pr_latencies = [
        m.get("latency") for m in pr_metrics if m.get("latency") is not None
    ]
    pr_avg_latency = _avg([float(x) for x in pr_latencies if x is not None])
    if pr_avg_latency is None or pr_avg_latency <= 0:
        return None

    base_latencies = [
        m.get("latency_base") for m in pr_metrics if m.get("latency_base") is not None
    ]
    base_avg_latency = _avg([float(x) for x in base_latencies if x is not None])
    if base_avg_latency is None or base_avg_latency <= 0:
        return None

    return base_avg_latency / pr_avg_latency


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
            details.append(
                {
                    "op_name": op_name,
                    "dtype": dtype,
                    "speedup": None,
                    "error": "No valid latency/latency_base",
                }
            )

    if speedups:
        geometric_mean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        normalized_score = 100 * (geometric_mean - 1.0)
    else:
        geometric_mean = 0.0
        normalized_score = 0.0

    return {
        "total_score": round(normalized_score, 2),
        "geometric_mean_speedup": round(geometric_mean, 4),
        "num_tests": len(speedups),
        "num_failed": len(benchmark_results) - len(speedups),
        "details": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calculate competition score from benchmark log"
    )
    parser.add_argument("--log", type=Path, required=True, help="Benchmark log file")
    parser.add_argument(
        "--output", type=Path, default=Path("score.json"), help="Output score JSON file"
    )
    parser.add_argument("--pr-id", type=str, default="unknown", help="PR ID")
    parser.add_argument("--pr-title", type=str, default="", help="PR title")
    parser.add_argument("--pr-author", type=str, default="", help="PR author")
    parser.add_argument("--commit-sha", type=str, default="", help="Commit SHA")
    parser.add_argument("--correctness-failed", action="store_true")

    args = parser.parse_args()

    if args.correctness_failed:
        output_data = {
            "version": "1.0",
            "pr_id": args.pr_id,
            "pr_title": args.pr_title,
            "pr_author": args.pr_author,
            "commit_sha": args.commit_sha,
            "correctness": {
                "passed": False,
                "reason": "Correctness tests failed - performance tests were not executed",
            },
            "performance": {
                "total_score": 0.0,
                "geometric_mean_speedup": 0.0,
                "num_tests": 0,
                "num_failed": 0,
                "details": [],
            },
            "status": "correctness_failed",
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        return 0

    benchmark_results = parse_benchmark_log(args.log)
    score_data = calculate_normalized_score(benchmark_results)

    output_data = {
        "version": "1.0",
        "pr_id": args.pr_id,
        "pr_title": args.pr_title,
        "pr_author": args.pr_author,
        "commit_sha": args.commit_sha,
        "correctness": {"passed": True, "reason": "All correctness tests passed"},
        "performance": score_data,
        "status": "success" if score_data["num_failed"] == 0 else "partial",
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

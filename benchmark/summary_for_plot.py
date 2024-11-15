"""
Script for Generating Operation Benchmark Summary Results

This script processes benchmark log files to calculate the average speedup for each
operation, categorized by data type. The summary provides an organized view of performance
gains, making it easier to analyze benchmark results by each tested data type.

Usage:
    Pre-Step:
    Collect benchmark results by running a command similar to the following:

        pytest test_blas_perf.py --level core --record log

    **Note**: The command above is an example. It runs benchmark tests on a subset of files.
    You may need to modify it based on the files or parameters you want to test. Be sure to
    include the `--record log` option, as it is required to generate the benchmark log file.

    The example command above will generate a log file named `result_test_blas_perf--level_core--record_log.log`
    in the benchmark directory.

    Step 1:
    Run this script with the generated log file as an argument:

        python summary_for_plot.py result_test_blas_perf--level_core--record_log.log

Options:
    -h, --help            Show this help message and exit.
    log_file_path         Path to the benchmark log file to be processed.
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import List

from attri_util import BenchmarkMetrics, BenchmarkResult


@dataclass
class SummaryResultOverDtype:
    op_name: str = ""
    float16_speedup: float = 0.0
    float32_speedup: float = 0.0
    bfloat16_speedup: float = 0.0
    int16_speedup: float = 0.0
    int32_speedup: float = 0.0
    bool_speedup: float = 0.0
    cfloat_speedup: float = 0.0
    all_tests_passed: bool = False

    def __str__(self) -> str:
        all_shapes_status = "yes" if self.all_tests_passed else "no"
        return (
            f"{self.op_name:<30} "
            f"{self.float16_speedup:<20.6f} "
            f"{self.float32_speedup:<20.6f} "
            f"{self.bfloat16_speedup:<20.6f} "
            f"{self.int16_speedup:<20.6f} "
            f"{self.int32_speedup:<20.6f} "
            f"{self.bool_speedup:<20.6f} "
            f"{self.cfloat_speedup:<20.6f}"
            f"{all_shapes_status:<20}"
        )


def parse_log(log_file_path: str) -> List[BenchmarkResult]:
    with open(log_file_path, "r") as file:
        log_lines = [
            line
            for line in file.read().strip().split("\n")
            if line.startswith("[INFO]")
        ]

    benchmark_results = []
    for line in log_lines:
        if line.startswith("[INFO]"):
            json_str = line[len("[INFO] ") :]
            data = json.loads(json_str)
            benchmark_result = BenchmarkResult(
                op_name=data["op_name"],
                dtype=data["dtype"],
                mode=data["mode"],
                level=data["level"],
                result=[
                    BenchmarkMetrics(
                        legacy_shape=metric.get("legacy_shape"),
                        shape_detail=metric.get("shape_detail", []),
                        latency_base=metric.get("latency_base"),
                        latency=metric.get("latency"),
                        speedup=metric.get("speedup"),
                        accuracy=metric.get("accuracy"),
                        tflops=metric.get("tflops"),
                        utilization=metric.get("utilization"),
                        error_msg=metric.get("error_msg"),
                    )
                    for metric in data["result"]
                ],
            )

            benchmark_results.append(benchmark_result)

    return benchmark_results


def calculate_avg_speedup_over_dtype(metrics):
    speedups = [
        metric.speedup
        for metric in metrics
        if metric.speedup is not None and metric.error_msg is None
    ]
    return sum(speedups) / len(speedups) if speedups else 0.0


def all_benchshape_passed(metrics):
    return all(metric.error_msg is None for metric in metrics)


def summary_for_plot(benchmark_results):
    summary = defaultdict(SummaryResultOverDtype)

    dtype_mapping = {
        "torch.float16": "float16_speedup",
        "torch.float32": "float32_speedup",
        "torch.bfloat16": "bfloat16_speedup",
        "torch.int16": "int16_speedup",
        "torch.int32": "int32_speedup",
        "torch.bool": "bool_speedup",
        "torch.complex64": "cfloat_speedup",
    }

    for item in benchmark_results:
        dtype_suffix = ""
        if item.dtype in ["torch.float16", "torch.float32", "torch.bfloat16"]:
            dtype_suffix = ""  # No suffix for float types
        else:
            dtype_suffix = (
                "_complex"
                if "complex64" in item.dtype
                else "_int"
                if "int" in item.dtype
                else "_bool"
            )

        op_name = item.op_name + dtype_suffix
        avg_speedup = calculate_avg_speedup_over_dtype(item.result)
        cur_op_summary = summary[op_name]
        cur_op_summary.op_name = op_name
        cur_op_summary.all_tests_passed = all_benchshape_passed(item.result)
        setattr(
            summary[op_name],
            dtype_mapping.get(item.dtype, "float16_speedup"),
            avg_speedup,
        )

    # sort the keys based on `op_name`
    sorted_summary = sorted(summary.values(), key=lambda x: x.op_name)

    header = (
        f"{'op_name':<30} "
        f"{'float16_speedup':<20} "
        f"{'float32_speedup':<20} "
        f"{'bfloat16_speedup':<20} "
        f"{'int16_speedup':<20} "
        f"{'int32_speedup':<20} "
        f"{'bool_speedup':<20} "
        f"{'cfloat_speedup':<20}"
        f"{'all_tests_passed':<20}"
    )

    print(header)
    for result in sorted_summary:
        print(result)

    return summary


def main(log_file_path):
    result = parse_log(log_file_path)
    summary_for_plot(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse benchmark log file.")
    parser.add_argument("log_file_path", type=str, help="Path to the log file.")
    args = parser.parse_args()

    main(args.log_file_path)

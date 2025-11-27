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
from typing import Any, Dict, List

from attri_util import BenchmarkMetrics, BenchmarkResult

# to enable log files crossing speedup calculation
ENABLE_COMPARE = False


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

    # to calculate the speedup across log files.
    compared_float16_speedup: float = 0.0
    compared_float32_speedup: float = 0.0
    compared_bfloat16_speedup: float = 0.0
    compared_int16_speedup: float = 0.0
    compared_int32_speedup: float = 0.0
    compared_bool_speedup: float = 0.0
    compared_cfloat_speedup: float = 0.0
    all_tests_passed: bool = False

    def __str__(self) -> str:
        all_shapes_status = "yes" if self.all_tests_passed else "no"
        return (
            (
                f"{self.op_name:<30} "
                f"{self.float16_speedup:<20.6f} "
                f"{self.float32_speedup:<20.6f} "
                f"{self.bfloat16_speedup:<20.6f} "
                f"{self.int16_speedup:<20.6f} "
                f"{self.int32_speedup:<20.6f} "
                f"{self.bool_speedup:<20.6f} "
                f"{self.cfloat_speedup:<20.6f}"
                f"{self.compared_float16_speedup:<20.6f}"
                f"{self.compared_float32_speedup:<20.6f}"
                f"{self.compared_bfloat16_speedup:<20.6f}"
                f"{self.compared_int16_speedup:<20.6f}"
                f"{self.compared_int32_speedup:<20.6f}"
                f"{self.compared_bool_speedup:<20.6f}"
                f"{self.compared_cfloat_speedup:<20.6f}"
                f"{all_shapes_status:<20}"
            )
            if ENABLE_COMPARE
            else (
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
        )


def parse_log(log_file_path: str) -> List[BenchmarkResult]:
    with open(log_file_path, "r") as file:
        log_lines = [
            line
            for line in file.read().strip().split("\n")
            if line.startswith("[INFO] {")
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


def get_key_by_op_dtype_shape(op_name, dtype, shape):
    return hex(hash((hash(op_name), hash(dtype), hash(shape))))


def parse_log_to_dict(log_file_path: str) -> Dict[int, Any]:
    with open(log_file_path, "r") as file:
        log_lines = [
            line
            for line in file.read().strip().split("\n")
            if line.startswith("[INFO]")
        ]

    # dict(op_name, dict(dtype, dict(shape, latency))
    benchmark_results = dict()
    for line in log_lines:
        if line.startswith("[INFO]"):
            json_str = line[len("[INFO] ") :]
            data = json.loads(json_str)
            op_name = (data["op_name"],)
            dtype = (data["dtype"],)
            mode = (data["mode"],)
            level = (data["level"],)
            benchmark_result = BenchmarkResult(
                op_name,
                dtype,
                mode,
                level,
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
            for result in benchmark_result.result:
                key = get_key_by_op_dtype_shape(
                    op_name[0], dtype[0], str(result.shape_detail)
                )
                benchmark_results[key] = result.latency
    return benchmark_results


def calculate_avg_speedup_over_dtype(metrics):
    speedups = [
        metric.speedup
        for metric in metrics
        if metric.speedup is not None and metric.error_msg is None
    ]
    return sum(speedups) / len(speedups) if speedups else 0.0


def calculate_avg_compared_speedup_over_dtype(metrics):
    compared_speedups = [
        metric.compared_speedup
        for metric in metrics
        if metric.compared_speedup is not None and metric.error_msg is None
    ]
    return sum(compared_speedups) / len(compared_speedups) if compared_speedups else 0.0


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
        op_name = item.op_name
        avg_speedup = calculate_avg_speedup_over_dtype(item.result)
        avg_compared_speedup = calculate_avg_compared_speedup_over_dtype(item.result)
        cur_op_summary = summary[op_name]
        cur_op_summary.op_name = op_name
        cur_op_summary.all_tests_passed = all_benchshape_passed(item.result)
        setattr(
            summary[op_name],
            dtype_mapping.get(item.dtype, "float16_speedup"),
            avg_speedup,
        )
        if ENABLE_COMPARE:
            setattr(
                summary[op_name],
                "compared_" + dtype_mapping.get(item.dtype, "float16_speedup"),
                avg_compared_speedup,
            )

    # sort the keys based on `op_name`
    sorted_summary = sorted(summary.values(), key=lambda x: x.op_name)

    header = (
        (
            f"{'op_name':<30} "
            f"{'float16_speedup':<20} "
            f"{'float32_speedup':<20} "
            f"{'bfloat16_speedup':<20} "
            f"{'int16_speedup':<20} "
            f"{'int32_speedup':<20} "
            f"{'bool_speedup':<20} "
            f"{'cfloat_speedup':<20}"
            f"{'comp_fp16_speedup':<20}"
            f"{'comp_fp32_speedup':<20}"
            f"{'comp_bf16_speedup':<20}"
            f"{'comp_int16_speedup':<20}"
            f"{'comp_int32_speedup':<20}"
            f"{'comp_bool_speedup':<20}"
            f"{'comp_cfloat_speedup':<20}"
            f"{'all_tests_passed':<20}"
        )
        if ENABLE_COMPARE
        else (
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
    )

    print(header)
    for result in sorted_summary:
        print(result)

    return summary


def compare_main(log_file_a, log_file_b):
    result_a = parse_log(log_file_a)
    result_b = parse_log_to_dict(log_file_b)
    for result in result_a:
        for sub_result in result.result:
            key = get_key_by_op_dtype_shape(
                result.op_name, result.dtype, str(sub_result.shape_detail)
            )
            sub_result.compared_speedup = result_b.get(key, 0) / sub_result.latency

    summary_for_plot(result_a)


def main(log_file_path):
    result = parse_log(log_file_path)
    summary_for_plot(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse benchmark log file.")
    parser.add_argument("log_file_path", type=str, help="Path to the log file.")
    parser.add_argument(
        "--compare",
        "-c",
        type=str,
        default="",
        help="Path to a log file with baseline data to get speedup statistics across 2 log files",
    )
    args = parser.parse_args()

    if not args.compare == "":
        ENABLE_COMPARE = True
        compare_main(args.log_file_path, args.compare)
    else:
        main(args.log_file_path)

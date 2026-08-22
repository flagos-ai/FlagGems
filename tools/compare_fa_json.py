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

"""Compare FA2 and FA3 benchmark JSON files and write a CSV report.

Performance is calculated as ``FA2 latency / FA3 latency``.  Therefore, a
value greater than 1 means FA3 is faster.  When a vLLM benchmark JSON exists,
the report also includes vLLM latency and ``vLLM latency / FA latency`` ratios;
values greater than 1 in those ratio columns mean FA2/FA3 is faster than vLLM.

The preferred CLI uses repeatable ``--fa2-json``, ``--fa3-json``, and
``--vllm-json`` options.  Multi-file inputs use ``SCENARIO=PATH`` so repeated
case names from different workload scenarios remain distinct.  The historical
two positional paths remain supported so existing commands keep working.

Rows are filtered by the number of backends that contain a latency for the
same ``(scenario, benchmark, case, dtype)`` key.  The default requires two
backends; ``--filter 1`` selects the union and ``--filter 3`` selects the
three-way intersection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

DTYPE_ORDER = ("bfloat16", "float16")
MIN_BACKEND_CHOICES = (1, 2, 3)
DEFAULT_SCENARIO = "default"
ResultKey = Tuple[str, str, str, str]
CaseKey = Tuple[str, str, str]
SCENARIO_FIELDNAMES = ("scenario",)
BASE_FIELDNAMES = (
    "benchmark",
    "case",
    "dtype",
    "fa2_latency_ms",
    "fa3_latency_ms",
    "performance",
)
VLLM_FIELDNAMES = (
    "vllm_latency_ms",
    "fa2_vs_vllm",
    "fa3_vs_vllm",
)


class Result(NamedTuple):
    latency: float


class BenchmarkInput(NamedTuple):
    scenario: str
    path: Path


def _parse_benchmark_input(value: str) -> BenchmarkInput:
    scenario, separator, raw_path = value.partition("=")
    if not separator:
        return BenchmarkInput(DEFAULT_SCENARIO, Path(value))
    if not scenario or not raw_path:
        raise argparse.ArgumentTypeError(
            "benchmark JSON must be PATH or non-empty SCENARIO=PATH"
        )
    return BenchmarkInput(scenario, Path(raw_path))


def _normalize_dtype(dtype: str) -> str:
    return dtype.removeprefix("torch.")


def _format_latency(result: Optional[Result]) -> str:
    return "" if result is None else f"{result.latency:.9f}"


def _format_ratio(numerator: Optional[Result], denominator: Optional[Result]) -> str:
    if numerator is None or denominator is None:
        return ""
    return f"{numerator.latency / denominator.latency:.3f}"


def _load_results(
    path: Path,
    *,
    scenario: str = DEFAULT_SCENARIO,
    expected_fa_version: Optional[int] = None,
    backend_name: str = "benchmark",
) -> Tuple[Dict[ResultKey, Result], List[CaseKey]]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)

    results: Dict[ResultKey, Result] = {}
    case_order: List[CaseKey] = []
    seen_cases = set()
    seen_result_keys = set()

    for benchmark_name, benchmark_data in data.items():
        for detail in benchmark_data.get("details", []):
            dtype = _normalize_dtype(detail["dtype"])
            for result in detail.get("result", []):
                shape_detail = result.get("shape_detail") or {}
                case_name = shape_detail.get("name")
                if not case_name:
                    raise ValueError(
                        f"{path}: benchmark result is missing shape_detail.name"
                    )
                if expected_fa_version is not None:
                    fa_version = shape_detail.get("fa_version")
                    if fa_version != expected_fa_version:
                        raise ValueError(
                            f"{path}: expected {backend_name} results with "
                            f"fa_version={expected_fa_version}, but {case_name} reports "
                            f"fa_version={fa_version!r}"
                        )

                key = (scenario, benchmark_name, case_name, dtype)
                if key in seen_result_keys:
                    raise ValueError(f"{path}: duplicate benchmark result: {key}")
                seen_result_keys.add(key)

                case_key = (scenario, benchmark_name, case_name)
                if case_key not in seen_cases:
                    seen_cases.add(case_key)
                    case_order.append(case_key)

                latency = result.get("latency")
                if latency is None:
                    continue
                if (
                    isinstance(latency, bool)
                    or not isinstance(latency, (int, float))
                    or not math.isfinite(float(latency))
                    or latency <= 0
                ):
                    raise ValueError(
                        f"{path}: invalid latency for "
                        f"{benchmark_name}/{case_name}/{dtype}: {latency!r}"
                    )

                results[key] = Result(float(latency))

    return results, case_order


def _normalize_benchmark_inputs(value) -> List[BenchmarkInput]:
    if value is None:
        return []
    if isinstance(value, BenchmarkInput):
        return [value]
    if isinstance(value, (str, Path)):
        return [BenchmarkInput(DEFAULT_SCENARIO, Path(value))]

    inputs = list(value)
    if not all(isinstance(item, BenchmarkInput) for item in inputs):
        raise TypeError(
            "multiple benchmark JSON inputs must use BenchmarkInput(scenario, path)"
        )
    return inputs


def _load_result_files(
    value,
    *,
    expected_fa_version: Optional[int] = None,
    backend_name: str = "benchmark",
    ignore_missing: bool = False,
) -> Tuple[Dict[ResultKey, Result], List[CaseKey], List[BenchmarkInput]]:
    merged_results: Dict[ResultKey, Result] = {}
    merged_order: List[CaseKey] = []
    loaded_inputs: List[BenchmarkInput] = []

    for benchmark_input in _normalize_benchmark_inputs(value):
        if not benchmark_input.path.is_file():
            if ignore_missing:
                continue
            raise FileNotFoundError(
                f"Benchmark JSON '{benchmark_input.path}' does not exist."
            )
        results, case_order = _load_results(
            benchmark_input.path,
            scenario=benchmark_input.scenario,
            expected_fa_version=expected_fa_version,
            backend_name=backend_name,
        )
        duplicates = sorted(set(merged_results).intersection(results))
        if duplicates:
            raise ValueError(
                f"{benchmark_input.path}: duplicate scenario benchmark results: "
                f"{duplicates}"
            )
        merged_results.update(results)
        merged_order.extend(case_order)
        loaded_inputs.append(benchmark_input)

    return merged_results, merged_order, loaded_inputs


def _merge_case_orders(*case_orders: Iterable[CaseKey]) -> List[CaseKey]:
    scenario_order: List[str] = []
    grouped: Dict[str, List[CaseKey]] = {}
    seen = set()
    for case_order in case_orders:
        for case_key in case_order:
            scenario = case_key[0]
            if scenario not in grouped:
                grouped[scenario] = []
                scenario_order.append(scenario)
            if case_key not in seen:
                seen.add(case_key)
                grouped[scenario].append(case_key)
    return [case_key for scenario in scenario_order for case_key in grouped[scenario]]


def _comparison_rows(
    fa2_results: Dict[ResultKey, Result],
    fa3_results: Dict[ResultKey, Result],
    case_order: Iterable[CaseKey],
    vllm_results: Optional[Dict[ResultKey, Result]] = None,
    *,
    min_backends: int = 2,
    include_scenario: bool = False,
) -> Iterable[Dict[str, str]]:
    if min_backends not in MIN_BACKEND_CHOICES:
        raise ValueError(
            f"min_backends must be one of {MIN_BACKEND_CHOICES}, "
            f"got {min_backends!r}"
        )

    backend_results = [fa2_results, fa3_results]
    if vllm_results is not None:
        backend_results.append(vllm_results)
    all_keys = set().union(*(set(results) for results in backend_results))
    selected_keys = {
        key
        for key in all_keys
        if sum(key in results for results in backend_results) >= min_backends
    }

    case_order = list(case_order)
    available_dtypes = {dtype for _, _, _, dtype in selected_keys}
    ordered_dtypes = [dtype for dtype in DTYPE_ORDER if dtype in available_dtypes]
    ordered_dtypes.extend(sorted(available_dtypes - set(DTYPE_ORDER)))

    # Keep dtype as the outer loop so every bfloat16 result appears before any
    # float16 result, while retaining the benchmark/case order from the JSON.
    for dtype in ordered_dtypes:
        for scenario, benchmark_name, case_name in case_order:
            key = (scenario, benchmark_name, case_name, dtype)
            if key not in selected_keys:
                continue
            fa2_result = fa2_results.get(key)
            fa3_result = fa3_results.get(key)
            vllm_result = vllm_results.get(key) if vllm_results is not None else None
            row = {}
            if include_scenario:
                row["scenario"] = scenario
            row.update(
                {
                    "benchmark": benchmark_name,
                    "case": case_name,
                    "dtype": dtype,
                    "fa2_latency_ms": _format_latency(fa2_result),
                    "fa3_latency_ms": _format_latency(fa3_result),
                    "performance": _format_ratio(fa2_result, fa3_result),
                }
            )
            if vllm_results is not None:
                row.update(
                    {
                        "vllm_latency_ms": _format_latency(vllm_result),
                        "fa2_vs_vllm": _format_ratio(vllm_result, fa2_result),
                        "fa3_vs_vllm": _format_ratio(vllm_result, fa3_result),
                    }
                )
            yield row


def compare_json(
    fa2_path,
    fa3_path,
    output_path: Path,
    vllm_path=None,
    *,
    min_backends: int = 2,
) -> None:
    fa2_results, fa2_case_order, fa2_inputs = _load_result_files(
        fa2_path,
        expected_fa_version=2,
        backend_name="FA2",
    )
    fa3_results, fa3_case_order, fa3_inputs = _load_result_files(
        fa3_path,
        expected_fa_version=3,
        backend_name="FA3",
    )
    vllm_results, vllm_case_order, vllm_inputs = _load_result_files(
        vllm_path,
        expected_fa_version=3,
        backend_name="vLLM",
        ignore_missing=False,
    )
    loaded_inputs = fa2_inputs + fa3_inputs + vllm_inputs
    scenario_names = {benchmark_input.scenario for benchmark_input in loaded_inputs}
    include_scenario = len(scenario_names) > 1 or scenario_names != {DEFAULT_SCENARIO}
    vllm_results_or_none = vllm_results if vllm_inputs else None
    case_order = _merge_case_orders(fa2_case_order, fa3_case_order, vllm_case_order)
    rows = list(
        _comparison_rows(
            fa2_results,
            fa3_results,
            case_order,
            vllm_results_or_none,
            min_backends=min_backends,
            include_scenario=include_scenario,
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(SCENARIO_FIELDNAMES if include_scenario else ())
            + BASE_FIELDNAMES
            + (VLLM_FIELDNAMES if vllm_results_or_none is not None else ()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare FA2/FA3 benchmark JSON files. Performance is "
            "FA2 latency / FA3 latency; values above 1 mean FA3 is faster."
        )
    )
    parser.add_argument(
        "legacy_fa2_json",
        nargs="?",
        type=Path,
        metavar="FA2_JSON",
        help="legacy positional FA2 benchmark JSON (prefer --fa2-json)",
    )
    parser.add_argument(
        "legacy_fa3_json",
        nargs="?",
        type=Path,
        metavar="FA3_JSON",
        help="legacy positional FA3 benchmark JSON (prefer --fa3-json)",
    )
    parser.add_argument(
        "--fa2-json",
        action="append",
        type=_parse_benchmark_input,
        metavar="[SCENARIO=]PATH",
        help=(
            "FA2 benchmark JSON; repeat with matching SCENARIO=PATH labels "
            "to aggregate multiple files"
        ),
    )
    parser.add_argument(
        "--fa3-json",
        action="append",
        type=_parse_benchmark_input,
        metavar="[SCENARIO=]PATH",
        help=(
            "FA3 benchmark JSON; repeat with matching SCENARIO=PATH labels "
            "to aggregate multiple files"
        ),
    )
    parser.add_argument(
        "--vllm-json",
        action="append",
        type=_parse_benchmark_input,
        metavar="[SCENARIO=]PATH",
        help=(
            "optional vLLM benchmark JSON; repeat with matching "
            "SCENARIO=PATH labels to aggregate multiple files. vLLM columns "
            "are added when this option is provided; every path must exist"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("fa2_vs_fa3.csv"),
        help="output CSV path (default: fa2_vs_fa3.csv)",
    )
    parser.add_argument(
        "--filter",
        "--min-backends",
        dest="min_backends",
        type=int,
        choices=MIN_BACKEND_CHOICES,
        default=2,
        help=(
            "minimum backends with latency for each "
            "scenario/benchmark/case/dtype: 1=union, 2=at least two "
            "(default), 3=all three"
        ),
    )
    args = parser.parse_args(argv)

    uses_legacy_paths = (
        args.legacy_fa2_json is not None or args.legacy_fa3_json is not None
    )
    uses_named_paths = args.fa2_json is not None or args.fa3_json is not None
    if uses_legacy_paths and uses_named_paths:
        parser.error("do not mix positional FA2/FA3 paths with --fa2-json/--fa3-json")

    if uses_named_paths:
        missing = [
            option
            for option, value in (
                ("--fa2-json", args.fa2_json),
                ("--fa3-json", args.fa3_json),
            )
            if value is None
        ]
    else:
        missing = [
            label
            for label, value in (
                ("FA2_JSON", args.legacy_fa2_json),
                ("FA3_JSON", args.legacy_fa3_json),
            )
            if value is None
        ]
        args.fa2_json = args.legacy_fa2_json
        args.fa3_json = args.legacy_fa3_json

    if missing:
        parser.error(f"missing required benchmark path(s): {', '.join(missing)}")

    def finalize_named_inputs(option: str, inputs):
        if inputs is None:
            return None
        if len(inputs) == 1 and inputs[0].scenario == DEFAULT_SCENARIO:
            return inputs[0].path
        if any(item.scenario == DEFAULT_SCENARIO for item in inputs):
            parser.error(
                f"when {option} is repeated, every value must use " "SCENARIO=PATH"
            )
        return inputs

    if uses_named_paths:
        args.fa2_json = finalize_named_inputs("--fa2-json", args.fa2_json)
        args.fa3_json = finalize_named_inputs("--fa3-json", args.fa3_json)
    if args.vllm_json is not None:
        args.vllm_json = finalize_named_inputs("--vllm-json", args.vllm_json)

    del args.legacy_fa2_json
    del args.legacy_fa3_json
    return args


def main() -> None:
    args = _parse_args()
    compare_json(
        args.fa2_json,
        args.fa3_json,
        args.output,
        args.vllm_json,
        min_backends=args.min_backends,
    )
    print(f"Wrote comparison CSV to {args.output}")


if __name__ == "__main__":
    main()

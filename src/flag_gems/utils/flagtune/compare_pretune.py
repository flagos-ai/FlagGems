#!/usr/bin/env python3
"""Compare two Pretune CSV artifacts and emit FlagTune Schema v2 results.

Both Schema v1 and v2 Pretune CSV inputs are accepted. Rows are joined by the
original input row index and validated by named workload dimensions, so repeated
shapes remain distinct. The output CSV is accompanied by a nested JSONL sibling.
Derived metrics use full-precision inputs and are rounded only for serialization.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[4]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from flag_gems.utils.flagtune.output_schema import (
    SCHEMA_VERSION,
    format_derived,
    format_ms,
    rounded_derived,
    rounded_ms,
)

COPY_COLUMNS_PREFIX = (
    "input_row_index",
    "op_id",
    "op_name",
    "variant",
)
COPY_COLUMNS_SUFFIX = (
    "Count",
    "input_dtypes",
    "output_dtypes",
    "model_dtype_key",
)
POLICY_COLUMNS = (
    "status",
    "tuning_cache_hit",
    "config_count",
    "timing_count",
    "cached_count",
    "measured_count",
    "best_config",
    "error",
)
V1_ALIASES = {
    "source_index": "input_row_index",
    "dtype_key": "model_dtype_key",
    "gpu_key": "model_gpu_key",
    "worker_id": "worker_index",
    "cache_hit": "tuning_cache_hit",
    "candidate_config_count": "config_count",
    "timed_config_count": "timing_count",
    "benchmark_cache_hit_count": "cached_count",
    "benchmark_success_count": "measured_count",
}


class ComparisonError(RuntimeError):
    """Report an invalid input artifact or incompatible comparison request."""


def _dimension_columns(fieldnames: Sequence[str], label: str) -> tuple[str, ...]:
    """Infer ordered workload dimensions from a Pretune CSV header."""
    try:
        count_index = fieldnames.index("Count")
        if "input_row_index" in fieldnames:
            start_index = fieldnames.index("variant") + 1
        elif "shape_key" in fieldnames:
            start_index = fieldnames.index("shape_key") + 1
        else:
            start_index = fieldnames.index("variant") + 1
    except ValueError as exc:
        raise ComparisonError(
            f"{label} CSV cannot locate workload dimensions between "
            "variant/shape_key and Count"
        ) from exc
    dimensions = tuple(fieldnames[start_index:count_index])
    if not dimensions:
        raise ComparisonError(f"{label} CSV has no workload dimension columns")
    if len(set(dimensions)) != len(dimensions):
        raise ComparisonError(
            f"{label} CSV has duplicate workload dimension columns"
        )
    return dimensions


def _copy_columns(dimensions: Sequence[str]) -> tuple[str, ...]:
    """Return stable copied columns for one operator-defined workload schema."""
    return (*COPY_COLUMNS_PREFIX, *dimensions, *COPY_COLUMNS_SUFFIX)


def build_parser() -> argparse.ArgumentParser:
    """Build the public comparison CLI parser."""
    parser = argparse.ArgumentParser(
        description="Compare baseline and ours Pretune CSV rows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baseline", required=True, help="Baseline pretune.csv.")
    parser.add_argument("--ours", required=True, help="Ours pretune.csv.")
    parser.add_argument("--output", required=True, help="Comparison CSV path.")
    parser.add_argument(
        "--tuning-column",
        default="tuning_time_ms",
        help="Input column containing per-shape tuning time in milliseconds.",
    )
    parser.add_argument(
        "--latency-column",
        default="latency_p50_ms",
        help="Input column containing per-shape operator latency in milliseconds.",
    )
    return parser


def _read_rows(path: Path, label: str) -> tuple[list[dict[str, str]], list[str]]:
    """Read a non-empty CSV table."""
    if not path.is_file():
        raise ComparisonError(f"{label} CSV does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ComparisonError(f"{label} CSV has no header: {path}")
            rows = list(reader)
            fieldnames = list(reader.fieldnames)
    except (OSError, csv.Error) as exc:
        raise ComparisonError(f"cannot read {label} CSV {path}: {exc}") from exc
    if not rows:
        raise ComparisonError(f"{label} CSV contains no rows: {path}")
    return rows, fieldnames


def _normalize_schema(
    rows: Sequence[Mapping[str, str]], fieldnames: Sequence[str]
) -> tuple[list[dict[str, str]], list[str]]:
    """Map a v1 or v2 Pretune CSV to canonical v2 input names."""
    names = set(fieldnames)
    is_v2 = "input_row_index" in names
    if not is_v2 and "source_index" not in names:
        raise ComparisonError(
            "Pretune CSV is missing input_row_index/source_index"
        )
    normalized = []
    for source in rows:
        row = dict(source)
        if not is_v2:
            for old, new in V1_ALIASES.items():
                if new not in row and old in row:
                    row[new] = row[old]
            row["schema_version"] = str(SCHEMA_VERSION)
        normalized.append(row)
    normalized_fields = list(fieldnames)
    for new in V1_ALIASES.values():
        if normalized and new in normalized[0] and new not in normalized_fields:
            normalized_fields.append(new)
    return normalized, normalized_fields


def _require_columns(
    fieldnames: Sequence[str], required: Sequence[str], label: str
) -> None:
    """Reject an input missing comparison-critical columns."""
    missing = [name for name in required if name not in fieldnames]
    if missing:
        raise ComparisonError(
            f"{label} CSV is missing required columns: {', '.join(missing)}"
        )


def _index_rows(
    rows: Sequence[dict[str, str]], fieldnames: Sequence[str], label: str
) -> dict[int, dict[str, str]]:
    """Index rows by original input row while rejecting duplicates."""
    _require_columns(fieldnames, ["input_row_index"], label)
    indexed: dict[int, dict[str, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        raw_index = row.get("input_row_index", "")
        try:
            input_row_index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise ComparisonError(
                f"{label} row {row_number} has invalid input_row_index "
                f"{raw_index!r}"
            ) from exc
        if input_row_index in indexed:
            raise ComparisonError(
                f"{label} CSV has duplicate input_row_index {input_row_index}"
            )
        indexed[input_row_index] = row
    return indexed


def _parse_metric(row: Mapping[str, str], column: str) -> Optional[float]:
    """Return a finite non-negative metric, or ``None``."""
    try:
        value = float(row.get(column, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0.0 else None


def _validate_identity(
    input_row_index: int,
    baseline: Mapping[str, str],
    ours: Mapping[str, str],
    identity_columns: Sequence[str],
) -> None:
    """Ensure both rows describe the same named workload."""
    mismatches = [
        name
        for name in identity_columns
        if baseline.get(name, "") != ours.get(name, "")
    ]
    if mismatches:
        details = ", ".join(
            f"{name}={baseline.get(name)!r}/{ours.get(name)!r}"
            for name in mismatches
        )
        raise ComparisonError(
            f"input_row_index {input_row_index} identity mismatch: {details}"
        )


def compare_rows(
    baseline_rows: Sequence[dict[str, str]],
    baseline_fields: Sequence[str],
    ours_rows: Sequence[dict[str, str]],
    ours_fields: Sequence[str],
    tuning_column: str,
    latency_column: str,
) -> list[dict[str, str]]:
    """Normalize, join, validate, and compare two Pretune row sequences."""
    baseline_dimensions = _dimension_columns(baseline_fields, "baseline")
    ours_dimensions = _dimension_columns(ours_fields, "ours")
    if baseline_dimensions != ours_dimensions:
        raise ComparisonError(
            "workload dimension columns differ: "
            f"baseline={list(baseline_dimensions)!r}; "
            f"ours={list(ours_dimensions)!r}"
        )
    identity_columns = ("op_name", "variant", *baseline_dimensions)
    copy_columns = _copy_columns(baseline_dimensions)
    baseline_rows, baseline_fields = _normalize_schema(
        baseline_rows, baseline_fields
    )
    ours_rows, ours_fields = _normalize_schema(ours_rows, ours_fields)
    required = [
        "input_row_index",
        "status",
        tuning_column,
        latency_column,
        *identity_columns,
    ]
    _require_columns(baseline_fields, required, "baseline")
    _require_columns(ours_fields, required, "ours")
    baseline_by_index = _index_rows(baseline_rows, baseline_fields, "baseline")
    ours_by_index = _index_rows(ours_rows, ours_fields, "ours")
    baseline_indexes = set(baseline_by_index)
    ours_indexes = set(ours_by_index)
    if baseline_indexes != ours_indexes:
        raise ComparisonError(
            "input_row_index sets differ: "
            f"missing from ours={sorted(baseline_indexes - ours_indexes)}; "
            f"missing from baseline={sorted(ours_indexes - baseline_indexes)}"
        )

    compared: list[dict[str, str]] = []
    for input_row_index in sorted(baseline_indexes):
        baseline = baseline_by_index[input_row_index]
        ours = ours_by_index[input_row_index]
        _validate_identity(input_row_index, baseline, ours, identity_columns)
        baseline_tuning = _parse_metric(baseline, tuning_column)
        ours_tuning = _parse_metric(ours, tuning_column)
        baseline_latency = _parse_metric(baseline, latency_column)
        ours_latency = _parse_metric(ours, latency_column)

        shared_errors = []
        if baseline.get("status") != "ok":
            shared_errors.append(f"baseline status is {baseline.get('status')!r}")
        if ours.get("status") != "ok":
            shared_errors.append(f"ours status is {ours.get('status')!r}")
        tuning_errors = list(shared_errors)
        if baseline_tuning is None:
            tuning_errors.append(
                f"baseline {tuning_column} is not finite/non-negative"
            )
        if ours_tuning is None:
            tuning_errors.append(f"ours {tuning_column} is not finite/non-negative")
        elif ours_tuning == 0:
            tuning_errors.append(f"ours {tuning_column} is zero")
        throughput_errors = list(shared_errors)
        if baseline_latency is None:
            throughput_errors.append(
                f"baseline {latency_column} is not finite/non-negative"
            )
        elif baseline_latency == 0:
            throughput_errors.append(f"baseline {latency_column} is zero")
        if ours_latency is None:
            throughput_errors.append(
                f"ours {latency_column} is not finite/non-negative"
            )
        elif ours_latency == 0:
            throughput_errors.append(f"ours {latency_column} is zero")

        tuning_speedup = (
            None if tuning_errors else baseline_tuning / ours_tuning
        )
        relative_throughput = (
            None
            if throughput_errors
            else baseline_latency / ours_latency * 100.0
        )
        errors = tuning_errors + [
            error for error in throughput_errors if error not in tuning_errors
        ]

        output = {name: baseline.get(name, "") for name in copy_columns}
        output["schema_version"] = str(SCHEMA_VERSION)
        output["baseline_tuning_time_ms"] = format_ms(baseline_tuning)
        output["ours_tuning_time_ms"] = format_ms(ours_tuning)
        output["tuning_speedup"] = format_derived(tuning_speedup)
        output["baseline_latency_p50_ms"] = format_ms(baseline_latency)
        output["ours_latency_p50_ms"] = format_ms(ours_latency)
        output["relative_throughput_pct"] = format_derived(relative_throughput)
        for name in POLICY_COLUMNS:
            output[f"baseline_{name}"] = baseline.get(name, "")
            output[f"ours_{name}"] = ours.get(name, "")
        output["comparison_status"] = "invalid" if errors else "ok"
        output["comparison_error"] = "; ".join(errors)
        compared.append(output)
    return compared


def output_fieldnames(dimensions: Sequence[str]) -> list[str]:
    """Return the stable flat Comparison CSV Schema v2 header."""
    fields = ["schema_version", *_copy_columns(dimensions)]
    fields.extend(
        [
            "baseline_tuning_time_ms",
            "ours_tuning_time_ms",
            "tuning_speedup",
            "baseline_latency_p50_ms",
            "ours_latency_p50_ms",
            "relative_throughput_pct",
        ]
    )
    for name in POLICY_COLUMNS:
        fields.extend((f"baseline_{name}", f"ours_{name}"))
    fields.extend(("comparison_status", "comparison_error"))
    return fields


def _json_value(text: Any, default: Any = None) -> Any:
    """Decode a compact JSON CSV cell when possible."""
    if text in (None, ""):
        return default
    try:
        return json.loads(str(text))
    except (TypeError, json.JSONDecodeError):
        return text


def _policy_json(row: Mapping[str, str], prefix: str) -> dict[str, Any]:
    """Build one nested policy result from a flat comparison row."""
    return {
        "execution": {
            "status": row.get(f"{prefix}_status"),
            "error": row.get(f"{prefix}_error", ""),
            "tuning_time_ms": rounded_ms(
                row.get(f"{prefix}_tuning_time_ms")
            ),
            "latency_p50_ms": rounded_ms(
                row.get(f"{prefix}_latency_p50_ms")
            ),
        },
        "config_search": {
            "tuning_cache_hit": _json_value(
                str(row.get(f"{prefix}_tuning_cache_hit", "")).lower()
            ),
            "config_count": _json_value(row.get(f"{prefix}_config_count")),
            "timing_count": _json_value(row.get(f"{prefix}_timing_count")),
            "cached_count": _json_value(row.get(f"{prefix}_cached_count")),
            "measured_count": _json_value(row.get(f"{prefix}_measured_count")),
            "best_config": _json_value(row.get(f"{prefix}_best_config")),
        },
    }


def comparison_json_row(
    row: Mapping[str, str], dimensions: Sequence[str]
) -> dict[str, Any]:
    """Convert one flat comparison result to nested JSONL Schema v2."""
    return {
        "schema_version": SCHEMA_VERSION,
        "input_row_index": int(row["input_row_index"]),
        "operator": {
            "id": row.get("op_id") or None,
            "name": row.get("op_name") or None,
            "variant": row.get("variant") or None,
        },
        "workload": {
            "dimensions": {
                name: _json_value(row.get(name)) for name in dimensions
            },
            "Count": _json_value(row.get("Count")),
        },
        "dtypes": {
            "inputs": _json_value(row.get("input_dtypes"), []),
            "outputs": _json_value(row.get("output_dtypes"), []),
            "model_dtype_key": row.get("model_dtype_key") or None,
        },
        "baseline": _policy_json(row, "baseline"),
        "ours": _policy_json(row, "ours"),
        "comparison": {
            "tuning_speedup": rounded_derived(row.get("tuning_speedup")),
            "relative_throughput_pct": rounded_derived(
                row.get("relative_throughput_pct")
            ),
            "status": row.get("comparison_status"),
            "error": row.get("comparison_error", ""),
        },
    }


def write_comparison(path: Path, rows: Sequence[Mapping[str, str]]) -> Path:
    """Write the v2 CSV and its same-stem nested JSONL sibling."""
    if path.suffix.lower() != ".csv":
        raise ComparisonError("--output must name a .csv file")
    path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = path.with_suffix(".jsonl")
    if not rows:
        raise ComparisonError("cannot write an empty comparison")
    dimensions = _dimension_columns(list(rows[0]), "comparison")
    try:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=output_fieldnames(dimensions)
            )
            writer.writeheader()
            writer.writerows(rows)
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        comparison_json_row(row, dimensions),
                        sort_keys=True,
                        allow_nan=False,
                    )
                )
                handle.write("\n")
    except (OSError, csv.Error) as exc:
        raise ComparisonError(f"cannot write comparison artifacts {path}: {exc}") from exc
    return jsonl_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run CSV loading, comparison, and v2 artifact serialization."""
    args = build_parser().parse_args(argv)
    try:
        baseline_rows, baseline_fields = _read_rows(Path(args.baseline), "baseline")
        ours_rows, ours_fields = _read_rows(Path(args.ours), "ours")
        compared = compare_rows(
            baseline_rows,
            baseline_fields,
            ours_rows,
            ours_fields,
            args.tuning_column,
            args.latency_column,
        )
        output = Path(args.output)
        jsonl_output = write_comparison(output, compared)
    except ComparisonError as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 2
    valid_count = sum(row["comparison_status"] == "ok" for row in compared)
    print(
        f"wrote {len(compared)} rows ({valid_count} comparable) to "
        f"{output} and {jsonl_output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

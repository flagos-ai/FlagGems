"""Serialize FlagTune Schema v3 artifacts consistently.

The execution layer intentionally keeps its private LibTuner-oriented names.
This module is the single boundary that converts those raw worker records into
the public nested JSONL and flat CSV contracts. Floating-point measurements are
rounded only here so training and comparison calculations retain full precision.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 3
MILLISECOND_DIGITS = 6
DERIVED_DIGITS = 3


class ReportSchemaError(ValueError):
    """Reject a worker row that cannot satisfy the public identity contract."""


def _platform_identity(row: Mapping[str, Any]) -> str:
    platform_key = row.get("platform_key")
    if not isinstance(platform_key, str) or not platform_key.strip():
        raise ReportSchemaError("report row is missing required platform_key")
    return platform_key


def rounded_number(value: Any, digits: int) -> float | None:
    """Return a finite rounded float, or ``None`` for unavailable values."""
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits)


def rounded_ms(value: Any) -> float | None:
    """Round a millisecond value to nanosecond precision."""
    return rounded_number(value, MILLISECOND_DIGITS)


def format_ms(value: Any) -> str:
    """Format a finite millisecond value with exactly six decimal places."""
    number = rounded_ms(value)
    return "" if number is None else f"{number:.{MILLISECOND_DIGITS}f}"


def rounded_derived(value: Any) -> float | None:
    """Round a speedup, ratio, or percentage to three decimal places."""
    return rounded_number(value, DERIVED_DIGITS)


def format_derived(value: Any) -> str:
    """Format a finite derived metric with exactly three decimal places."""
    number = rounded_derived(value)
    return "" if number is None else f"{number:.{DERIVED_DIGITS}f}"


def _round_ms_fields(value: Any) -> Any:
    """Recursively round mapping values whose keys end in ``_ms``."""
    if isinstance(value, Mapping):
        return {
            str(key): (
                rounded_ms(item) if str(key).endswith("_ms") else _round_ms_fields(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_round_ms_fields(item) for item in value]
    return value


def _dimensions(row: Mapping[str, Any], shape_fields: Sequence[str]) -> dict[str, Any]:
    """Return ordered named dimensions, falling back to the positional shape."""
    positional = row.get("shape")
    values = list(positional) if isinstance(positional, (list, tuple)) else []
    return {
        name: row.get(name, values[index] if index < len(values) else None)
        for index, name in enumerate(shape_fields)
    }


def pretune_json_row(
    row: Mapping[str, Any], shape_fields: Sequence[str]
) -> dict[str, Any]:
    """Convert one private worker result to the public nested Schema v3 row."""
    platform_key = _platform_identity(row)
    config_search: dict[str, Any] = {
        "tuning_cache_hit": row.get("cache_hit"),
        "tuning_time_ms": rounded_ms(row.get("tuning_time_ms")),
        "config_count": row.get("candidate_config_count"),
        "timing_count": row.get("timed_config_count"),
        "cached_count": row.get("benchmark_cache_hit_count"),
        "measured_count": row.get("benchmark_success_count"),
        "best_config": row.get("best_config"),
    }
    if row.get("config_timings") is not None:
        config_search["timings"] = _round_ms_fields(row["config_timings"])
    return {
        "schema_version": SCHEMA_VERSION,
        "input_row_index": row.get("source_index"),
        "operator": {
            "id": row.get("op_id"),
            "name": row.get("op_name"),
            "variant": row.get("variant"),
        },
        "workload": {
            "dimensions": _dimensions(row, shape_fields),
            "Count": row.get("Count"),
        },
        "dtypes": {
            "inputs": row.get("input_dtypes"),
            "outputs": row.get("output_dtypes"),
        },
        "model_identity": {
            "platform_key": platform_key,
            "dtype_key": row.get("dtype_key"),
        },
        "device": {
            "gpu_token": row.get("gpu"),
            "name": row.get("gpu_name"),
            "worker_index": row.get("worker_id"),
            "metadata": row.get("gpu_metadata"),
        },
        "execution": {
            "status": row.get("status"),
            "error": row.get("error", ""),
            "first_call_ms": rounded_ms(row.get("first_call_ms")),
            "benchmark_protocol": row.get("benchmark_protocol"),
            "latency_measurement": {
                "source": row.get("latency_source"),
                "warmup_ms": row.get("latency_warmup_ms"),
                "measurement_ms": row.get("latency_iterations_ms"),
                "trials": row.get("latency_trial_count"),
            },
            "latency_ms": {
                "p20": rounded_ms(row.get("latency_p20_ms")),
                "p50": rounded_ms(row.get("latency_p50_ms")),
                "p80": rounded_ms(row.get("latency_p80_ms")),
            },
        },
        "config_search": config_search,
    }


def pretune_csv_fieldnames(shape_fields: Sequence[str]) -> list[str]:
    """Return the stable flat Pretune CSV Schema v3 header."""
    return [
        "schema_version",
        "input_row_index",
        "op_id",
        "op_name",
        "variant",
        *shape_fields,
        "Count",
        "input_dtypes",
        "output_dtypes",
        "model_dtype_key",
        "gpu",
        "gpu_name",
        "model_platform_key",
        "worker_index",
        "status",
        "tuning_cache_hit",
        "first_call_ms",
        "tuning_time_ms",
        "latency_source",
        "benchmark_requested_mode",
        "benchmark_resolved_mode",
        "benchmark_implementation",
        "benchmark_cache_policy",
        "benchmark_warmup_ms",
        "benchmark_measurement_ms",
        "benchmark_retries",
        "benchmark_per_replay_ms",
        "benchmark_fallback_reason",
        "latency_warmup_ms",
        "latency_measurement_ms",
        "latency_trials",
        "latency_p20_ms",
        "latency_p50_ms",
        "latency_p80_ms",
        "config_count",
        "timing_count",
        "cached_count",
        "measured_count",
        "best_config",
        "error",
    ]


def pretune_csv_row(
    row: Mapping[str, Any], shape_fields: Sequence[str]
) -> dict[str, Any]:
    """Convert one private worker result to the public flat CSV Schema v3 row."""
    platform_key = _platform_identity(row)
    dimensions = _dimensions(row, shape_fields)
    benchmark_protocol = row.get("benchmark_protocol")
    if not isinstance(benchmark_protocol, Mapping):
        benchmark_protocol = {}
    return {
        "schema_version": SCHEMA_VERSION,
        "input_row_index": row.get("source_index"),
        "op_id": row.get("op_id"),
        "op_name": row.get("op_name"),
        "variant": row.get("variant"),
        **dimensions,
        "Count": row.get("Count"),
        "input_dtypes": row.get("input_dtypes"),
        "output_dtypes": row.get("output_dtypes"),
        "model_dtype_key": row.get("dtype_key"),
        "gpu": row.get("gpu"),
        "gpu_name": row.get("gpu_name"),
        "model_platform_key": platform_key,
        "worker_index": row.get("worker_id"),
        "status": row.get("status"),
        "tuning_cache_hit": row.get("cache_hit"),
        "first_call_ms": format_ms(row.get("first_call_ms")),
        "tuning_time_ms": format_ms(row.get("tuning_time_ms")),
        "latency_source": row.get("latency_source"),
        "benchmark_requested_mode": benchmark_protocol.get("requested_mode"),
        "benchmark_resolved_mode": benchmark_protocol.get("resolved_mode"),
        "benchmark_implementation": benchmark_protocol.get("implementation"),
        "benchmark_cache_policy": benchmark_protocol.get("cache_policy"),
        "benchmark_warmup_ms": benchmark_protocol.get("warmup_ms"),
        "benchmark_measurement_ms": benchmark_protocol.get("measurement_ms"),
        "benchmark_retries": benchmark_protocol.get("n_retries"),
        "benchmark_per_replay_ms": format_ms(benchmark_protocol.get("per_replay_ms")),
        "benchmark_fallback_reason": benchmark_protocol.get("fallback_reason"),
        "latency_warmup_ms": row.get("latency_warmup_ms"),
        "latency_measurement_ms": row.get("latency_iterations_ms"),
        "latency_trials": row.get("latency_trial_count"),
        "latency_p20_ms": format_ms(row.get("latency_p20_ms")),
        "latency_p50_ms": format_ms(row.get("latency_p50_ms")),
        "latency_p80_ms": format_ms(row.get("latency_p80_ms")),
        "config_count": row.get("candidate_config_count"),
        "timing_count": row.get("timed_config_count"),
        "cached_count": row.get("benchmark_cache_hit_count"),
        "measured_count": row.get("benchmark_success_count"),
        "best_config": row.get("best_config"),
        "error": row.get("error", ""),
    }

"""Unit tests for strict per-source Pretune CSV comparison.

Inputs and outputs:
    Tests construct small in-memory CSV-like rows and temporary files, invoke
    the comparison API or CLI, and assert joined columns and derived metrics.

Implementation:
The CLI module is imported directly so the tests do not depend
    on an installed FlagGems package.  Fixtures intentionally repeat shape
    keys under different source indexes and inject malformed measurements.

Limitations:
    The suite does not run GPU kernels or validate real Pretune measurement
    quality; remote end-to-end validation covers production artifacts.
"""

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "flag_gems"
    / "flagtune"
    / "cli"
    / "compare.py"
)


def load_module():
    """Load the comparison CLI module without requiring package installation."""
    spec = importlib.util.spec_from_file_location("compare_pretune_tested", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_row(source_index, tuning, latency, M="16", status="ok"):
    """Create one compact production-shaped CSV mapping for test comparisons."""
    return {
        "source_index": str(source_index),
        "selected_index": str(source_index),
        "op_name": "mm",
        "variant": "general_tma",
        "shape": f"[1,{M},32,64]",
        "shape_key": f"1,{M},32,64",
        "B": "1",
        "M": str(M),
        "N": "32",
        "K": "64",
        "Count": "7",
        "dtype": "bfloat16",
        "dtype_key": "bf16-bf16-bf16",
        "platform_key": "nvidia-h20",
        "status": status,
        "cache_hit": "False",
        "tuning_time_ms": str(tuning),
        "latency_p50_ms": str(latency),
        "candidate_config_count": "10",
        "timed_config_count": "10",
        "benchmark_cache_hit_count": "0",
        "benchmark_success_count": "10",
        "best_config": "{}",
        "error": "" if status == "ok" else "failed fixture",
    }


def compare(mod, baseline, ours):
    """Invoke the comparison API using fixture keys as both CSV schemas."""
    fields = list(baseline[0])
    return mod.compare_rows(
        baseline, fields, ours, list(ours[0]), "tuning_time_ms", "latency_p50_ms"
    )


def make_v2_row(input_row_index, tuning, latency):
    """Create one Schema v2 Pretune CSV mapping."""
    return {
        "schema_version": "2",
        "input_row_index": str(input_row_index),
        "op_id": "flaggems/mm",
        "op_name": "mm",
        "variant": "general_tma",
        "B": "1",
        "M": "16",
        "N": "32",
        "K": "64",
        "Count": "7",
        "input_dtypes": '["bfloat16","bfloat16"]',
        "output_dtypes": '["bfloat16"]',
        "model_dtype_key": "bf16-bf16-bf16",
        "model_platform_key": "nvidia-h20",
        "status": "ok",
        "tuning_cache_hit": "false",
        "tuning_time_ms": str(tuning),
        "latency_p50_ms": str(latency),
        "config_count": "10",
        "timing_count": "10",
        "cached_count": "10",
        "measured_count": "0",
        "best_config": "{}",
        "error": "",
    }


def make_non_mm_v2_row(input_row_index, tuning, latency):
    """Create a Schema v2 row with non-MM workload dimensions."""
    row = make_v2_row(input_row_index, tuning, latency)
    row["op_id"] = "flaggems/reduction"
    row["op_name"] = "reduction"
    row["variant"] = "axis"
    items = list(row.items())
    variant_index = next(
        index for index, (name, _) in enumerate(items) if name == "variant"
    )
    count_index = next(
        index for index, (name, _) in enumerate(items) if name == "Count"
    )
    return dict(
        [
            *items[: variant_index + 1],
            ("X", "8"),
            ("Y", "16"),
            ("axis", "1"),
            *items[count_index:],
        ]
    )


def make_v3_row(input_row_index, tuning, latency, mode="replay"):
    """Create a protocol-aware Schema v3 Pretune CSV mapping."""
    row = make_v2_row(input_row_index, tuning, latency)
    row["schema_version"] = "3"
    row.update(
        {
            "benchmark_requested_mode": mode,
            "benchmark_resolved_mode": mode,
            "benchmark_implementation": (
                "triton_cuda_graph_replay_v1" if mode == "replay" else "triton_do_bench"
            ),
            "benchmark_cache_policy": ("warm_l2" if mode == "replay" else "cold_l2"),
            "benchmark_warmup_ms": "25",
            "benchmark_measurement_ms": "100",
            "benchmark_retries": "10" if mode == "replay" else "1",
            "benchmark_per_replay_ms": ("10.000000" if mode == "replay" else ""),
            "benchmark_fallback_reason": "",
            "latency_warmup_ms": "25",
            "latency_measurement_ms": "100",
            "latency_trials": "3",
        }
    )
    return row


def test_computes_requested_metrics_and_preserves_duplicate_shapes():
    """Compute both formulas per source row without collapsing repeated shapes."""
    mod = load_module()
    baseline = [make_row(0, 20, 2), make_row(1, 40, 4)]
    ours = [make_row(0, 5, 2.2), make_row(1, 10, 3)]

    rows = compare(mod, baseline, ours)

    assert len(rows) == 2
    assert rows[0]["tuning_speedup"] == "4.000"
    assert rows[0]["relative_throughput_pct"] == "90.909"
    assert rows[1]["tuning_speedup"] == "4.000"
    assert rows[1]["relative_throughput_pct"] == "133.333"
    assert rows[0]["baseline_measured_count"] == "10"
    assert rows[0]["input_row_index"] == "0"
    assert "source_index" not in rows[0]
    assert all(row["comparison_status"] == "ok" for row in rows)


def test_reads_v2_rows_and_rounds_only_serialized_metrics():
    """Accept native v2 input and enforce six/three decimal output precision."""
    mod = load_module()

    row = compare(
        mod,
        [make_v2_row(4, 10.123456789, 2.123456789)],
        [make_v2_row(4, 3.123456789, 1.987654321)],
    )[0]

    assert row["input_row_index"] == "4"
    assert row["baseline_tuning_time_ms"] == "10.123457"
    assert row["ours_latency_p50_ms"] == "1.987654"
    assert row["tuning_speedup"] == "3.241"
    assert row["relative_throughput_pct"] == "106.832"
    assert row["baseline_cached_count"] == "10"
    assert row["ours_measured_count"] == "0"


def test_private_rows_map_platform_key_to_public_model_column():
    """Normalize the private platform field to its public column name."""
    mod = load_module()
    source = make_row(0, 1, 1)
    source["platform_key"] = "nvidia-h20"

    rows, fields = mod._normalize_schema([source], list(source))

    assert rows[0]["model_platform_key"] == "nvidia-h20"
    assert "model_platform_key" in fields


def test_v3_requires_identical_benchmark_protocols():
    """Prevent event and graph-replay measurements from being mixed."""
    mod = load_module()
    baseline = make_v3_row(4, 10, 2, mode="event")
    ours = make_v3_row(4, 5, 2, mode="replay")

    with pytest.raises(mod.ComparisonError, match="benchmark protocol mismatch"):
        compare(mod, [baseline], [ours])


def test_v3_protocol_is_preserved_in_comparison_output():
    """Carry the shared timing protocol into CSV and JSONL-ready rows."""
    mod = load_module()
    row = compare(
        mod,
        [make_v3_row(4, 10, 2)],
        [make_v3_row(4, 5, 2)],
    )[0]

    assert row["benchmark_resolved_mode"] == "replay"
    assert row["benchmark_warmup_ms"] == "25"
    assert row["benchmark_per_replay_ms"] == "10.000000"


@pytest.mark.parametrize(
    "baseline, ours, message",
    [
        ([make_row(0, 1, 1), make_row(0, 2, 2)], [make_row(0, 1, 1)], "duplicate"),
        ([make_row(0, 1, 1)], [make_row(1, 1, 1)], "index sets differ"),
        (
            [make_row(0, 1, 1)],
            [make_row(0, 1, 1, M="17")],
            "identity mismatch",
        ),
    ],
)
def test_rejects_ambiguous_or_incompatible_source_rows(baseline, ours, message):
    """Reject duplicate indexes, mismatched sets, and mismatched identities."""
    mod = load_module()

    with pytest.raises(mod.ComparisonError, match=message):
        compare(mod, baseline, ours)


@pytest.mark.parametrize(
    "baseline, ours, expected_error",
    [
        (make_row(0, "nan", 1), make_row(0, 1, 1), "baseline tuning_time_ms"),
        (make_row(0, 1, 1), make_row(0, 0, 1), "ours tuning_time_ms is zero"),
        (make_row(0, 1, 0), make_row(0, 1, 1), "baseline latency_p50_ms is zero"),
        (
            make_row(0, 1, 1, status="failed"),
            make_row(0, 1, 1),
            "baseline status",
        ),
    ],
)
def test_keeps_invalid_measurements_but_omits_derived_metrics(
    baseline, ours, expected_error
):
    """Retain auditable raw values while suppressing undefined comparisons."""
    mod = load_module()

    row = compare(mod, [baseline], [ours])[0]

    assert row["comparison_status"] == "invalid"
    assert expected_error in row["comparison_error"]
    if "tuning_time_ms" in expected_error or "status" in expected_error:
        assert row["tuning_speedup"] == ""
    else:
        assert row["tuning_speedup"] == "1.000"
    if "latency_p50_ms" in expected_error or "status" in expected_error:
        assert row["relative_throughput_pct"] == ""
    else:
        assert row["relative_throughput_pct"] == "100.000"


def write_csv(path, rows):
    """Write fixture mappings as a temporary Pretune-compatible CSV file."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_cli_writes_baseline_and_ours_columns(tmp_path):
    """Exercise file I/O and verify the requested policy labels in output."""
    mod = load_module()
    baseline_path = tmp_path / "baseline.csv"
    ours_path = tmp_path / "ours.csv"
    output_path = tmp_path / "comparison.csv"
    write_csv(baseline_path, [make_row(0, 8, 2)])
    write_csv(ours_path, [make_row(0, 2, 1.5)])

    return_code = mod.main(
        [
            "--baseline",
            str(baseline_path),
            "--ours",
            str(ours_path),
            "--output",
            str(output_path),
        ]
    )

    with output_path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    json_row = json.loads(output_path.with_suffix(".jsonl").read_text())
    assert return_code == 0
    assert row["baseline_tuning_time_ms"] == "8.000000"
    assert row["ours_tuning_time_ms"] == "2.000000"
    assert row["tuning_speedup"] == "4.000"
    assert row["relative_throughput_pct"] == "133.333"
    assert row["model_platform_key"] == "nvidia-h20"
    assert json_row["schema_version"] == 3
    assert json_row["model_identity"] == {
        "platform_key": "nvidia-h20",
        "dtype_key": "bf16-bf16-bf16",
    }
    assert json_row["comparison"]["tuning_speedup"] == 4.0
    assert json_row["comparison"]["relative_throughput_pct"] == 133.333


def test_rejects_cross_platform_comparison():
    """Platform is part of model identity, not incidental device metadata."""
    mod = load_module()
    baseline = make_v3_row(0, 2, 1)
    ours = make_v3_row(0, 1, 1)
    ours["model_platform_key"] = "nvidia-h800"

    with pytest.raises(mod.ComparisonError, match="model_platform_key"):
        compare(mod, [baseline], [ours])


def test_comparison_requires_platform_identity():
    mod = load_module()
    baseline = make_v3_row(0, 2, 1)
    ours = make_v3_row(0, 1, 1)
    del ours["model_platform_key"]

    with pytest.raises(mod.ComparisonError, match="model_platform_key"):
        compare(mod, [baseline], [ours])


def test_non_mm_dimensions_are_preserved_in_csv_and_jsonl(tmp_path):
    """Infer arbitrary configured dimensions without introducing MM columns."""
    mod = load_module()
    baseline = make_non_mm_v2_row(2, 6, 3)
    ours = make_non_mm_v2_row(2, 2, 2)
    output_path = tmp_path / "comparison.csv"
    rows = compare(mod, [baseline], [ours])

    mod.write_comparison(output_path, rows)

    with output_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        csv_row = next(reader)
        assert reader.fieldnames is not None
        assert reader.fieldnames[5:8] == ["X", "Y", "axis"]
        assert not {"B", "M", "N", "K"} & set(reader.fieldnames)
    json_row = json.loads(output_path.with_suffix(".jsonl").read_text())
    assert {name: csv_row[name] for name in ("X", "Y", "axis")} == {
        "X": "8",
        "Y": "16",
        "axis": "1",
    }
    assert json_row["workload"]["dimensions"] == {"X": 8, "Y": 16, "axis": 1}


def test_rejects_different_workload_dimension_schemas():
    """Require baseline and ours dimension names and order to match exactly."""
    mod = load_module()
    baseline = make_non_mm_v2_row(2, 6, 3)
    ours = make_non_mm_v2_row(2, 2, 2)
    ours["dim"] = ours.pop("axis")

    with pytest.raises(mod.ComparisonError, match="dimension columns differ"):
        compare(mod, [baseline], [ours])

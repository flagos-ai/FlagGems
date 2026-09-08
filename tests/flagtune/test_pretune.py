"""Unit tests for Pretune planning, scheduling, storage, and serialization.

Inputs and outputs:
    Tests create temporary YAML, SQLite, JSONL, and CSV fixtures and assert the
    corresponding in-memory records or persisted rows.  Fake registry objects
    and an opaque adapter isolate generic planning and scheduling behavior.

Implementation:
The CLI and scheduler modules are loaded from their source paths so tests
exercise the new package boundary without requiring package installation. GPU worker
    launch is replaced with fakes for the public batch API tests, while the
    production YAML supplies operator semantics.

Limitations:
    This suite intentionally does not execute CUDA kernels, validate LibTuner
    timing accuracy, or exercise real multiprocessing.  Those paths require the
    separate Hopper smoke tests described by the Pretune validation workflow.
"""

import argparse
import csv
import importlib.util
import json
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from flag_gems.flagtune.reporting import schema as reporting_schema

HAS_FLAGTREE_FLAGTUNE = importlib.util.find_spec("triton.flagtune") is not None
pytestmark = pytest.mark.skipif(
    not HAS_FLAGTREE_FLAGTUNE,
    reason="FlagGems FlagTune Pretune tests require the optional FlagTree package",
)


@pytest.mark.parametrize(
    "converter", [reporting_schema.pretune_json_row, reporting_schema.pretune_csv_row]
)
def test_report_conversion_requires_platform_identity(converter):
    with pytest.raises(reporting_schema.ReportSchemaError, match="platform_key"):
        converter({"dtype_key": "bf16-bf16-bf16"}, ["M"])


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "flag_gems"
    / "flagtune"
    / "cli"
    / "pretune.py"
)
BENCHMARK_PATH = SCRIPT_PATH.parents[1] / "collection" / "scheduler.py"
CONFIG_PATH = (
    SCRIPT_PATH.parents[1] / "contracts" / "configs" / "mm_flagtune_configs.yaml"
)


def load_path(path, name):
    """Load one source file under an isolated module name and return the module."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_module():
    """Load the Pretune CLI module without requiring an installed package."""
    return load_path(SCRIPT_PATH, "flag_gems_pretune")


def load_benchmark_module():
    """Load the generic benchmark scheduler directly from its source path."""
    return load_path(BENCHMARK_PATH, "flag_gems_flagnbench")


def test_environment_snapshot_records_model_resolution_controls(monkeypatch):
    mod = load_module()
    expected = {
        "FLAGTUNE_LOCAL_MANIFEST": "/tmp/flagtune-manifest.json",
        "FLAGTUNE_MODEL_VERSION": "1.0.0",
        "FLAGTUNE_MODEL_DOWNLOAD_LATEST": "1",
    }
    for name, value in expected.items():
        monkeypatch.setenv(name, value)

    snapshot = mod.environment_snapshot()

    assert {name: snapshot[name] for name in expected} == expected


class FakeVariant:
    """Represent a registry variant controlled by a shape predicate."""

    def __init__(self, predicate):
        """Store the callable used to match normalized shape dictionaries."""
        self.predicate = predicate

    def matches(self, values):
        """Return the configured predicate result for one shape mapping."""
        return self.predicate(values)


class FakeOperator:
    """Model overlapping MM registry predicates used by dispatch tests."""

    op_id = "flaggems/mm"

    def __init__(self):
        """Create TMA, GEMV, and split-K variants with production-like overlap."""
        splitk = lambda shape: (
            shape["M"] < 2048 and shape["N"] < 2048 and shape["K"] >= 4096
        )
        self.variants = {
            "general_tma": FakeVariant(
                lambda shape: shape["N"] > 1 and not splitk(shape)
            ),
            "gemv": FakeVariant(lambda shape: shape["N"] == 1),
            # This intentionally overlaps gemv, like the current registry.
            "splitk": FakeVariant(splitk),
        }


class FakeSpec:
    """Provide ordered data-driven dispatch without an operator module."""

    operator_info = FakeOperator()
    op_id = "flaggems/mm"
    public_operator_name = "mm"

    def resolve_variant(self, values):
        """Select the first matching variant in the configured priority order."""
        for name in ("gemv", "splitk", "general_tma"):
            if self.operator_info.variants[name].matches(values):
                return name
        raise RuntimeError("no variant matches")


def write_yaml(path, payload):
    """Serialize one safe YAML fixture to ``path`` and return no value."""
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_operator_yaml_compiles_shape_dispatch_and_benchmark_contract():
    """Compile all Pretune semantics without importing an MM support module."""
    mod = load_module()
    spec = mod.load_operator_benchmark_spec(CONFIG_PATH)

    assert spec.op_id == "flaggems/mm"
    assert spec.public_operator_name == "mm"
    assert spec.shape.identity == ("B", "M", "N", "K")
    assert spec.shape.count_field == "Count"
    assert spec.dispatch_order == ("gemv", "splitk", "general_tma")
    assert [tensor.name for tensor in spec.benchmark.tensors] == ["a", "b"]
    assert [tensor.dtype for tensor in spec.benchmark.tensors] == [
        "runtime",
        "runtime",
    ]
    assert tuple(reference.name for reference in spec.benchmark.args) == ("a", "b")
    assert spec.resolve_variant({"B": 1, "M": 16, "N": 1, "K": 4096}) == "gemv"


def test_operator_yaml_rejects_device_placement_policy(tmp_path):
    """Keep device selection exclusively in the registered runtime adapter."""
    mod = load_module()
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["pretune"]["benchmark"]["tensors"]["a"]["device"] = "cuda"
    path = tmp_path / "device-bound.yaml"
    write_yaml(path, payload)

    with pytest.raises(mod.OperatorConfigError, match=r"unknown keys: \['device'\]"):
        mod.load_operator_benchmark_spec(path)


def test_operator_yaml_rejects_arbitrary_invocation(tmp_path):
    """Reject YAML-selected imports or callables outside the public API policy."""
    mod = load_module()
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["pretune"]["benchmark"]["invoke"]["kind"] = "python_import"
    path = tmp_path / "unsafe.yaml"
    write_yaml(path, payload)

    with pytest.raises(mod.OperatorConfigError, match="flag_gems_public"):
        mod.load_operator_benchmark_spec(path)


def test_shared_safe_references_cover_fields_shapes_and_invoke_args(tmp_path):
    """Use one ordered symbol contract across Pretune schema locations."""
    mod = load_module()
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["pretune"]["shape"]["fields"]["N"]["required"] = False
    payload["pretune"]["shape"]["fields"]["N"]["default"] = "M"
    path = tmp_path / "shared-symbols.yaml"
    write_yaml(path, payload)

    spec = mod.load_operator_benchmark_spec(path)
    values, _count = spec.shape.normalize_values({"B": 1, "M": 32, "K": 64}, "case")
    assert values["N"] == 32
    assert tuple(reference.name for reference in spec.benchmark.args) == ("a", "b")


def test_shared_safe_references_reject_forward_dependencies_and_calls(tmp_path):
    mod = load_module()
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["pretune"]["shape"]["fields"]["B"]["default"] = "M"
    path = tmp_path / "forward.yaml"
    write_yaml(path, payload)
    with pytest.raises(mod.OperatorConfigError, match="unknown symbol 'M'"):
        mod.load_operator_benchmark_spec(path)

    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["pretune"]["benchmark"]["tensors"]["a"]["shape"][0] = {
        "op": "add",
        "args": ["M", 1],
    }
    path = tmp_path / "shape-call.yaml"
    write_yaml(path, payload)
    with pytest.raises(mod.OperatorConfigError, match="does not allow operation"):
        mod.load_operator_benchmark_spec(path)


def test_operator_yaml_rejects_unknown_identity_namespace(tmp_path):
    """Allow only code-owned FlagGems public-operator resolution."""
    mod = load_module()
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["op_id"] = "external/mm"
    path = tmp_path / "external.yaml"
    write_yaml(path, payload)

    with pytest.raises(mod.OperatorConfigError, match="namespace"):
        mod.load_operator_benchmark_spec(path)


def test_public_operator_resolution_reports_missing_callable():
    """Fail explicitly when the op_id suffix is not exported by FlagGems."""
    config_mod = importlib.import_module("flag_gems.flagtune.contracts.operator")
    with pytest.raises(config_mod.OperatorConfigError, match="no public callable"):
        config_mod.resolve_public_operator(SimpleNamespace(), "flaggems/mm")


def test_load_new_shape_spec_with_optional_count(tmp_path):
    """Parse the new ordered shape schema and preserve optional Count metadata."""
    mod = load_module()
    path = tmp_path / "shapes.yaml"
    write_yaml(
        path,
        {
            "mm": {
                "shape_spec": ["M", "N", "K", "Count"],
                "shapes": [[16, 32, 64, 7]],
            }
        },
    )

    records = mod.load_shape_records(
        path, mod.load_operator_benchmark_spec(CONFIG_PATH)
    )

    assert len(records) == 1
    assert records[0].shape == [1, 16, 32, 64]
    assert records[0].shape_key == "1,16,32,64"
    assert records[0].count == 7


def test_load_legacy_shape_desc_without_count(tmp_path):
    """Retain legacy shape-description compatibility when Count is absent."""
    mod = load_module()
    path = tmp_path / "legacy.yaml"
    write_yaml(
        path,
        {
            "mm": {
                "shape_desc": "B, M, N, K",
                "shapes": [[1, 128, 256, 512]],
            }
        },
    )

    records = mod.load_shape_records(
        path, mod.load_operator_benchmark_spec(CONFIG_PATH)
    )

    assert records[0].shape == [1, 128, 256, 512]
    assert records[0].count is None


@pytest.mark.parametrize(
    "payload, message",
    [
        (
            {"mm": {"shape_spec": ["B", "M", "N", "K"], "shapes": [[2, 1, 2, 3]]}},
            "above maximum 1",
        ),
        (
            {"mm": {"shape_spec": ["M", "N", "K"], "shapes": [[0, 2, 3]]}},
            "below minimum 1",
        ),
        (
            {"addmm": {"shape_spec": ["M", "N", "K"], "shapes": [[1, 2, 3]]}},
            "exact operator key",
        ),
    ],
)
def test_shape_validation_is_strict(tmp_path, payload, message):
    """Reject bad batches, dimensions, and operator keys with explicit errors."""
    mod = load_module()
    path = tmp_path / "bad.yaml"
    write_yaml(path, payload)

    with pytest.raises(mod.PretuneError, match=message):
        mod.load_shape_records(path, mod.load_operator_benchmark_spec(CONFIG_PATH))


def make_records(mod):
    """Return deterministic MM records covering every supported variant."""
    return [
        mod.ShapeRecord(0, None, {"B": 1, "M": 16, "N": 1, "K": 4096}, 9),
        mod.ShapeRecord(1, None, {"B": 1, "M": 1024, "N": 1024, "K": 4096}, 3),
        mod.ShapeRecord(2, None, {"B": 1, "M": 4096, "N": 4096, "K": 4096}, 7),
    ]


def test_variant_resolution_follows_public_mm_priority():
    """Resolve overlapping predicates using GEMV, split-K, then TMA priority."""
    mod = load_module()
    records = make_records(mod)
    spec = FakeSpec()

    assert spec.resolve_variant(records[0].values) == "gemv"
    assert spec.resolve_variant(records[1].values) == "splitk"
    assert spec.resolve_variant(records[2].values) == "general_tma"


def test_variant_filter_happens_before_sort_and_limit():
    """Filter by exact variant before applying ordering and shape limits."""
    mod = load_module()
    selected = mod.select_shape_records(
        make_records(mod),
        FakeSpec(),
        "splitk",
        mod.SortSpec("count_descending"),
        1,
    )

    assert len(selected) == 1
    assert selected[0].source_index == 1
    assert selected[0].variant == "splitk"
    assert selected[0].selected_index == 0


def test_percentage_shape_limit_rounds_down_after_variant_filtering():
    """Apply percentages to eligible shapes and never exceed the requested share."""
    mod = load_module()
    records = [
        mod.ShapeRecord(
            0,
            None,
            {"B": 1, "M": 16, "N": 1, "K": 4096},
            0,
        ),
        *[
            mod.ShapeRecord(
                index + 1,
                None,
                {"B": 1, "M": 4096 + index, "N": 4096, "K": 4096},
                index,
            )
            for index in range(5)
        ],
    ]

    selected = mod.select_shape_records(
        records,
        FakeSpec(),
        "general_tma",
        mod.SortSpec("default"),
        "50%",
    )

    assert [record.source_index for record in selected] == [1, 2]
    assert [record.selected_index for record in selected] == [0, 1]


def test_percentage_shape_limit_rejects_zero_resolved_shapes():
    """Reject a floor-rounded empty selection instead of running no work."""
    mod = load_module()

    with pytest.raises(mod.PretuneError, match="selects 0 of 3"):
        mod.select_shape_records(
            make_records(mod),
            FakeSpec(),
            None,
            mod.SortSpec("default"),
            "33%",
        )


@pytest.mark.parametrize(
    "text, expected",
    [
        ("50", 50),
        ("50%", "50%"),
        ("12.50%", "12.5%"),
        ("100.0%", "100%"),
    ],
)
def test_max_shapes_parser_accepts_counts_and_percentages(text, expected):
    """Normalize supported absolute and percentage CLI forms."""
    mod = load_module()

    assert mod.parse_max_shapes(text) == expected


@pytest.mark.parametrize("text", ["0", "-1", "1.5", "0%", "101%", "%", "abc"])
def test_max_shapes_parser_rejects_invalid_limits(text):
    """Reject empty, non-integral absolute, and out-of-range percentage limits."""
    mod = load_module()

    with pytest.raises(argparse.ArgumentTypeError):
        mod.parse_max_shapes(text)


def test_count_sort_requires_count():
    """Fail count-based sorting when a selected shape lacks Count metadata."""
    mod = load_module()
    records = [
        mod.ShapeRecord(0, None, {"B": 1, "M": 4096, "N": 4096, "K": 4096}, None)
    ]

    with pytest.raises(mod.PretuneError, match="requires Count"):
        mod.select_shape_records(
            records,
            FakeSpec(),
            None,
            mod.SortSpec("count_ascending"),
            None,
        )


def test_random_sort_is_reproducible():
    """Produce a nontrivial but deterministic order for an explicit seed."""
    mod = load_module()
    records = [
        mod.ShapeRecord(
            index,
            None,
            {"B": 1, "M": 4096 + index, "N": 4096, "K": 4096},
            index,
        )
        for index in range(10)
    ]
    first = mod.select_shape_records(
        records,
        FakeSpec(),
        None,
        mod.SortSpec("random", 2026),
        None,
    )
    second = mod.select_shape_records(
        records,
        FakeSpec(),
        None,
        mod.SortSpec("random", 2026),
        None,
    )

    assert [record.source_index for record in first] == [
        record.source_index for record in second
    ]
    assert [record.source_index for record in first] != list(range(10))


def test_cli_defaults_and_sort_parser():
    """Verify Pretune defaults plus random-sort and operator parsing."""
    mod = load_module()
    args = mod.build_parser().parse_args(
        [
            "--shape-config",
            "x.yaml",
            "--flagtune-config",
            "mm.yaml",
            "--op",
            "mm",
        ]
    )

    assert args.dtypes == "bfloat16"
    assert args.parallel is None
    assert args.warmup == 25
    assert args.iterations == 100
    assert args.benchmark_mode == "replay"
    assert args.benchmark_retries == 10
    assert args.latency_warmup == 25
    assert args.latency_iterations == 100
    assert args.latency_trials == 3
    assert args.max_shapes is None
    assert args.keep_intermediate_files is False
    kept = mod.build_parser().parse_args(
        [
            "--shape-config",
            "x.yaml",
            "--flagtune-config",
            "mm.yaml",
            "--op",
            "mm",
            "--keep-intermediate-files",
        ]
    )
    assert kept.keep_intermediate_files is True
    assert mod.parse_sort("random=-7") == mod.SortSpec("random", -7)
    assert mod.parse_op("mm/gemv") == ("mm", "gemv")

    percentage = mod.build_parser().parse_args(
        [
            "--shape-config",
            "x.yaml",
            "--flagtune-config",
            "mm.yaml",
            "--op",
            "mm",
            "--max-shapes",
            "50%",
        ]
    )
    assert percentage.max_shapes == "50%"


def test_intermediate_cleanup_is_strictly_scoped_to_run_directory(tmp_path):
    """Delete named process files while protecting run roots and outside data."""
    mod = load_module()
    run_dir = tmp_path / "run"
    worker_dir = run_dir / "benchmark-workers"
    worker_dir.mkdir(parents=True)
    (worker_dir / "worker_0.log").write_text("log", encoding="utf-8")
    corpus = run_dir / "benchmark_data.jsonl"
    corpus.write_text("{}\n", encoding="utf-8")
    final = run_dir / "pretune.csv"
    final.write_text("status\nok\n", encoding="utf-8")
    outside = tmp_path / "outside.db"
    outside.write_text("keep", encoding="utf-8")

    removed = mod.remove_intermediate_artifacts(
        run_dir, [worker_dir, corpus, run_dir / "missing"]
    )

    assert removed == ["benchmark-workers", "benchmark_data.jsonl"]
    assert not worker_dir.exists()
    assert not corpus.exists()
    assert final.exists()
    assert outside.exists()
    with pytest.raises(mod.PretuneIOError, match="outside run"):
        mod.remove_intermediate_artifacts(run_dir, [run_dir])
    with pytest.raises(mod.PretuneIOError, match="outside run"):
        mod.remove_intermediate_artifacts(run_dir, [outside])


def create_cache_db(path, config_rows, benchmark_rows):
    """Create a minimal ConfigCache/BenchmarkCache SQLite fixture at ``path``."""
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE "kernel" (key_0 INTEGER PRIMARY KEY, BLOCK INTEGER)')
        conn.execute(
            'CREATE TABLE "kernel_benchmark" '
            "(key_0 INTEGER, BLOCK INTEGER, p50 REAL, PRIMARY KEY (key_0, BLOCK))"
        )
        conn.executemany('INSERT INTO "kernel" VALUES (?, ?)', config_rows)
        conn.executemany(
            'INSERT INTO "kernel_benchmark" VALUES (?, ?, ?)', benchmark_rows
        )


def test_merge_sqlite_shards_preserves_existing_rows(tmp_path):
    """Merge shards with insert-ignore semantics while retaining target rows."""
    mod = load_benchmark_module()
    target = tmp_path / "target.db"
    shard0 = tmp_path / "worker0.db"
    shard1 = tmp_path / "worker1.db"
    create_cache_db(target, [(1, 16)], [(1, 16, 1.0)])
    create_cache_db(shard0, [(1, 32), (2, 64)], [(1, 32, 0.8), (2, 64, 0.7)])
    create_cache_db(shard1, [(2, 128), (3, 256)], [(2, 128, 0.6), (3, 256, 0.5)])

    summary = mod.merge_sqlite_shards(target, [shard0, shard1])

    with sqlite3.connect(target) as conn:
        configs = conn.execute('SELECT * FROM "kernel" ORDER BY key_0').fetchall()
        benchmarks = conn.execute(
            'SELECT * FROM "kernel_benchmark" ORDER BY key_0, BLOCK'
        ).fetchall()
    assert configs == [(1, 16), (2, 64), (3, 256)]
    assert len(benchmarks) == 5
    assert summary["visited_tables"] == 4
    assert summary["inserted_rows"] == 6


def test_write_outputs_keeps_structured_jsonl_and_flat_csv(tmp_path):
    """Preserve nested JSON while flattening cache audit fields into CSV."""
    mod = load_module()
    row = {
        "source_index": 0,
        "selected_index": 0,
        "op_id": "flaggems/mm",
        "op_name": "mm",
        "variant": "general_tma",
        "shape": [1, 16, 32, 64],
        "shape_key": "1,16,32,64",
        "B": 1,
        "M": 16,
        "N": 32,
        "K": 64,
        "Count": 7,
        "input_dtypes": ["bfloat16", "bfloat16"],
        "output_dtypes": ["bfloat16"],
        "dtype_key": "bf16-bf16-bf16",
        "gpu": "0",
        "gpu_name": "Fake GPU",
        "platform_key": "nvidia-h20",
        "worker_id": 0,
        "cache_hit": False,
        "first_call_ms": 1.234567891,
        "tuning_time_ms": 2.345678912,
        "latency_source": "libtuner_selected_config_fresh",
        "benchmark_protocol": {
            "requested_mode": "replay",
            "resolved_mode": "replay",
            "implementation": "triton_cuda_graph_replay_v1",
            "cache_policy": "warm_l2",
            "warmup_ms": 25,
            "measurement_ms": 100,
            "n_retries": 10,
            "per_replay_ms": 10.0,
            "fallback_reason": None,
        },
        "latency_warmup_ms": 200,
        "latency_iterations_ms": 500,
        "latency_trial_count": 5,
        "latency_p20_ms": 0.123456789,
        "latency_p50_ms": 0.234567891,
        "latency_p80_ms": 0.345678912,
        "candidate_config_count": 12,
        "timed_config_count": 10,
        "best_config": {"BLOCK_M": 16},
        "benchmark_cache_hit_count": 9,
        "benchmark_success_count": 3,
        "status": "ok",
        "error": "",
    }
    failed_row = {
        **row,
        "source_index": 1,
        "selected_index": 1,
        "status": "failed",
        "benchmark_cache_hit_count": None,
        "benchmark_success_count": None,
        "best_config": None,
        "error": "benchmark failed",
    }

    mod.write_outputs(tmp_path, [row, failed_row], ["B", "M", "N", "K"])

    json_rows = [
        json.loads(line)
        for line in (tmp_path / "pretune.jsonl").read_text().splitlines()
    ]
    with (tmp_path / "pretune.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        csv_rows = list(reader)
        assert reader.fieldnames == [
            "schema_version",
            "input_row_index",
            "op_id",
            "op_name",
            "variant",
            "B",
            "M",
            "N",
            "K",
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
    assert json_rows[0]["schema_version"] == 3
    assert json_rows[0]["input_row_index"] == 0
    assert json_rows[0]["workload"] == {
        "dimensions": {"B": 1, "M": 16, "N": 32, "K": 64},
        "Count": 7,
    }
    assert json_rows[0]["config_search"]["best_config"] == {"BLOCK_M": 16}
    assert json_rows[0]["config_search"]["cached_count"] == 9
    assert json_rows[0]["config_search"]["measured_count"] == 3
    assert json_rows[0]["model_identity"] == {
        "platform_key": "nvidia-h20",
        "dtype_key": "bf16-bf16-bf16",
    }
    assert json_rows[0]["execution"]["first_call_ms"] == 1.234568
    assert json_rows[0]["execution"]["benchmark_protocol"]["resolved_mode"] == "replay"
    assert json_rows[0]["execution"]["latency_measurement"] == {
        "source": "libtuner_selected_config_fresh",
        "warmup_ms": 200,
        "measurement_ms": 500,
        "trials": 5,
    }
    assert json_rows[1]["config_search"]["cached_count"] is None
    assert json_rows[1]["config_search"]["measured_count"] is None
    assert "selected_index" not in json_rows[0]
    assert "shape_key" not in json_rows[0]
    assert csv_rows[0]["schema_version"] == "3"
    assert csv_rows[0]["input_row_index"] == "0"
    assert csv_rows[0]["Count"] == "7"
    assert csv_rows[0]["model_platform_key"] == "nvidia-h20"
    assert json.loads(csv_rows[0]["best_config"]) == {"BLOCK_M": 16}
    assert csv_rows[0]["cached_count"] == "9"
    assert csv_rows[0]["measured_count"] == "3"
    assert csv_rows[0]["first_call_ms"] == "1.234568"
    assert csv_rows[0]["latency_source"] == "libtuner_selected_config_fresh"
    assert csv_rows[0]["benchmark_requested_mode"] == "replay"
    assert csv_rows[0]["benchmark_resolved_mode"] == "replay"
    assert csv_rows[0]["benchmark_warmup_ms"] == "25"
    assert csv_rows[0]["benchmark_retries"] == "10"
    assert csv_rows[0]["benchmark_per_replay_ms"] == "10.000000"
    assert csv_rows[0]["latency_warmup_ms"] == "200"
    assert csv_rows[0]["latency_measurement_ms"] == "500"
    assert csv_rows[0]["latency_trials"] == "5"
    assert csv_rows[0]["latency_p50_ms"] == "0.234568"
    assert csv_rows[1]["cached_count"] == ""
    assert csv_rows[1]["measured_count"] == ""
    assert "selected_index" not in csv_rows[0]
    assert "shape_key" not in csv_rows[0]


def test_worker_success_and_failure_rows_use_platform_key(monkeypatch, tmp_path):
    """Keep the private worker identity field aligned in both result branches."""
    from flag_gems.flagtune.runtime import executor as executor_mod

    libentry_mod = importlib.import_module("flag_gems.utils.libentry")

    class FakeConfig:
        pre_hook = None

        @staticmethod
        def all_kwargs():
            return {"BLOCK": 16}

    class FakeTuner:
        def __init__(self):
            self.configs = [FakeConfig()]
            self.strategy = []
            self.do_bench = lambda _call, _quantiles: [1.0, 0.9, 1.1]

        def apply_flagtune(self):
            return None

        def _set_configs_and_strategy(self, configs, strategy):
            self.configs = configs
            self.strategy = strategy

        @contextmanager
        def use_benchmark_protocol(self, *_args):
            yield SimpleNamespace(as_dict=lambda: {"resolved_mode": "event"})

        @contextmanager
        def use_run_mode(self, _mode):
            yield

    tuner = FakeTuner()
    worker = executor_mod.BenchmarkWorker.__new__(executor_mod.BenchmarkWorker)
    worker.spec = SimpleNamespace(
        source_sha256="sha256",
        op_id="flaggems/mm",
        public_operator_name="mm",
        shape=SimpleNamespace(identity=("M",)),
    )
    worker.base_states = {}
    worker.device_runtime = SimpleNamespace(
        dtype=lambda name: name,
        synchronize=lambda: None,
        descriptor=SimpleNamespace(device_name="NVIDIA H20-3e"),
        metadata=lambda _index=0: {
            "backend": "cuda",
            "vendor": "NVIDIA",
            "device_name": "NVIDIA H20-3e",
            "architecture": "sm90",
            "platform_key": "nvidia-h20",
        },
    )
    worker._find_tuner = lambda _variant: (object(), tuner)
    worker._make_tensors = lambda _values, _dtypes: {}
    worker._benchmark_selected_config = lambda **_kwargs: (0.9, 1.0, 1.1)

    def invoke(_tensors):
        tuner.best_config = tuner.configs[0]
        return SimpleNamespace(dtype="bfloat16")

    worker._invoke = invoke
    monkeypatch.setattr(libentry_mod, "clear_libentry_dispatch_cache", lambda _k: None)
    payload = {
        "config_sha256": "sha256",
        "source_index": 0,
        "selected_index": 0,
        "variant": "general",
        "values": {"M": 16},
        "count": 1,
        "configs": None,
    }

    success = worker.benchmark(
        payload,
        dtype_names=["bfloat16"],
        warmup=1,
        iterations=1,
        benchmark_mode="event",
        benchmark_retries=1,
        tuning_run_mode="normal",
        latency_warmup=1,
        latency_iterations=1,
        latency_trials=1,
        gpu_token="0",
        worker_id=0,
    )
    failure = worker.failure_result(
        payload,
        dtype_names=["bfloat16"],
        gpu_token="0",
        worker_id=0,
        exc=RuntimeError("failed"),
    )

    assert success["platform_key"] == "nvidia-h20"
    assert "gpu_key" not in success
    assert failure["gpu_name"] == "NVIDIA H20-3e"
    assert failure["platform_key"] == "nvidia-h20"
    assert failure["gpu_metadata"]["architecture"] == "sm90"
    assert "gpu_key" not in failure
    assert reporting_schema.pretune_json_row(failure, ["M"])["model_identity"] == {
        "platform_key": "nvidia-h20",
        "dtype_key": None,
    }
    load_module().write_outputs(tmp_path, [failure], ["M"])


def test_generic_scheduler_prepares_cases_from_operator_yaml():
    """Prepare shapes and configs without importing an operator adapter."""
    mod = load_benchmark_module()
    shape = {"M": 17, "N": 3, "K": 32}
    configs = [{"BLOCK_M": 16, "num_warps": 4}]
    tasks = mod._prepare_tasks(
        [
            (shape, configs),
            ({"M": 16, "N": 1, "K": 64}, None),
            [1, 32, 32, 32],
            (
                {
                    "values": {"B": 1, "M": 64, "N": 64, "K": 64},
                    "count": "7",
                    "variant": "general_tma",
                },
                None,
            ),
        ],
        CONFIG_PATH,
    )

    assert tasks[0].payload["values"] == {"B": 1, "M": 17, "N": 3, "K": 32}
    assert tasks[0].payload["configs"] == configs
    assert tasks[0].payload["variant"] == "general_tma"
    assert set(tasks[0].to_json()) == {"task_index", "payload"}
    assert tasks[1].payload["variant"] == "gemv"
    assert tasks[2].payload["values"] == {"B": 1, "M": 32, "N": 32, "K": 32}
    assert tasks[3].payload["count"] == 7


def test_public_batch_api_returns_input_order_and_fail_fast_state(
    tmp_path, monkeypatch
):
    """Forward scheduler options and preserve ordered rows plus batch metadata."""
    mod = load_benchmark_module()
    seen = {}

    def fake_launch(tasks, **kwargs):
        """Capture scheduler inputs and return deterministic mixed-status rows."""
        seen["tasks"] = tasks
        seen["kwargs"] = kwargs
        return (
            [
                {"task_index": 0, "token": tasks[0].payload["values"], "status": "ok"},
                {
                    "task_index": 1,
                    "token": tasks[1].payload["values"],
                    "status": "failed",
                },
            ],
            [1],
            True,
            [tmp_path / "worker.log"],
        )

    monkeypatch.setattr(mod, "_launch_workers", fake_launch)
    batch = mod.run_shape_config_benchmarks(
        [
            ({"M": 16, "N": 16, "K": 16}, [{"BLOCK_M": 16}]),
            ({"M": 32, "N": 32, "K": 32}, None),
        ],
        operator_config=CONFIG_PATH,
        gpu_tokens=["3"],
        database_url="postgresql://example/flaggems",
        work_dir=tmp_path,
        fail_fast=True,
        tuning_run_mode="exhaustive_collection",
    )

    assert [row["token"] for row in batch.results] == [
        {"B": 1, "M": 16, "N": 16, "K": 16},
        {"B": 1, "M": 32, "N": 32, "K": 32},
    ]
    assert len(seen["tasks"]) == 2
    assert seen["kwargs"]["operator_config"] == CONFIG_PATH.resolve()
    assert seen["kwargs"]["fail_fast"] is True
    assert seen["kwargs"]["tuning_run_mode"] == "exhaustive_collection"
    assert seen["kwargs"]["benchmark_mode"] == "replay"
    assert seen["kwargs"]["benchmark_retries"] == 10
    assert seen["kwargs"]["dtypes"] == ["bfloat16", "bfloat16"]
    assert batch.worker_returncodes == [1]
    assert batch.fail_fast_triggered is True


def test_public_batch_api_rejects_unknown_or_misaligned_input_dtypes(tmp_path):
    mod = load_benchmark_module()
    cases = [({"M": 16, "N": 16, "K": 16}, None)]
    common = {
        "operator_config": CONFIG_PATH,
        "gpu_tokens": ["0"],
        "database_url": "postgresql://example/flaggems",
        "work_dir": tmp_path,
    }
    with pytest.raises(mod.BenchmarkError, match="unsupported tensor dtype"):
        mod.run_shape_config_benchmarks(cases, dtypes="float8_unknown", **common)
    with pytest.raises(mod.BenchmarkError, match="invoke.args has 2"):
        mod.run_shape_config_benchmarks(
            cases, dtypes="bfloat16,float16,float32", **common
        )
    with pytest.raises(mod.BenchmarkError, match="tuning_run_mode"):
        mod.run_shape_config_benchmarks(
            cases, tuning_run_mode="ambient_environment", **common
        )
    with pytest.raises(mod.BenchmarkError, match="benchmark_mode"):
        mod.run_shape_config_benchmarks(cases, benchmark_mode="cuda_graph", **common)
    with pytest.raises(mod.BenchmarkError, match="benchmark_retries"):
        mod.run_shape_config_benchmarks(cases, benchmark_retries=0, **common)


def test_sqlite_url_classification(tmp_path):
    """Distinguish file, in-memory, and externally shared database URLs."""
    mod = load_benchmark_module()
    path = tmp_path / "cache.db"

    assert mod.parse_sqlite_url(mod._sqlite_url(path)) == ("sqlite", path.resolve())
    assert mod.parse_sqlite_url("sqlite:///:memory:") == ("sqlite_memory", None)
    assert mod.parse_sqlite_url("postgresql+psycopg:///flaggems") == ("shared", None)

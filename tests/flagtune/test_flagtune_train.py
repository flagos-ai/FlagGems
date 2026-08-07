"""Validate the offline FlagTune training CLI and progress-reporting helpers.

The tests avoid CUDA execution. They load the uninstalled CLI module, verify
argument and streaming-record contracts, and exercise text progress behavior
with temporary worker logs and lightweight fake registry/config objects.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HAS_FLAGTREE_FLAGTUNE = importlib.util.find_spec("triton.flagtune") is not None
pytestmark = pytest.mark.skipif(
    not HAS_FLAGTREE_FLAGTUNE,
    reason="FlagGems offline FlagTune training tests require the optional FlagTree package",
)

TRAIN_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "flag_gems"
    / "flagtune"
    / "cli"
    / "train.py"
)


def load_path(path, name):
    """Import one source file under an isolated module name and return it.

    The helper allows tests to exercise the CLI before an editable FlagGems
    installation. It intentionally registers the module in ``sys.modules`` so
    dataclasses and later normal imports observe the same module object.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_training_cli_requires_config_and_variant():
    """Check required training config, variant identity, and numeric defaults."""
    mod = load_path(TRAIN_PATH, "flag_gems_flagtune_train")
    parser = mod.build_parser()
    args = parser.parse_args(
        [
            "--shape-config",
            "shapes.yaml",
            "--flagtune-config",
            "mm_flagtune_configs.yaml",
            "--variant",
            "general_tma",
            "--model-version",
            "1.0.0",
        ]
    )

    mod.validate_args(args)
    assert args.dtypes == "bfloat16"
    assert args.warmup == 25
    assert args.iterations == 100
    assert args.benchmark_mode == "replay"
    assert args.benchmark_retries == 10
    assert args.n_estimators == 1200
    assert args.progress_interval == 50
    assert args.model_version == "1.0.0"
    assert args.keep_intermediate_files is False

    percentage = parser.parse_args(
        [
            "--shape-config",
            "shapes.yaml",
            "--flagtune-config",
            "mm_flagtune_configs.yaml",
            "--variant",
            "general_tma",
            "--model-version",
            "1.0.0",
            "--max-shapes",
            "50%",
        ]
    )
    mod.validate_args(percentage)
    assert percentage.max_shapes == "50%"

    kept = parser.parse_args(
        [
            "--shape-config",
            "shapes.yaml",
            "--flagtune-config",
            "mm_flagtune_configs.yaml",
            "--variant",
            "general_tma",
            "--model-version",
            "1.0.0",
            "--keep-intermediate-files",
        ]
    )
    assert kept.keep_intermediate_files is True

    args.flagtune_config = ""
    with pytest.raises(mod.TrainError, match="flagtune-config"):
        mod.validate_args(args)


def test_training_cli_rejects_unsafe_variant():
    """Reject variants that could escape the derived model directory."""
    mod = load_path(TRAIN_PATH, "flag_gems_flagtune_train_unsafe")
    args = mod.build_parser().parse_args(
        [
            "--shape-config",
            "shapes.yaml",
            "--flagtune-config",
            "mm_flagtune_configs.yaml",
            "--variant",
            "../outside",
            "--model-version",
            "1.0.0",
        ]
    )

    with pytest.raises(mod.TrainError, match="single-segment"):
        mod.validate_args(args)


def test_training_cli_rejects_negative_progress_interval():
    """Reject negative config-progress intervals while allowing zero to disable."""
    mod = load_path(TRAIN_PATH, "flag_gems_flagtune_train_bad_progress")
    args = mod.build_parser().parse_args(
        [
            "--shape-config",
            "shapes.yaml",
            "--flagtune-config",
            "mm_flagtune_configs.yaml",
            "--variant",
            "general_tma",
            "--model-version",
            "1.0.0",
            "--progress-interval",
            "-1",
        ]
    )

    with pytest.raises(mod.TrainError, match="progress-interval"):
        mod.validate_args(args)


def test_status_and_disabled_progress_have_explicit_output_contract(capsys):
    """Keep lifecycle status visible while disabled fine progress stays silent."""
    mod = load_path(TRAIN_PATH, "flag_gems_flagtune_train_status")

    mod._status("unit-test lifecycle")
    reporter = mod._progress(total=3, enabled=False)
    reporter.update(2)
    reporter.close()

    captured = capsys.readouterr()
    assert "[FlagTune train " in captured.out
    assert "unit-test lifecycle" in captured.out
    assert "Benchmark shapes" not in captured.out


class FakeVariant:
    """Provide the minimal variant normalization contract used by row export."""

    op_id = "flaggems/mm"
    name = "general_tma"
    input_names = ["M", "N", "K", "stride_am", "stride_bk"]

    @staticmethod
    def normalize_inputs(values):
        """Return model inputs while deriving contiguous matrix strides."""
        return {
            "M": values["M"],
            "N": values["N"],
            "K": values["K"],
            "stride_am": values["K"],
            "stride_bk": values["N"],
        }


def _run_training_with_results(mod, monkeypatch, tmp_path, results, exported):
    """Run the training coordinator with collection and XGBoost boundaries faked."""

    class Record:
        @staticmethod
        def to_benchmark_shape():
            return {"M": 16}

    variant = SimpleNamespace(
        name="general_tma",
        feature_names=("M",),
        iter_configs=lambda: iter(({"BLOCK": 16},)),
    )
    operator_info = SimpleNamespace(
        op_id="flaggems/mm",
        get_variant=lambda _name: variant,
    )
    spec = SimpleNamespace(
        operator_info=operator_info,
        source_sha256="sha256",
        shape=SimpleNamespace(identity=("M",)),
    )
    context = SimpleNamespace(
        visible_device_count=1,
        backend_name="cuda",
        device_names=("NVIDIA H20-3e",),
        device_architectures=("sm90",),
    )
    batch = SimpleNamespace(
        results=results,
        worker_returncodes=[0],
        database_merge={},
        database_merge_error="",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    monkeypatch.setattr(mod, "load_operator_benchmark_spec", lambda _path: spec)
    monkeypatch.setattr(
        mod, "load_shape_records", lambda _path, _spec: [Record(), Record()]
    )
    monkeypatch.setattr(
        mod,
        "initialize_planning_context",
        lambda _spec: (context, SimpleNamespace(op_id="flaggems/mm")),
    )
    monkeypatch.setattr(
        mod,
        "parse_sort",
        lambda _text: SimpleNamespace(mode="source", seed=None),
    )
    monkeypatch.setattr(mod, "select_shape_records", lambda records, *_args: records)
    monkeypatch.setattr(mod, "visible_device_tokens", lambda _context: ["0"])
    monkeypatch.setattr(mod, "make_run_dir", lambda *_args: run_dir)
    monkeypatch.setattr(
        mod,
        "_database_url",
        lambda *_args: ("sqlite:///test.db", tmp_path / "test.db", "fixture"),
    )
    monkeypatch.setattr(mod, "sanitize_db_url", lambda value: value)
    monkeypatch.setattr(mod, "environment_snapshot", lambda: {})
    monkeypatch.setattr(
        mod,
        "_progress",
        lambda *_args: SimpleNamespace(update=lambda _count: None, close=lambda: None),
    )
    monkeypatch.setattr(
        mod, "run_shape_config_benchmarks", lambda *_args, **_kwargs: batch
    )
    monkeypatch.setattr(
        mod,
        "_append_collection_rows",
        lambda *_args: (len(results), len(results), 0),
    )
    monkeypatch.setattr(mod, "_training_options", lambda _args: object())
    monkeypatch.setattr(mod, "write_manifest", lambda *_args: None)

    from triton.flagtune.contract import operator_schema
    from triton.flagtune.training import ranker

    monkeypatch.setattr(
        operator_schema, "model_config_sha256", lambda _config: "digest"
    )
    monkeypatch.setattr(
        ranker,
        "train_xgboost_ranker",
        lambda *_args: (object(), {"xgboost_fit_elapsed_s": 0.0}),
    )

    def fake_export(_model, _variant, _run_dir, _summary, **kwargs):
        exported.update(kwargs)
        return SimpleNamespace(model_path=run_dir / "model.tar.gz", model_config={})

    monkeypatch.setattr(ranker, "export_ranker_model", fake_export)
    args = mod.build_parser().parse_args(
        [
            "--shape-config",
            str(tmp_path / "shapes.yaml"),
            "--flagtune-config",
            str(tmp_path / "config.yaml"),
            "--variant",
            "general_tma",
            "--model-version",
            "1.2.3",
            "--output",
            str(tmp_path),
            "--keep-intermediate-files",
        ]
    )
    return mod.run_main(args)


def _successful_collection_result(architecture="sm90"):
    return {
        "status": "ok",
        "platform_key": "nvidia-h20",
        "dtype_key": "bf16-bf16-bf16",
        "input_dtypes": ["bfloat16", "bfloat16"],
        "output_dtypes": ["bfloat16"],
        "gpu_metadata": {
            "backend": "cuda",
            "vendor": "NVIDIA",
            "device_name": "NVIDIA H20-3e",
            "architecture": architecture,
            "platform_key": "nvidia-h20",
        },
    }


def test_training_rejects_same_platform_with_different_architectures(
    monkeypatch, tmp_path
):
    mod = load_path(TRAIN_PATH, "flag_gems_flagtune_train_architecture")

    with pytest.raises(mod.TrainError, match="platform/architecture metadata"):
        _run_training_with_results(
            mod,
            monkeypatch,
            tmp_path,
            [
                _successful_collection_result("sm90"),
                _successful_collection_result("sm89"),
            ],
            {},
        )


def test_training_exports_platform_identity_and_retained_architecture(
    monkeypatch, tmp_path
):
    mod = load_path(TRAIN_PATH, "flag_gems_flagtune_train_export_identity")
    exported = {}

    result = _successful_collection_result()
    assert (
        _run_training_with_results(
            mod, monkeypatch, tmp_path, [result, dict(result)], exported
        )
        == 0
    )
    assert exported["identity"].platform_key == "nvidia-h20"
    assert exported["gpu"]["architecture"] == "sm90"


def test_collection_rows_are_flattened_to_streaming_training_jsonl(tmp_path):
    """Flatten complete per-shape timings into one auditable JSONL row each."""
    mod = load_path(TRAIN_PATH, "flag_gems_flagtune_train_rows")
    data_path = tmp_path / "data.jsonl"
    failure_path = tmp_path / "failures.jsonl"
    result = {
        "status": "ok",
        "shape_key": "1,64,32,128",
        "source_index": 3,
        "selected_index": 0,
        "Count": 9,
        "B": 1,
        "M": 64,
        "N": 32,
        "K": 128,
        "gpu": "1",
        "gpu_name": "Fake GPU",
        "platform_key": "nvidia-h20",
        "gpu_metadata": {
            "backend": "cuda",
            "vendor": "nvidia",
            "device_name": "Fake GPU",
            "architecture": "sm90",
            "platform_key": "nvidia-h20",
        },
        "input_dtypes": ["bfloat16", "bfloat16"],
        "output_dtypes": ["bfloat16"],
        "dtype_key": "bf16-bf16-bf16",
        "config_timings": [
            {
                "config": {"BLOCK_M": 16},
                "latency_ms": 1.234567891,
                "latency_p50_ms": 1.234567891,
                "latency_p20_ms": 1.000000499,
                "latency_p80_ms": 1.500000501,
                "status": "ok",
            }
        ],
    }

    counts = mod._append_collection_rows(
        data_path,
        failure_path,
        [result],
        FakeVariant(),
        ["B", "M", "N", "K"],
        1,
    )

    row = json.loads(data_path.read_text())
    assert counts == (1, 1, 0)
    assert row["schema_version"] == 3
    assert row["input_row_index"] == 3
    assert row["operator"] == {
        "id": "flaggems/mm",
        "variant": "general_tma",
    }
    assert row["workload"] == {
        "dimensions": {"B": 1, "M": 64, "N": 32, "K": 128},
        "Count": 9,
    }
    assert row["ranking_group"] == {
        "operator_id": "flaggems/mm",
        "variant": "general_tma",
        "dimensions": {
            "M": 64,
            "N": 32,
            "K": 128,
            "stride_am": 128,
            "stride_bk": 32,
        },
        "model_dtype_key": "bf16-bf16-bf16",
    }
    assert row["model_identity"]["platform_key"] == "nvidia-h20"
    assert row["model_identity"]["dtype_key"] == "bf16-bf16-bf16"
    assert row["Count"] == 9
    assert row["latency_ms"] == 1.234568
    assert row["latency_p20_ms"] == 1.0
    assert row["latency_p80_ms"] == 1.500001
    assert "shape_key" not in row
    assert "source_shape_key" not in row
    assert row["inputs"]["stride_am"] == 128
    assert row["config"] == {"BLOCK_M": 16}


def test_collection_failure_uses_workload_not_model_input_dimensions(tmp_path):
    """Keep failed-shape identity independent from derived model inputs."""
    mod = load_path(TRAIN_PATH, "flag_gems_flagtune_train_failure_rows")
    data_path = tmp_path / "data.jsonl"
    failure_path = tmp_path / "failures.jsonl"
    result = {
        "status": "failed",
        "source_index": 4,
        "Count": 2,
        "B": 3,
        "M": 64,
        "N": 32,
        "K": 128,
        "platform_key": "nvidia-h20",
        "dtype_key": "bf16-bf16-bf16",
        "error": "fixture failure",
    }

    counts = mod._append_collection_rows(
        data_path,
        failure_path,
        [result],
        FakeVariant(),
        ["B", "M", "N", "K"],
        1,
    )

    row = json.loads(failure_path.read_text())
    assert counts == (0, 0, 1)
    assert row["workload"]["dimensions"] == {
        "B": 3,
        "M": 64,
        "N": 32,
        "K": 128,
    }
    assert "stride_am" not in row["workload"]["dimensions"]
    assert not data_path.read_text()


def test_collection_failure_reports_missing_workload_dimensions(tmp_path):
    """Reject incomplete configured shape identities before model normalization."""
    mod = load_path(TRAIN_PATH, "flag_gems_flagtune_train_missing_shape")
    data_path = tmp_path / "data.jsonl"
    failure_path = tmp_path / "failures.jsonl"
    result = {
        "status": "ok",
        "source_index": 5,
        "Count": 1,
        "B": 1,
        "M": 64,
        "N": 32,
        "platform_key": "nvidia-h20",
        "dtype_key": "bf16-bf16-bf16",
        "config_timings": [],
    }

    counts = mod._append_collection_rows(
        data_path,
        failure_path,
        [result],
        FakeVariant(),
        ["B", "M", "N", "K"],
        0,
    )

    row = json.loads(failure_path.read_text())
    assert counts == (0, 0, 1)
    assert row["collection_error"] == "missing workload dimensions: K"
    assert row["workload"]["dimensions"]["K"] is None
    assert not data_path.read_text()


class FakeConfig:
    """Mimic the transport method exposed by a Triton Config object."""

    def all_kwargs(self):
        """Return compile-time and launch-time fields as one mapping."""
        return {"BLOCK_M": 16, "num_warps": 4}


def test_generic_config_timing_serialization_uses_triton_quantile_order():
    """Map LibTuner's p50/p20/p80 samples and null non-finite values."""
    load_path(TRAIN_PATH, "flag_gems_flagtune_train_for_mm")
    mod = importlib.import_module("flag_gems.flagtune.runtime.executor")

    rows = mod._serialize_config_timings({FakeConfig(): [1.2, 1.0, float("inf")]})

    assert rows == [
        {
            "config": {"BLOCK_M": 16, "num_warps": 4},
            "latency_ms": 1.2,
            "latency_p50_ms": 1.2,
            "latency_p20_ms": 1.0,
            "latency_p80_ms": None,
            "status": "ok",
        }
    ]


def test_generic_config_progress_interval_handles_environment_values(monkeypatch):
    """Interpret positive intervals and safely disable malformed values."""
    load_path(TRAIN_PATH, "flag_gems_flagtune_train_for_progress_env")
    mod = importlib.import_module("flag_gems.flagtune.runtime.executor")

    monkeypatch.delenv("FLAGTUNE_TRAIN_PROGRESS_INTERVAL", raising=False)
    assert mod._progress_interval() == 0
    monkeypatch.setenv("FLAGTUNE_TRAIN_PROGRESS_INTERVAL", "17")
    assert mod._progress_interval() == 17
    monkeypatch.setenv("FLAGTUNE_TRAIN_PROGRESS_INTERVAL", "-4")
    assert mod._progress_interval() == 0
    monkeypatch.setenv("FLAGTUNE_TRAIN_PROGRESS_INTERVAL", "invalid")
    assert mod._progress_interval() == 0


def test_worker_log_forwarding_buffers_partial_lines(tmp_path, capsys):
    """Forward each complete worker line once and flush a final partial line."""
    load_path(TRAIN_PATH, "flag_gems_flagtune_train_for_log_forwarding")
    mod = importlib.import_module("flag_gems.flagtune.collection.scheduler")
    first = tmp_path / "worker_0.log"
    second = tmp_path / "worker_1.log"
    first.write_text("started\npart", encoding="utf-8")
    second.write_text("ready\n", encoding="utf-8")
    states = [(0, ""), (0, "")]

    mod._forward_worker_log_updates([first, second], states)
    initial = capsys.readouterr().out
    assert initial.splitlines() == [
        "[benchmark worker 0] started",
        "[benchmark worker 1] ready",
    ]

    with first.open("a", encoding="utf-8") as handle:
        handle.write("ial\ntail")
    mod._forward_worker_log_updates([first, second], states)
    update = capsys.readouterr().out
    assert update.splitlines() == ["[benchmark worker 0] partial"]

    mod._forward_worker_log_updates([first, second], states, final=True)
    final = capsys.readouterr().out
    assert final.splitlines() == ["[benchmark worker 0] tail"]


def test_worker_environment_contains_device_and_database_overrides():
    """Keep tuning behavior out of process-global worker environment state."""
    load_path(TRAIN_PATH, "flag_gems_flagtune_train_for_worker_environment")
    mod = importlib.import_module("flag_gems.flagtune.collection.scheduler")

    class FakeRuntime:
        @staticmethod
        def apply_worker_visibility(environment, token):
            environment["TEST_VISIBLE_DEVICES"] = token

    runtime = FakeRuntime()
    environment = mod._worker_environment(runtime, "5", "sqlite:///training.db")

    assert environment["TEST_VISIBLE_DEVICES"] == "5"
    assert environment["FLAGGEMS_DB_URL"] == "sqlite:///training.db"

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

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from benchmark import base
from benchmark import conftest as benchmark_conftest
from benchmark import consts

SKIP_REASON = "aten::unique_dim falls back to CPU on Ascend"


def _marker(*args, **kwargs):
    return SimpleNamespace(args=args, kwargs=kwargs)


@pytest.mark.parametrize("vendors", ["ascend", ("metax", "ascend")])
def test_skip_native_marker_matches_vendor(vendors):
    marker = _marker(vendors=vendors, reason=f"  {SKIP_REASON}  ")

    reason = benchmark_conftest._get_native_baseline_skip_reason(marker, "ascend")

    assert reason == SKIP_REASON


def test_skip_native_marker_ignores_other_vendors():
    marker = _marker(vendors=("metax",), reason=SKIP_REASON)

    reason = benchmark_conftest._get_native_baseline_skip_reason(marker, "ascend")

    assert reason is None


@pytest.mark.parametrize(
    "marker",
    [
        _marker("ascend", reason=SKIP_REASON),
        _marker(reason=SKIP_REASON),
        _marker(vendors=(), reason=SKIP_REASON),
        _marker(vendors=("ascend",), reason=""),
        _marker(vendors=("ascend",), reason=SKIP_REASON, typo=True),
    ],
)
def test_skip_native_marker_rejects_invalid_configuration(marker):
    with pytest.raises(pytest.UsageError):
        benchmark_conftest._get_native_baseline_skip_reason(marker, "ascend")


class _CollectionNode:
    def __init__(self, marker):
        self.own_markers = [marker]


class _CollectionItem(_CollectionNode):
    def get_closest_marker(self, name):
        return self.own_markers[0] if self.own_markers else None

    def listchain(self):
        return [self]


@pytest.mark.parametrize(
    ("vendors", "remaining_markers"),
    [
        (("ascend",), 1),
        (("metax",), 0),
    ],
)
def test_inactive_native_marker_is_removed_before_pytest_selection(
    vendors, remaining_markers
):
    item = _CollectionItem(_marker(vendors=vendors, reason=SKIP_REASON))

    benchmark_conftest._deactivate_inactive_native_marker(item, "ascend")

    assert len(item.own_markers) == remaining_markers


def test_invalid_native_marker_remains_for_setup_error():
    marker = _marker(vendors=("metax",), reason="")
    item = _CollectionItem(marker)

    benchmark_conftest._deactivate_inactive_native_marker(item, "ascend")

    assert item.own_markers == [marker]


def _native_op(inp):
    return inp


def _gems_op(inp):
    return inp


class _StubBenchmark(base.Benchmark):
    def __init__(self):
        super().__init__(
            op_name="stub",
            torch_op=_native_op,
            gems_op=_gems_op,
            dtypes=[torch.float32],
        )
        self.latency_calls = []
        self.gbps_latencies = []

    def init_user_config(self):
        self.to_bench_dtypes = [torch.float32]
        self.to_bench_metrics = ["latency_base", "latency", "speedup", "gbps"]

    def get_input_iter(self, dtype):
        yield torch.ones(1, dtype=dtype),

    def get_latency(self, op, *args, **kwargs):
        self.latency_calls.append(op)
        return 4.0 if op is _native_op else 2.0

    def get_gbps(self, args, latency=None):
        self.gbps_latencies.append(latency)
        return 8.0


@pytest.mark.parametrize("skip_native", [False, True])
def test_benchmark_native_policy_preserves_gems_metrics(
    monkeypatch, capsys, skip_native
):
    reason = SKIP_REASON if skip_native else None
    config = SimpleNamespace(
        query=False,
        skip_native=skip_native,
        native_baseline_skip_reason=reason,
        bench_level=consts.BenchLevel.CORE,
        mode=consts.BenchMode.KERNEL,
    )
    recorded = []
    monkeypatch.setattr(base, "Config", config)
    monkeypatch.setattr(base, "update_result", lambda op, data: recorded.append(data))
    monkeypatch.setattr(base, "emit_record_logger", lambda message: None)

    benchmark = _StubBenchmark()
    benchmark.run()

    metric = recorded[0]["result"][0]
    assert metric["latency"] == 2.0
    assert metric["gbps"] == 8.0
    if skip_native:
        assert benchmark.latency_calls == [_gems_op]
        assert benchmark.gbps_latencies == [2.0]
        assert metric["latency_base"] is None
        assert metric["gbps_base"] is None
        assert metric["speedup"] is None
        assert recorded[0]["native_baseline_skip_reason"] == SKIP_REASON
        assert f"Native baseline: N/A ({SKIP_REASON})" in capsys.readouterr().out
    else:
        assert benchmark.latency_calls == [_native_op, _gems_op]
        assert benchmark.gbps_latencies == [4.0, 2.0]
        assert metric["latency_base"] == 4.0
        assert metric["gbps_base"] == 8.0
        assert metric["speedup"] == 2.0
        assert "native_baseline_skip_reason" not in recorded[0]
        assert "Native baseline: N/A" not in capsys.readouterr().out


def test_unmarked_result_serialization_keeps_legacy_schema():
    metric = consts.BenchmarkMetrics(
        shape_detail=[torch.Size([64, 64])],
        latency_base=4.0,
        latency=2.0,
        speedup=2.0,
    )
    result = consts.BenchmarkResult(
        op_name="stub",
        dtype="torch.float32",
        mode="kernel",
        level="core",
        result=[metric],
    )

    serialized = json.loads(result.to_json())

    assert list(serialized) == ["op_name", "dtype", "mode", "level", "result"]
    assert "native_baseline_skip_reason" not in serialized


def _load_run_tests_module(monkeypatch):
    tools_dir = Path(__file__).parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools_dir))
    monkeypatch.delitem(sys.modules, "consts", raising=False)
    spec = importlib.util.spec_from_file_location(
        "flag_gems_run_tests", tools_dir / "run_tests.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unmarked_result_parsing_preserves_legacy_overwrite(monkeypatch, tmp_path):
    raw_result = {
        "abs": {
            "result": "passed",
            "details": [
                {
                    "dtype": "torch.float16",
                    "result": [
                        {
                            "shape_detail": [[64, 64]],
                            "latency_base": 4.0,
                            "latency": 2.0,
                            "speedup": 2.0,
                        }
                    ],
                },
                {
                    "dtype": "torch.float16",
                    "result": [
                        {
                            "shape_detail": [[128, 128]],
                            "latency_base": 9.0,
                            "latency": 3.0,
                            "speedup": 3.0,
                        }
                    ],
                },
            ],
        }
    }
    raw_path = tmp_path / "performance_result.json"
    raw_path.write_text(json.dumps(raw_result))

    run_tests = _load_run_tests_module(monkeypatch)
    parsed = run_tests.parse_perf_data("abs", raw_path)

    assert parsed == {
        "status": "Passed",
        "data": {
            "fp16": {
                "result": "OK",
                "details": {"[[128,128]]": {"base": 9.0, "gems": 3.0, "speedup": 3.0}},
                "speedup": 3.0,
            }
        },
        "test_case": "Unknown",
    }


def test_native_skip_result_parsing_and_markdown(monkeypatch, tmp_path):
    raw_result = {
        "unique_dim": {
            "result": "passed",
            "reason": None,
            "test_case": "benchmark/test_unique_dim.py::test_unique_dim_dim0",
            "details": [
                {
                    "op_name": "unique_dim",
                    "dtype": "torch.int16",
                    "mode": "kernel",
                    "level": "core",
                    "native_baseline_skip_reason": SKIP_REASON,
                    "result": [
                        {
                            "shape_detail": [[64, 64]],
                            "latency_base": None,
                            "latency": 1.25,
                            "gbps_base": None,
                            "gbps": 3.5,
                            "speedup": None,
                            "error_msg": None,
                        }
                    ],
                },
                {
                    "op_name": "unique_dim",
                    "dtype": "torch.int16",
                    "mode": "kernel",
                    "level": "core",
                    "native_baseline_skip_reason": SKIP_REASON,
                    "result": [
                        {
                            "shape_detail": [[256, 256]],
                            "latency_base": None,
                            "latency": 2.5,
                            "gbps_base": None,
                            "gbps": 7.0,
                            "speedup": None,
                            "error_msg": None,
                        }
                    ],
                },
            ],
        }
    }
    raw_path = tmp_path / "performance_result.json"
    raw_path.write_text(json.dumps(raw_result))

    run_tests = _load_run_tests_module(monkeypatch)
    parsed = run_tests.parse_perf_data("unique_dim", raw_path)

    assert parsed["status"] == "Passed"
    assert parsed["native_baseline_skip_reason"] == SKIP_REASON
    assert parsed["data"]["int16"]["speedup"] is None
    details = list(parsed["data"]["int16"]["details"].values())
    assert details == [
        {"base": None, "gems": 1.25, "speedup": None},
        {"base": None, "gems": 2.5, "speedup": None},
    ]

    summary = {
        "timestamp": "2026-08-12 00:00:00",
        "env": {},
        "result": {
            "unique_dim": {
                "accuracy": {
                    "status": "Passed",
                    "duration": 0,
                    "total": 0,
                    "passed": 0,
                    "skipped": 0,
                    "failed": 0,
                },
                "performance": {"duration": 1.0, **parsed},
            }
        },
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    psum_text = Path(__file__).parents[1] / "tools" / "psum_text"
    completed = subprocess.run(
        [
            sys.executable,
            str(psum_text),
            "--format",
            "markdown",
            "--single",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert f"Native baseline: N/A ({SKIP_REASON})" in completed.stdout
    assert "| int16 | **N/A** |" in completed.stdout
    assert "| |    N/A |    N/A |  1.250 |" in completed.stdout

    compare_dir = tmp_path / "compare"
    before_dir = compare_dir / "before"
    after_dir = compare_dir / "after"
    before_dir.mkdir(parents=True)
    after_dir.mkdir()
    before_summary = json.loads(json.dumps(summary))
    before_performance = before_summary["result"]["unique_dim"]["performance"]
    before_performance.pop("native_baseline_skip_reason")
    before_dtype = before_performance["data"]["int16"]
    before_dtype["speedup"] = 1.0
    for detail in before_dtype["details"].values():
        detail["base"] = detail["gems"]
        detail["speedup"] = 1.0
    (before_dir / "summary.json").write_text(json.dumps(before_summary))
    after_dtype = summary["result"]["unique_dim"]["performance"]["data"]["int16"]
    after_dtype["details"]["[[512,512]]"] = {
        "base": None,
        "gems": 3.0,
        "speedup": None,
    }
    (after_dir / "summary.json").write_text(json.dumps(summary))

    compared = subprocess.run(
        [
            sys.executable,
            str(psum_text),
            "--format",
            "markdown",
            "--single",
            "--compare",
            str(compare_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "| int16 | **1.000 -> N/A** |" in compared.stdout
    assert "color:Red'>**1.000 -> N/A**" not in compared.stdout
    assert "color:Green'>**1.000 -> N/A**" not in compared.stdout
    assert "| | 0 -> N/A | 0 -> N/A | 0 -> 3.000 | `[[512,512]]` |" in compared.stdout


def test_unmarked_markdown_compare_preserves_legacy_zero_formatting(tmp_path):
    accuracy = {
        "status": "Passed",
        "duration": 0,
        "total": 0,
        "passed": 0,
        "skipped": 0,
        "failed": 0,
    }
    before_performance = {
        "status": "Passed",
        "duration": 1.0,
        "data": {
            "fp16": {
                "result": "OK",
                "speedup": 2.0,
                "details": {"existing": {"base": 4.0, "gems": 2.0, "speedup": 2.0}},
            }
        },
    }
    after_performance = json.loads(json.dumps(before_performance))
    after_performance["data"]["fp16"]["details"]["new-shape"] = {
        "base": 6.0,
        "gems": 2.0,
        "speedup": 3.0,
    }
    after_performance["data"]["fp32"] = {
        "result": "OK",
        "speedup": 4.0,
        "details": {"new-dtype": {"base": 8.0, "gems": 2.0, "speedup": 4.0}},
    }

    for name, performance in (
        ("before", before_performance),
        ("after", after_performance),
    ):
        result_dir = tmp_path / name
        result_dir.mkdir()
        result = {
            "timestamp": "2026-08-12 00:00:00",
            "env": {},
            "result": {
                "abs": {
                    "accuracy": accuracy,
                    "performance": performance,
                }
            },
        }
        (result_dir / "summary.json").write_text(json.dumps(result))

    psum_text = Path(__file__).parents[1] / "tools" / "psum_text"
    completed = subprocess.run(
        [
            sys.executable,
            str(psum_text),
            "--format",
            "markdown",
            "--single",
            "--compare",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (
        "| | 0 -> 3.000 | 0 -> 6.000 | 0 -> 2.000 | `new-shape` |" in completed.stdout
    )
    assert "<span style='color:Green'>****0 -> 4.000****</span>" in completed.stdout
    assert (
        "| | 0 -> 4.000 | 0 -> 8.000 | 0 -> 2.000 | `new-dtype` |" in completed.stdout
    )

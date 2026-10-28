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

"""Benchmark integration with real CPU tensors and deterministic test timers."""

import shlex
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from benchmark import base, conftest
from benchmark.baseline import BaselineData, describe_call
from benchmark.consts import BenchmarkResult
from tools import run_tests


@pytest.mark.parametrize("skip_native", [False, True])
@pytest.mark.parametrize("matched", [False, True])
def test_measurement_preserves_native_execution_and_skip_contract(
    monkeypatch, skip_native, matched
):
    config = conftest.BenchConfig()
    config.skip_native = skip_native
    inputs = (torch.ones(4, 8),)
    shape = "[4, 8]" if matched else "[8, 4]"
    config.base_data = BaselineData(
        {"example": {"shapes": {shape: {"torch.float32": {"latency_ms": 8.0}}}}},
        "synthetic.json",
    )
    monkeypatch.setattr(base, "Config", config)
    monkeypatch.setattr(
        base, "torch_device_fn", SimpleNamespace(get_device_name=lambda: "NVIDIA H20")
    )
    reference = lambda x: x
    candidate = lambda x: x
    bench = base.Benchmark("example", torch_op=reference, gems_op=candidate)
    bench.to_bench_metrics = ["latency_base", "latency", "speedup"]
    monkeypatch.setattr(
        bench, "_candidate_call", lambda: (candidate, nullcontext(), False)
    )
    calls = []

    def latency(op, *args, **kwargs):
        calls.append(op)
        return 6.0 if op is reference else 2.0

    monkeypatch.setattr(bench, "get_latency", latency)
    metric = bench._measure_input(inputs, dtype=torch.float32)
    assert calls == ([candidate] if skip_native else [reference, candidate])
    assert metric.latency == 2.0
    assert metric.latency_base == (None if skip_native else 6.0)
    assert metric.speedup == (None if skip_native else pytest.approx(3.0))
    assert metric.compared_speedup == (4.0 if matched else None)
    report = BenchmarkResult("example", "torch.float32", "kernel", "core", [metric])
    assert ("vs Base" in str(report)) == matched


def test_named_binding_handles_default_parameters(monkeypatch):
    config = conftest.BenchConfig()
    x = torch.ones(4, 8)
    config.base_data = BaselineData(
        {
            "example": {
                "shapes": {
                    "semantic-key": {
                        "torch.float32": {
                            "latency_ms": 4,
                            "config": {
                                "inputs": {
                                    "x": {"shape": [4, 8], "dtype": "torch.float32"},
                                    "dim": {"scalar": 1},
                                }
                            },
                        }
                    }
                }
            }
        },
        "synthetic.json",
    )
    monkeypatch.setattr(base, "Config", config)

    def native(x, dim=1):
        return x

    bench = base.Benchmark("example", torch_op=native, gems_op=native)
    bench.to_bench_metrics = ["latency_base", "latency", "speedup"]
    monkeypatch.setattr(
        bench, "_candidate_call", lambda: (native, nullcontext(), False)
    )
    monkeypatch.setattr(bench, "get_latency", lambda *a, **kw: 2)
    metric = bench._measure_input((x,), dtype=torch.float32)
    assert metric.compared_speedup == 2


def test_no_baseline_does_not_describe_inputs_or_query_device(monkeypatch):
    config = conftest.BenchConfig()
    monkeypatch.setattr(base, "Config", config)

    def forbidden(*a, **kw):
        raise AssertionError("optional baseline path must stay disabled")

    monkeypatch.setattr(base, "describe_call", forbidden)
    monkeypatch.setattr(
        base, "torch_device_fn", SimpleNamespace(get_device_name=forbidden)
    )
    native = lambda x: x
    bench = base.Benchmark("example", torch_op=native, gems_op=native)
    bench.to_bench_metrics = ["latency_base", "latency", "speedup"]
    monkeypatch.setattr(
        bench, "_candidate_call", lambda: (native, nullcontext(), False)
    )
    monkeypatch.setattr(bench, "get_latency", lambda *a, **kw: 2)
    metric = bench._measure_input((torch.ones(4),), dtype=torch.float32)
    assert metric.speedup == pytest.approx(1) and metric.compared_speedup is None


def test_real_tensor_descriptors_keep_layout_nested_roles_and_mixed_dtypes():
    x = torch.ones(4, 8).t()
    call = describe_call(([x, torch.ones(2, dtype=torch.int64)],), {"dim": 0})
    assert call["args"][0][0]["stride"] == [1, 8]
    assert call["args"][0][1]["dtype"] == "torch.int64"
    assert call["kwargs"] == {"dim": 0}


def test_pytest_option_loads_baseline_and_can_reset_to_disabled(tmp_path, monkeypatch):
    import json

    from _pytest.config.argparsing import Parser

    path = tmp_path / "baseline with spaces.json"
    path.write_text(json.dumps({"example": {"shapes": {}}}))
    parser = Parser(_ispytest=True)
    conftest.pytest_addoption(parser)
    monkeypatch.setattr(conftest, "Config", conftest.Config)
    monkeypatch.setattr(conftest, "REPORT_FILE", conftest.REPORT_FILE)
    monkeypatch.setattr(conftest, "apply_overrides_from_args", lambda _: None)
    for argv in [[f"--base-data={path}"], []]:
        options = parser.parse(argv)
        config = SimpleNamespace(
            option=options,
            addinivalue_line=lambda *a: None,
            getini=lambda _: [],
            getoption=lambda key: getattr(options, key.lstrip("-").replace("-", "_")),
            hook=SimpleNamespace(pytest_flaggems_profile_scope=lambda **kw: None),
        )
        conftest.pytest_configure(config)
        if argv:
            assert conftest.Config.base_data.source == str(path)
        else:
            assert conftest.Config.base_data is None


def test_scheduler_passes_absolute_baseline_path_as_one_shell_argument(
    tmp_path, monkeypatch
):
    commands = []
    monkeypatch.setattr(run_tests, "ROOT", tmp_path)
    monkeypatch.setattr(
        run_tests,
        "CFG",
        SimpleNamespace(output_dir=tmp_path, base_data="/tmp/a b.json"),
    )
    monkeypatch.setattr(run_tests, "get_env", lambda _: {})
    monkeypatch.setattr(run_tests, "op_marker", lambda op: op)
    monkeypatch.setattr(
        run_tests, "run_cmd", lambda op, cmd, **kw: commands.append(cmd) or 0
    )
    run_tests.run_benchmark_q(0, "reduce")
    assert "--base-data=/tmp/a b.json" in shlex.split(commands[0])


def test_scheduler_imports_shared_report_parser():
    from tools.performance_report import parse_perf_data

    assert run_tests.parse_perf_data is parse_perf_data


@pytest.mark.parametrize("target", ["npu", "mlu"])
def test_factory_benchmark_passes_target_device_to_external_matching(
    monkeypatch, target
):
    config = conftest.BenchConfig()
    config.skip_native = True
    config.base_data = BaselineData(
        {
            "full": {
                "shapes": {
                    "factory-case": {
                        "torch.float32": {
                            "latency_ms": 8.0,
                            "config": {
                                "inputs": {
                                    "args": [],
                                    "kwargs": {
                                        "size": [4, 8],
                                        "fill_value": 3.0,
                                        "dtype": "torch.float32",
                                        "device": "cuda",
                                    },
                                }
                            },
                        }
                    }
                }
            }
        },
        "synthetic.json",
    )
    monkeypatch.setattr(base, "Config", config)
    bench = base.Benchmark("full", torch_op=torch.full, gems_op=torch.full)
    bench.device = target
    bench.to_bench_metrics = ["latency_base", "latency", "speedup"]
    monkeypatch.setattr(
        bench, "_candidate_call", lambda: (torch.full, nullcontext(), False)
    )
    monkeypatch.setattr(bench, "get_latency", lambda *a, **kw: 2.0)
    metric = bench._measure_input(
        (
            {
                "size": (4, 8),
                "fill_value": 3.0,
                "dtype": torch.float32,
                "device": target,
            },
        ),
        dtype=torch.float32,
    )
    assert metric.compared_speedup == 4.0
    assert metric.latency_base is None


def test_device_objects_are_described_without_allocating_on_an_accelerator():
    call = describe_call(
        (), {"device": torch.device("cuda:0"), "aux_device": torch.device("cpu")}
    )
    assert call["kwargs"] == {"device": "cuda:0", "aux_device": "cpu"}


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("case_mode", [False, True])
def test_argsort_subclass_accepts_dtype_in_both_runners(
    monkeypatch, enabled, case_mode
):
    from benchmark.test_argsort import ArgsortBenchmark

    config = conftest.BenchConfig()
    inputs = (torch.tensor([[3.0, 1.0, 2.0]]), {"dim": -1, "descending": False})
    args, kwargs = (inputs[0],), inputs[1]
    if enabled:
        config.base_data = BaselineData(
            {
                "argsort": {
                    "shapes": {
                        "case": {
                            "torch.float32": {
                                "latency_ms": 8,
                                "config": {"inputs": describe_call(args, kwargs)},
                            }
                        }
                    }
                }
            },
            "synthetic.json",
        )
    monkeypatch.setattr(base, "Config", config)
    candidate = lambda *a, **kw: torch.argsort(*a, stable=True, **kw)
    bench = ArgsortBenchmark(
        op_name="argsort",
        torch_op=torch.argsort,
        gems_op=candidate,
        input_fn=lambda *a: iter(()),
    )
    bench.to_bench_dtypes = [torch.float32]
    bench.to_bench_metrics = ["latency_base", "latency", "speedup"]
    monkeypatch.setattr(bench, "get_latency", lambda *a, **kw: 2.0)
    monkeypatch.setattr(bench, "_emit_result", lambda dtype, metrics: metrics)
    if case_mode:
        case = SimpleNamespace(case_id="argsort-case", dtype=torch.float32)
        monkeypatch.setattr(bench, "_collect_cases", lambda: [case])
        monkeypatch.setattr(bench, "build_inputs", lambda _: inputs)
        results = bench._run_cases(None)
    else:
        monkeypatch.setattr(bench, "get_input_iter", lambda _: iter([inputs]))
        results = bench._run_legacy()
    metric = results[0][0]
    assert metric.accuracy == 1
    assert metric.compared_speedup == (4 if enabled else None)
    assert metric.case_id == ("argsort-case" if case_mode else None)


def _parallel_blas_fixture(monkeypatch, tmp_path, enabled):
    import json

    from benchmark import test_blas_perf_parallel as blas

    config = conftest.BenchConfig()
    inputs = (torch.ones(2, 3), torch.ones(3, 4))
    path = tmp_path / "baseline with spaces.json"
    path.write_text(
        json.dumps(
            {
                "mm": {
                    "shapes": {
                        "case": {
                            "torch.float32": {
                                "latency_ms": 8,
                                "config": {"inputs": describe_call(inputs, {})},
                            }
                        }
                    }
                }
            }
        )
    )
    config.base_data = BaselineData.load(path) if enabled else None
    config.shape_file = str(tmp_path / "shapes.yaml")
    (tmp_path / "shapes.yaml").write_text("mm:\n  shapes: [[2, 3, 4]]\n")
    monkeypatch.setattr(base, "Config", config)
    monkeypatch.setattr(blas, "Config", config)
    native, candidate = lambda *a: None, lambda *a: None
    bench = blas.ParallelBlasBenchmark(
        "mm", torch_op=native, gems_op=candidate, input_fn=lambda *a: iter(())
    )
    bench.to_bench_dtypes = [torch.float32]
    bench.to_bench_metrics = ["latency_base", "latency", "speedup"]
    bench.shapes = [(2, 3, 4)]
    monkeypatch.setattr(bench, "init_user_config", lambda: None)
    monkeypatch.setattr(bench, "get_input_iter", lambda _: iter([inputs]))
    monkeypatch.setattr(
        bench, "get_latency", lambda op, *a, **kw: 4 if op is native else 2
    )
    monkeypatch.setattr(blas, "emit_record_logger", lambda _: None)
    return blas, config, bench, path


@pytest.mark.parametrize("worker", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_blas_serial_and_worker_run_apply_external_baseline(
    monkeypatch, tmp_path, enabled, worker
):
    import pickle

    blas, config, bench, _ = _parallel_blas_fixture(monkeypatch, tmp_path, enabled)
    config.parallel = 1 if worker else 0

    def forbidden_parallel_dispatch(*args):
        raise AssertionError("a worker must not launch more workers")

    monkeypatch.setattr(bench, "_run_parallel_dtype", forbidden_parallel_dispatch)
    result_path = tmp_path / "worker.pkl"
    if worker:
        monkeypatch.setenv(blas.PARALLEL_WORKER_ENV, "1")
    else:
        monkeypatch.delenv(blas.PARALLEL_WORKER_ENV, raising=False)
    monkeypatch.setenv(blas.PARALLEL_RESULT_FILE_ENV, str(result_path))
    bench.run()
    with result_path.open("rb") as handle:
        metric = pickle.load(handle).result[0]
    assert metric.speedup == 2
    assert metric.compared_speedup == (4 if enabled else None)


@pytest.mark.parametrize("enabled", [False, True])
def test_blas_subprocess_forwards_baseline_and_preserves_worker_result(
    monkeypatch, tmp_path, enabled
):

    blas, config, bench, path = _parallel_blas_fixture(monkeypatch, tmp_path, enabled)
    monkeypatch.chdir(tmp_path)
    if enabled:
        config.base_data.source = path.name

    def subprocess_run(cmd, *, env, **kwargs):
        if enabled:
            assert cmd[cmd.index("--base-data") + 1] == str(path.resolve())
        else:
            assert "--base-data" not in cmd
        with monkeypatch.context() as worker:
            worker.setenv(blas.PARALLEL_WORKER_ENV, env[blas.PARALLEL_WORKER_ENV])
            worker.setenv(
                blas.PARALLEL_RESULT_FILE_ENV, env[blas.PARALLEL_RESULT_FILE_ENV]
            )
            bench.run()
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(blas.subprocess, "run", subprocess_run)
    result = bench._run_parallel_worker_subprocess(
        "benchmark/test_blas_perf_parallel.py::test_mm", [(0, (2, 3, 4))], 0, "float32"
    )
    assert result["returncode"] == 0
    metric = result["result_payload"].result[0]
    assert metric.speedup == 2
    assert metric.compared_speedup == (4 if enabled else None)


def test_blas_paired_inputs_match_native_abi_before_timing(monkeypatch, tmp_path):
    blas, config, template, _ = _parallel_blas_fixture(monkeypatch, tmp_path, True)
    bench = object.__new__(blas.ParallelW8A8BlockFP8DeepGemmBenchmark)
    bench.__dict__.update(template.__dict__)
    native_inputs = next(template.get_input_iter(torch.float32))
    gems_inputs = (torch.ones(9),)
    captured = []

    def measure(input_item):
        captured.append(input_item)
        # A native timer can resize inputs. Matching must use the original ABI.
        native_inputs[0].resize_(6)
        return base.BenchmarkMetrics(latency_base=4, latency=2, speedup=2)

    monkeypatch.setattr(bench, "_build_metric_from_input", measure)
    metric = bench._run_inputs([(native_inputs, gems_inputs)], dtype=torch.float32)[0]
    assert captured == [(native_inputs, gems_inputs)]
    assert metric.compared_speedup == 4


def test_blas_external_miss_retains_native_result(monkeypatch, tmp_path):
    _, config, bench, _ = _parallel_blas_fixture(monkeypatch, tmp_path, True)
    config.base_data.data["mm"]["shapes"]["case"]["torch.float32"]["config"]["inputs"][
        "args"
    ][1]["shape"] = [4, 3]
    metric = bench._run_inputs(
        bench.get_input_iter(torch.float32), dtype=torch.float32
    )[0]
    assert metric.speedup == 2
    assert metric.compared_speedup is None
    assert metric.base_miss_reason == "shape_or_dtype_missing"


@pytest.mark.parametrize(
    "recorded", [{}, {"x": {"shape": [4, 8], "dtype": "torch.float32"}}]
)
def test_named_metadata_missing_default_parameter_keeps_native_ratio(
    monkeypatch, recorded
):
    config = conftest.BenchConfig()
    config.base_data = BaselineData(
        {
            "example": {
                "shapes": {
                    "case": {
                        "torch.float32": {
                            "latency_ms": 8,
                            "config": {"inputs": recorded},
                        }
                    }
                }
            }
        },
        "synthetic.json",
    )
    monkeypatch.setattr(base, "Config", config)

    def native(x, dim=1):
        return x

    candidate = lambda x: x
    bench = base.Benchmark("example", torch_op=native, gems_op=candidate)
    bench.to_bench_metrics = ["latency_base", "latency", "speedup"]
    monkeypatch.setattr(
        bench, "get_latency", lambda op, *a, **kw: 6 if op is native else 2
    )
    metric = bench._measure_input((torch.ones(4, 8),), dtype=torch.float32)
    assert metric.speedup == pytest.approx(3)
    assert metric.compared_speedup is None
    assert metric.base_miss_reason == "shape_or_dtype_missing"

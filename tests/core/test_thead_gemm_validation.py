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

"""Regression checks for platform-scoped GEMM validation.

Run with --confcutdir=tests/core in a FlagGems runtime environment.
"""

from types import SimpleNamespace

import pytest

from benchmark import base
from benchmark import conftest as benchmark_config
from tests import conftest as accuracy_config


@pytest.mark.parametrize("suite", [accuracy_config, benchmark_config])
@pytest.mark.parametrize("vendor", ["thead", "nvidia", "hygon", "mthreads"])
@pytest.mark.parametrize("op", ["mm", "mv", "addmm", "bmm", "baddbmm", "router_gemm"])
def test_gemm_collection_only_scopes_thead_single_operator(
    monkeypatch, tmp_path, suite, vendor, op
):
    monkeypatch.setattr(suite.flag_gems, "vendor_name", vendor)
    config = SimpleNamespace(getoption=lambda name: op)
    if suite is benchmark_config and op in ("addmm", "baddbmm", "router_gemm"):
        selected = tmp_path / "test_blas_perf_parallel.py"
    else:
        selected = tmp_path / f"test_{op}.py"
    unrelated = tmp_path / "test_unrelated_optional_dependency.py"
    selected.touch()
    unrelated.touch()
    assert suite.pytest_ignore_collect(selected, config) is None
    assert suite.pytest_ignore_collect(unrelated, config) is (
        True if vendor == "thead" else None
    )


@pytest.mark.parametrize("suite", [accuracy_config, benchmark_config])
@pytest.mark.parametrize("expression", ["", "mm or bmm", "not mm", "add"])
def test_other_marker_expressions_keep_normal_collection(
    monkeypatch, tmp_path, suite, expression
):
    monkeypatch.setattr(suite.flag_gems, "vendor_name", "thead")
    config = SimpleNamespace(getoption=lambda name: expression)
    unrelated = tmp_path / "test_unrelated.py"
    unrelated.touch()
    assert suite.pytest_ignore_collect(unrelated, config) is None


@pytest.mark.parametrize("vendor", ["thead", "nvidia", "hygon", "mthreads"])
@pytest.mark.parametrize("op", ["mm", "mm_out", "addmm", "addmm_out", "bmm"])
def test_shape_deduplication_preserves_other_vendors_and_real_batches(
    monkeypatch, vendor, op
):
    monkeypatch.setattr(base, "vendor_name", vendor)
    shapes = [(2, 4096, 4096, 4096), (16, 4096, 4096, 4096), (2, 384, 384, 384)]
    actual = base.deduplicate_thead_gemm_shapes(op, shapes)
    expected = [shapes[0], shapes[2]] if vendor == "thead" and op != "bmm" else shapes
    assert actual == expected


@pytest.mark.parametrize(
    "vendor,op,expected",
    [
        ("thead", "mm", True),
        ("thead", "mv", True),
        ("thead", "addmm", True),
        ("thead", "bmm", True),
        ("thead", "baddbmm", True),
        ("thead", "outer", False),
        ("nvidia", "addmm", False),
        ("hygon", "baddbmm", False),
        ("mthreads", "addmm", False),
    ],
)
def test_thead_gemm_parallel_metrics_are_recorded(monkeypatch, vendor, op, expected):
    from benchmark import test_blas_perf_parallel as parallel

    monkeypatch.setattr(parallel.flag_gems, "vendor_name", vendor)
    monkeypatch.setattr(
        parallel,
        "Config",
        SimpleNamespace(
            query=False,
            parallel=0,
            bench_level=base.consts.BenchLevel.CORE,
            mode=base.consts.BenchMode.KERNEL,
        ),
    )
    monkeypatch.setattr(parallel, "emit_record_logger", lambda message: None)
    monkeypatch.delenv(parallel.PARALLEL_RESULT_FILE_ENV, raising=False)
    monkeypatch.delenv(parallel.PARALLEL_WORKER_ENV, raising=False)
    recorded = []

    class Bench(parallel.ParallelBenchmarkMixin):
        op_name = op
        to_bench_dtypes = ["bfloat16"]

        def init_user_config(self):
            pass

        def get_input_iter(self, dtype):
            return iter(())

        def _run_inputs(self, inputs, **kwargs):
            return [
                parallel.BenchmarkMetrics(
                    latency_base=1.0,
                    latency=1.0,
                    speedup=1.0,
                    shape_detail=[(1, 1)],
                )
            ]

        def _emit_result(self, dtype, metrics):
            recorded.append(dtype)

    Bench().run()
    assert recorded == (["bfloat16"] if expected else [])


@pytest.mark.parametrize("vendor", ["thead", "nvidia", "hygon", "mthreads"])
def test_baddbmm_forward_inputs_only_disable_grad_on_thead(monkeypatch, vendor):
    from benchmark import test_blas_perf_parallel as parallel

    monkeypatch.setattr(parallel.flag_gems, "vendor_name", vendor)
    flags = []

    def randn(*args, **kwargs):
        flags.append(kwargs["requires_grad"])
        return object()

    monkeypatch.setattr(parallel.torch, "randn", randn)
    inputs = next(parallel.baddbmm_input_fn(2, 3, 4, 5, "bfloat16", "cuda", False))
    assert len(inputs) == 3
    assert flags == [vendor != "thead"] * 3

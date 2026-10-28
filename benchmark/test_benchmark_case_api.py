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

import pytest
import torch

from . import base, consts

_NODEID = "benchmark/test_case_api.py::test_case_api"


def _case_id(local_id):
    return f"{_NODEID}::{local_id}"


@pytest.fixture(autouse=True)
def _stable_case_identity(monkeypatch):
    monkeypatch.setattr(base.Config, "current_nodeid", _NODEID)
    monkeypatch.setattr(base.Config, "available_case_ids", set())
    monkeypatch.setattr(base.Config, "executed_case_ids", set())


def _make_benchmark(materialized):
    def input_fn(b, m, n, k, dtype, device, b_column_major):
        materialized.append((b, m, n, k, dtype, device, b_column_major))
        yield (m, n, k)

    benchmark = base.BlasBenchmark(
        op_name="case_api_test",
        input_fn=input_fn,
        torch_op=lambda *args: args,
        dtypes=[torch.float16],
    )
    benchmark.shapes = [(2, 4, 8, 16), (16, 4, 8, 16)]
    benchmark.to_bench_dtypes = [torch.float16]
    return benchmark


def test_list_cases_does_not_materialize_inputs(monkeypatch):
    materialized = []
    benchmark = _make_benchmark(materialized)
    monkeypatch.setattr(benchmark, "init_user_config", lambda: None)
    monkeypatch.setattr(base.Config, "bench_level", consts.BenchLevel.CORE)

    case_list = benchmark.list_cases()

    assert materialized == []
    assert [case.case_id for case in case_list.cases] == [
        _case_id("core::float16::0"),
        _case_id("core::float16::1"),
    ]
    assert [case.shape for case in case_list.cases] == [
        {"b": 2, "m": 4, "n": 8, "k": 16},
        {"b": 16, "m": 4, "n": 8, "k": 16},
    ]
    assert [case.params for case in case_list.cases] == [
        {"b_column_major": False},
        {"b_column_major": False},
    ]
    payload = case_list.to_dict()
    assert payload["schema_version"] == "flaggems.benchmark-case-list/v2"
    assert "nodeid" not in payload
    assert payload["cases"][0] == {
        "case_id": _case_id("core::float16::0"),
        "ordinal": 0,
        "dtype": "torch.float16",
        "shape": {"b": 2, "m": 4, "n": 8, "k": 16},
        "params": {"b_column_major": False},
    }


def test_selected_case_materializes_exactly_once(monkeypatch):
    materialized = []
    benchmark = _make_benchmark(materialized)
    monkeypatch.setattr(base.Config, "bench_level", consts.BenchLevel.CORE)
    monkeypatch.setattr(
        benchmark,
        "_measure_input",
        lambda input, case_id=None: consts.BenchmarkMetrics(case_id=case_id),
    )
    monkeypatch.setattr(
        benchmark, "_emit_result", lambda dtype, metrics: (dtype, metrics)
    )

    results = benchmark._run_cases([_case_id("core::float16::1")])

    assert len(materialized) == 1
    assert materialized[0][0] == 16
    assert results[0][1][0].case_id == _case_id("core::float16::1")


def test_case_selection_ignores_ids_owned_by_other_pytest_nodes(monkeypatch):
    materialized = []
    benchmark = _make_benchmark(materialized)
    monkeypatch.setattr(base.Config, "bench_level", consts.BenchLevel.CORE)

    results = benchmark._run_cases(
        ["benchmark/test_other.py::test_other::core::float16::0"]
    )

    assert results == []
    assert materialized == []
    assert base.Config.available_case_ids == {
        _case_id("core::float16::0"),
        _case_id("core::float16::1"),
    }
    assert base.Config.executed_case_ids == set()


def test_empty_case_selection_runs_nothing(monkeypatch):
    materialized = []
    benchmark = _make_benchmark(materialized)
    monkeypatch.setattr(base.Config, "bench_level", consts.BenchLevel.CORE)

    results = benchmark._run_cases([])

    assert results == []
    assert materialized == []


def test_generic_two_stage_cases_do_not_materialize_during_listing(monkeypatch):
    materialized = []

    def case_fn(shape, dtype):
        del dtype
        yield consts.BenchmarkCasePlan(
            shape={"input": shape},
            params={"dim": -1},
            builder_args=(shape,),
        )

    def build_inputs_fn(plan, dtype, device):
        materialized.append((plan, dtype, device))
        return plan.builder_args[0], plan.params["dim"]

    benchmark = base.GenericBenchmark(
        op_name="generic_case_api_test",
        case_fn=case_fn,
        build_inputs_fn=build_inputs_fn,
        torch_op=lambda *args: args,
        dtypes=[torch.float16],
    )
    benchmark.shapes = [(4, 8), (16, 32)]
    benchmark.to_bench_dtypes = [torch.float16]
    monkeypatch.setattr(benchmark, "init_user_config", lambda: None)
    monkeypatch.setattr(base.Config, "bench_level", consts.BenchLevel.CORE)

    case_list = benchmark.list_cases()

    assert materialized == []
    assert [case.shape for case in case_list.cases] == [
        {"input": (4, 8)},
        {"input": (16, 32)},
    ]
    assert [case.params for case in case_list.cases] == [
        {"dim": -1},
        {"dim": -1},
    ]

    monkeypatch.setattr(
        benchmark,
        "_measure_input",
        lambda input, case_id=None: consts.BenchmarkMetrics(case_id=case_id),
    )
    monkeypatch.setattr(
        benchmark, "_emit_result", lambda dtype, metrics: (dtype, metrics)
    )
    results = benchmark._run_cases([_case_id("core::float16::1")])

    assert len(materialized) == 1
    assert materialized[0][0].builder_args == ((16, 32),)
    assert results[0][1][0].case_id == _case_id("core::float16::1")


def test_generic_two_stage_full_run_preserves_planned_case_order(monkeypatch):
    materialized = []

    def case_fn(shape, dtype):
        del dtype
        for transposed in (False, True):
            yield consts.BenchmarkCasePlan(
                shape={"input": shape},
                params={"transposed": transposed},
                builder_args=(shape, transposed),
            )

    def build_inputs_fn(plan, dtype, device):
        del dtype, device
        materialized.append(plan.builder_args)
        return plan.builder_args

    benchmark = base.GenericBenchmark(
        op_name="generic_case_order_test",
        case_fn=case_fn,
        build_inputs_fn=build_inputs_fn,
        torch_op=lambda *args: args,
        dtypes=[torch.float16],
    )
    benchmark.shapes = [(4, 8), (16, 32)]
    benchmark.to_bench_dtypes = [torch.float16]
    monkeypatch.setattr(base.Config, "bench_level", consts.BenchLevel.CORE)
    monkeypatch.setattr(
        benchmark,
        "_measure_input",
        lambda input, case_id=None: consts.BenchmarkMetrics(case_id=case_id),
    )
    monkeypatch.setattr(
        benchmark, "_emit_result", lambda dtype, metrics: (dtype, metrics)
    )

    results = benchmark._run_cases(None)

    assert materialized == [
        ((4, 8), False),
        ((4, 8), True),
        ((16, 32), False),
        ((16, 32), True),
    ]
    assert [metric.case_id for metric in results[0][1]] == [
        _case_id("core::float16::0"),
        _case_id("core::float16::1"),
        _case_id("core::float16::2"),
        _case_id("core::float16::3"),
    ]


@pytest.mark.parametrize(
    "benchmark_cls",
    [
        base.UnaryReductionBenchmark,
        base.BinaryPointwiseBenchmark,
        base.ScalarBinaryPointwiseBenchmark,
        base.UnaryPointwiseBenchmark,
        base.UnaryPointwiseOutBenchmark,
        base.TexGluForwardBenchmark,
        base.TexGluBackwardBenchmark,
    ],
)
def test_common_shape_families_list_cases_without_tensor_inputs(
    monkeypatch, benchmark_cls
):
    benchmark = benchmark_cls(
        op_name="shape_family_case_api_test",
        torch_op=lambda *args, **kwargs: (args, kwargs),
        dtypes=[torch.float16],
    )
    benchmark.shapes = [(4, 8)]
    benchmark.to_bench_dtypes = [torch.float16]
    monkeypatch.setattr(benchmark, "init_user_config", lambda: None)
    monkeypatch.setattr(base.Config, "bench_level", consts.BenchLevel.CORE)
    monkeypatch.setattr(
        base,
        "generate_tensor_input",
        lambda *args, **kwargs: pytest.fail("listing materialized a tensor"),
    )

    case_list = benchmark.list_cases()

    assert len(case_list.cases) == 1
    assert case_list.cases[0].case_id == _case_id("core::float16::0")


def test_unported_benchmark_keeps_legacy_default_and_rejects_selection(
    monkeypatch,
):
    benchmark = base.GenericBenchmark(
        op_name="legacy_case_api_test",
        input_fn=lambda shape, dtype, device: iter(()),
        torch_op=lambda *args: args,
        dtypes=[torch.float16],
    )
    monkeypatch.setattr(benchmark, "init_user_config", lambda: None)
    monkeypatch.setattr(benchmark, "_run_legacy", lambda: "legacy")
    monkeypatch.setattr(base.Config, "query", False)
    monkeypatch.setattr(base.Config, "list_cases", False)
    monkeypatch.setattr(base.Config, "case_ids", None)

    assert benchmark.run() == "legacy"
    with pytest.raises(ValueError, match="does not support --case-id"):
        benchmark.run(case_ids=[])


def test_blas_subclass_inherits_family_cases_unless_it_replaces_the_loop():
    class StandardBlasSubclass(base.BlasBenchmark):
        pass

    class CustomLoopBlasSubclass(base.BlasBenchmark):
        def get_input_iter(self, dtype):
            yield from ()

    common_kwargs = {
        "op_name": "case_api_subclass_test",
        "input_fn": lambda *args: iter(()),
        "torch_op": lambda *args: args,
        "dtypes": [torch.float16],
    }

    assert StandardBlasSubclass(**common_kwargs).supports_cases()
    assert not CustomLoopBlasSubclass(**common_kwargs).supports_cases()


def test_custom_family_loop_is_supported_after_two_stage_migration(monkeypatch):
    materialized = []

    class CustomUnaryBenchmark(base.UnaryPointwiseBenchmark):
        def get_input_iter(self, dtype):
            pytest.fail("case listing used the legacy tensor loop")
            yield dtype

        def get_case_iter(self, dtype):
            yield self._case_from_plan(
                dtype,
                0,
                consts.BenchmarkCasePlan(
                    shape={"input": (4, 8), "aux": (4, 8)},
                    params={"dim": -1},
                    builder_args=((4, 8),),
                ),
            )

        def build_inputs(self, case):
            materialized.append(case.case_id)
            return case.builder_args[0].builder_args

    benchmark = CustomUnaryBenchmark(
        op_name="custom_family_case_api_test",
        torch_op=lambda *args: args,
        dtypes=[torch.float16],
    )
    benchmark.to_bench_dtypes = [torch.float16]
    monkeypatch.setattr(benchmark, "init_user_config", lambda: None)
    monkeypatch.setattr(base.Config, "bench_level", consts.BenchLevel.CORE)

    case_list = benchmark.list_cases()

    assert benchmark.supports_cases()
    assert materialized == []
    assert [case.case_id for case in case_list.cases] == [_case_id("core::float16::0")]


@pytest.mark.parametrize(
    "benchmark_cls",
    [
        base.UnaryReductionBenchmark,
        base.BinaryPointwiseBenchmark,
        base.ScalarBinaryPointwiseBenchmark,
        base.UnaryPointwiseBenchmark,
        base.UnaryPointwiseOutBenchmark,
        base.TexGluForwardBenchmark,
        base.TexGluBackwardBenchmark,
    ],
)
def test_family_builders_preserve_legacy_inputs(monkeypatch, benchmark_cls):
    benchmark = benchmark_cls(op_name="equivalence", torch_op=lambda *args: args)
    benchmark.device = "cpu"
    benchmark.shapes = [(4, 8), (2, 16)]
    monkeypatch.setattr(base.Config, "bench_level", consts.BenchLevel.CORE)
    torch.manual_seed(42)
    original = list(benchmark.get_input_iter(torch.float32))
    torch.manual_seed(42)
    rebuilt = [
        benchmark.build_inputs(case) for case in benchmark.get_case_iter(torch.float32)
    ]

    def compare(left, right):
        if isinstance(left, torch.Tensor):
            assert left.shape == right.shape and left.dtype == right.dtype
            # Out buffers are uninitialized; compare their metadata separately.
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                assert left[key].shape == right[key].shape
                assert left[key].dtype == right[key].dtype
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for a, b in zip(left, right):
                compare(a, b)
        else:
            assert left == right

    compare(original, rebuilt)


def test_duplicate_generated_ids_fail_during_listing(monkeypatch):
    benchmark = _make_benchmark([])
    monkeypatch.setattr(benchmark, "init_user_config", lambda: None)
    case = next(benchmark.get_case_iter(torch.float16))
    monkeypatch.setattr(benchmark, "get_case_iter", lambda dtype: iter([case, case]))
    with pytest.raises(ValueError, match="duplicate case IDs"):
        benchmark.list_cases()


@pytest.mark.parametrize("count", [0, 2])
def test_blas_factory_must_match_one_planned_case(count):
    benchmark = _make_benchmark([])
    benchmark.input_fn = lambda *args: iter([(1,)] * count)
    case = next(benchmark.get_case_iter(torch.float16))
    with pytest.raises(ValueError, match="no inputs|exactly one"):
        benchmark.build_inputs(case)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"case_fn": lambda *args: iter(())},
        {"build_inputs_fn": lambda *args: ()},
        {
            "input_fn": lambda *args: iter(()),
            "case_fn": lambda *args: iter(()),
            "build_inputs_fn": lambda *args: (),
        },
    ],
)
def test_generic_rejects_incomplete_or_mixed_provider(kwargs):
    with pytest.raises(ValueError):
        base.GenericBenchmark(op_name="invalid", torch_op=lambda *args: args, **kwargs)


def test_listing_run_never_builds_or_measures(monkeypatch, tmp_path):
    shapes = tmp_path / "shapes.yaml"
    shapes.write_text("softmax:\n  shapes: [[4, 8], [8, 16]]\n")
    benchmark = base.UnaryReductionBenchmark(op_name="softmax", torch_op=torch.softmax)
    monkeypatch.setattr(base.Config, "shape_file", str(shapes))
    monkeypatch.setattr(base.Config, "list_cases", True)
    monkeypatch.setattr(base.Config, "bench_level", consts.BenchLevel.CORE)
    monkeypatch.setattr(base.Config, "case_ids", None)
    monkeypatch.setattr(base.Config, "query", False)
    monkeypatch.setattr(base.Config, "user_desired_dtypes", [torch.float32])
    monkeypatch.setattr(
        benchmark, "build_inputs", lambda *args: pytest.fail("input constructed")
    )
    monkeypatch.setattr(
        benchmark, "_measure_input", lambda *args: pytest.fail("timing invoked")
    )
    result = benchmark.run()
    assert len(result.cases) == 2

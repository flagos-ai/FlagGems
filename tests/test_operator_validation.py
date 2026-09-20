"""Regression checks for assertion and stateful measurement boundaries."""

import importlib
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import flag_gems
from flag_gems import testing

from . import accuracy_utils as utils
from . import test_utils as tu


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_fp8_comparison_preserves_dtype_and_rejects_wrong_values(dtype):
    source = torch.tensor([0.0, 1.0, float("nan")]).to(dtype)
    tu.assert_result_equal(source, source.clone())
    tu.assert_result_close(source, source.clone())
    with pytest.raises(AssertionError):
        tu.assert_result_equal(source.float(), source)
    bad = torch.tensor([0.0, 2.0, float("nan")]).to(dtype)
    with pytest.raises(AssertionError):
        tu.assert_result_equal(bad, source)


def test_exact_copy_comparison_rejects_small_error():
    source = torch.ones(10)
    with pytest.raises(AssertionError):
        tu.assert_result_equal(source + 1e-4, source)
    testing.assert_equal(3, 3)
    with pytest.raises(AssertionError):
        testing.assert_equal(3, 4)


def test_reference_is_independent_even_when_no_upcast_or_cpu_copy(monkeypatch):
    monkeypatch.setattr(utils, "TO_CPU", False)
    source = torch.arange(6.0)
    reference = tu.to_reference(source)
    source.add_(100)
    testing.assert_equal(reference, torch.arange(6.0))


@pytest.mark.parametrize(
    "dtype,scenario",
    tu.special_value_cases([torch.float8_e4m3fn, torch.float8_e5m2, torch.bfloat16]),
)
def test_special_scenarios_retain_representable_payload(dtype, scenario):
    x = tu.make_special_input(dtype, scenario).float()
    assert torch.isnan(x).any().item() == (scenario in ("nan", "mixed"))
    assert torch.isinf(x).any().item() == (scenario in ("inf", "mixed"))


@pytest.fixture
def bench_config(monkeypatch):
    from benchmark import base
    from benchmark.consts import BenchMode

    config = SimpleNamespace(mode=BenchMode.OPERATOR, warm_up=0.03, repetition=0.03)
    monkeypatch.setattr(base, "Config", config)
    monkeypatch.setattr(
        base, "torch_device_fn", SimpleNamespace(synchronize=lambda: None)
    )
    return base


def test_missing_direct_candidate_never_falls_back(bench_config):
    bench = bench_config.Benchmark("missing_for_validation", lambda x: x, gems_op=None)
    with pytest.raises(LookupError):
        bench._candidate_context_and_op()


def test_explicit_none_resolves_existing_public_candidate(bench_config, monkeypatch):
    def op(x):
        return x

    monkeypatch.setattr(flag_gems, "validation_op", op, raising=False)
    bench = bench_config.Benchmark("validation_op", lambda x: x, gems_op=None)
    _, resolved = bench._candidate_context_and_op()
    assert resolved is op


def test_stateful_latency_restores_before_every_call_and_each_side(bench_config):
    seen = []
    source = torch.sparse_coo_tensor(
        torch.tensor([[0, 1], [1, 0]]), torch.ones(2), (2, 2)
    )

    def clear(x):
        seen.append((x._nnz(), tuple(x.shape)))
        x.sparse_resize_and_clear_((4, 4), 2, 0)

    bench = bench_config.Benchmark(
        "sparse_resize_and_clear_",
        clear,
        is_inplace=True,
        fresh_inputs=True,
        gems_op=clear,
    )
    assert bench.get_latency(clear, source) > 0
    assert bench.get_latency(clear, source) > 0
    assert len(seen) >= 4
    assert set(seen) == {(2, (2, 2))}
    assert source._nnz() == 2


def test_stateful_profile_restores_outside_each_capture(bench_config, monkeypatch):
    events = []

    def mutate(x):
        events.append(("call", x.item()))
        x.add_(1)

    bench = bench_config.Benchmark(
        "validation_", mutate, is_inplace=True, fresh_inputs=True, gems_op=mutate
    )
    original = bench_config._clone_benchmark_inputs

    def snapshot(x, *args, **kwargs):
        events.append(("prepare", None))
        return original(x, *args, **kwargs)

    monkeypatch.setattr(bench_config, "_clone_benchmark_inputs", snapshot)
    monkeypatch.setattr(
        bench, "_external_profiler_start", lambda: events.append(("start", None))
    )
    monkeypatch.setattr(
        bench, "_external_profiler_stop", lambda: events.append(("stop", None))
    )
    bench._run_candidate_input(
        (torch.zeros(1),), (torch.zeros(1),), warmup=2, iterations=3, profile=True
    )
    assert [v for k, v in events if k == "call"] == [0.0] * 5
    active = False
    for name, _ in events:
        if name == "start":
            active = True
        elif name == "stop":
            active = False
        elif name == "prepare":
            assert not active


def test_stateful_measurement_boundary_excludes_restoration(bench_config, monkeypatch):
    phase = []
    ticks = iter([0.0, 0.001])
    bench_config.Config.warm_up = 0
    bench_config.Config.repetition = 0
    monkeypatch.setattr(
        bench_config.time,
        "perf_counter",
        lambda: (phase.append("clock"), next(ticks))[1],
    )
    original = bench_config._clone_benchmark_inputs

    def snapshot(value, *args, **kwargs):
        phase.append("prepare")
        return original(value, *args, **kwargs)

    monkeypatch.setattr(bench_config, "_clone_benchmark_inputs", snapshot)

    def mutate(x):
        phase.append("call")
        x.add_(1)

    bench = bench_config.Benchmark(
        "validation_", mutate, gems_op=mutate, is_inplace=True, fresh_inputs=True
    )
    result = bench.get_latency(mutate, torch.zeros(1))
    assert result == pytest.approx(1.0)
    start = phase.index("clock")
    assert phase[start:] == ["clock", "call", "clock"]


def test_arithmetic_helper_rejects_candidate_dtype_change():
    with pytest.raises(AssertionError):
        tu.assert_result_close(
            torch.ones(4, dtype=torch.float64), torch.ones(4, dtype=torch.float32)
        )


@pytest.mark.parametrize("unsupported", ["cudagraph", "backward"])
def test_stateful_unsupported_measurement_never_reports_forward_latency(
    bench_config, unsupported
):
    from benchmark.consts import BenchMode

    if unsupported == "cudagraph":
        bench_config.Config.mode = BenchMode.CUDAGRAPH
    bench = bench_config.Benchmark(
        "validation_",
        lambda x: x,
        is_inplace=True,
        fresh_inputs=True,
        is_backward=unsupported == "backward",
    )
    with pytest.raises(ValueError):
        bench.get_latency(lambda x: x, torch.ones(1))


@pytest.mark.parametrize(
    "complex_dtype,real_dtype",
    [
        (torch.complex32, torch.float16),
        (torch.complex64, torch.float32),
        (torch.complex128, torch.float64),
    ],
)
def test_complex_bounds_follow_component_dtype(complex_dtype, real_dtype):
    assert tu.dtype_bounds(complex_dtype) == (
        torch.finfo(real_dtype).min,
        torch.finfo(real_dtype).max,
    )


def test_existing_inplace_benchmark_keeps_legacy_timing(bench_config, monkeypatch):
    from benchmark import base

    bench = base.Benchmark("add_", lambda x: x, is_inplace=True)
    assert not bench.fresh_inputs
    monkeypatch.setattr(base.Config, "mode", base.consts.BenchMode.OPERATOR)
    monkeypatch.setattr(base, "get_iter_count", lambda fn: (1, 1))
    calls = []
    bench.get_latency(lambda x: calls.append(x), torch.ones(1))
    assert len(calls) == 2
    assert calls[0] is calls[1]


def test_reference_retains_view_metadata_and_independent_gradients():
    source = torch.arange(40.0).reshape(5, 8)[1:, ::2].requires_grad_()
    reference = tu.to_reference(source)
    assert reference.stride() == source.stride()
    assert reference.storage_offset() == source.storage_offset()
    assert not torch._C._is_alias_of(reference, source)
    with torch.no_grad():
        source.add_(100)
    torch.testing.assert_close(reference, torch.arange(40.0).reshape(5, 8)[1:, ::2])
    reference.sum().backward()
    assert source.grad is None


@pytest.mark.parametrize(
    "module_path",
    [
        "benchmark/test_col_indices.py",
        "tests/test__choose_qparams_per_tensor.py",
        "tests/test__coalesce.py",
        "tests/test__dimI.py",
        "tests/test__dimV.py",
        "tests/test__dim_arange.py",
        "tests/test__efficientzerotensor.py",
        "tests/test__fw_primal.py",
        "tests/test__has_same_storage_numel.py",
        "tests/test__indices.py",
        "tests/test__make_dual.py",
        "tests/test__make_per_tensor_quantized_tensor.py",
        "tests/test__neg_view.py",
        "tests/test__neg_view_copy.py",
        "tests/test__nested_tensor_size.py",
        "tests/test__nested_tensor_storage_offsets.py",
        "tests/test__nested_tensor_strides.py",
        "tests/test__nnz.py",
        "tests/test__shape_as_tensor.py",
        "tests/test__unpack_dual.py",
        "tests/test__values.py",
        "tests/test__version.py",
        "tests/test_atleast_1d.py",
        "tests/test_atleast_2d.py",
        "tests/test_atleast_3d.py",
        "tests/test_cartesian_prod.py",
        "tests/test_ccol_indices.py",
        "tests/test_ccol_indices_copy.py",
        "tests/test_chain_matmul.py",
        "tests/test_coalesce.py",
        "tests/test_col_indices.py",
        "tests/test_col_indices_copy.py",
        "tests/test_combinations.py",
        "tests/test_copy_sparse_to_sparse_.py",
        "tests/test_crow_indices.py",
        "tests/test_crow_indices_copy.py",
        "tests/test_data.py",
        "tests/test_dense_dim.py",
        "tests/test_diagflat.py",
        "tests/test_dim.py",
        "tests/test_dstack.py",
        "tests/test_flatten_dense_tensors.py",
        "tests/test_slow_conv_transpose3d.py",
        "tests/test_sparse_bsc_tensor.py",
        "tests/test_sparse_bsr_tensor.py",
        "tests/test_sparse_coo_tensor.py",
        "tests/test_sparse_dim.py",
        "tests/test_sparse_mask.py",
        "tests/test_sparse_resize_.py",
        "tests/test_sparse_resize_and_clear_.py",
        "tests/test_adjoint.py",
    ],
)
def test_operator_collection_does_not_probe_runtime(module_path, monkeypatch):
    calls = []

    def reject_probe(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("Runtime probe during test collection")

    path = Path(__file__).resolve().parents[1] / module_path
    operator = path.stem.removeprefix("test_")
    # Parameter generation may use CPU randperm for index lists; operator
    # execution and construction of probe inputs must wait until the test runs.
    for module, name in [
        (torch.ops.aten, operator),
        (tu, "make_input"),
        (torch.testing, "make_tensor"),
        (torch, "tensor"),
        (torch, "zeros"),
        (torch, "ones"),
        (torch, "empty"),
        (torch, "full"),
    ]:
        monkeypatch.setattr(module, name, reject_probe)
    runpy.run_path(str(path), run_name=f"{path.parent.name}._collection_check")
    # A probe that catches the injected error must still fail this check.
    assert not calls


def test_combinations_missing_candidate_cannot_pass_against_reference(monkeypatch):
    from . import test_combinations as cases

    missing = Mock(side_effect=LookupError("candidate missing"))
    monkeypatch.setattr(testing, "resolve_gems_op", missing)
    with pytest.raises(LookupError, match="candidate missing"):
        cases.test_combinations_spec_shapes_value_ranges(
            (4,), ["0", "1"], torch.float32
        )
    missing.assert_called_once()


def test_version_input_error_does_not_retry_with_another_range(monkeypatch):
    from . import test__version as cases

    failed = Mock(side_effect=RuntimeError("input construction failed"))
    monkeypatch.setattr(torch.testing, "make_tensor", failed)
    with pytest.raises(RuntimeError, match="input construction failed"):
        cases._make_value_tensor(torch.int32, (4,), ["0", "1"], "cpu")
    failed.assert_called_once()


def test_sparse_copy_input_error_does_not_use_another_generator(monkeypatch):
    from . import test_copy_sparse_to_sparse_ as cases

    failed = Mock(side_effect=RuntimeError("input construction failed"))
    monkeypatch.setattr(tu, "make_input", failed)
    with pytest.raises(RuntimeError, match="input construction failed"):
        cases._make_values((4,), torch.uint8, value_range=["0", "1"])
    failed.assert_called_once()


@pytest.mark.parametrize(
    "operator", ["copy_sparse_to_sparse_", "sparse_mask", "sparse_resize_"]
)
def test_sparse_storage_operations_reject_small_value_changes(operator):
    from . import test_copy_sparse_to_sparse_ as copy_cases
    from . import test_sparse_mask as mask_cases
    from . import test_sparse_resize_ as resize_cases

    def corrupted(*args, **kwargs):
        result = getattr(torch.ops.aten, operator)(*args, **kwargs)
        result._values().add_(1e-6)
        return result

    checks = {
        "copy_sparse_to_sparse_": lambda: copy_cases.test_copy_sparse_to_sparse_(
            ((4, 5), 2, 3), torch.float32, False
        ),
        "sparse_mask": lambda: mask_cases.test_sparse_mask_value_ranges(
            (4, 5), ["0", "1"], torch.float32
        ),
        "sparse_resize_": lambda: resize_cases.test_sparse_resize_(
            ((4, 5), 2, 3, (6, 5), 2, 0), torch.float32
        ),
    }
    with testing.override_gems_op(operator, corrupted):
        with pytest.raises(AssertionError):
            checks[operator]()


def test_sparse_copy_rejects_changed_coalesced_flag():
    from . import test_copy_sparse_to_sparse_ as cases

    def corrupted(dst, src, non_blocking):
        result = torch.ops.aten.copy_sparse_to_sparse_(dst, src, non_blocking)
        result._coalesced_(not result.is_coalesced())
        return result

    with testing.override_gems_op("copy_sparse_to_sparse_", corrupted):
        with pytest.raises(AssertionError):
            cases.test_copy_sparse_to_sparse_(((4, 5), 2, 3), torch.float32, False)


@pytest.mark.parametrize("layout", ["coo", "csc", "bsc"])
def test_sparse_constructors_reject_small_storage_changes(layout):
    from . import test_sparse_bsc_tensor as bsc
    from . import test_sparse_coo_tensor as coo
    from . import test_sparse_csc_tensor as csc

    operator = f"sparse_{layout}_tensor"

    def corrupted(*args, **kwargs):
        result = getattr(torch.ops.aten, operator)(*args, **kwargs).clone()
        values = result._values() if layout == "coo" else result.values()
        values.add_(1e-6)
        return result

    checks = {
        "coo": lambda: coo.test_sparse_coo_tensor_indices_size(
            coo._COO_2D_CASES[0], torch.float32
        ),
        "csc": lambda: csc.test_sparse_csc_tensor(
            (4, 4), 4, torch.float32, torch.int64, ["0", "1"]
        ),
        "bsc": lambda: bsc.test_sparse_bsc_tensor(
            bsc._BSC_CASES[0], torch.float32, torch.int64
        ),
    }
    with testing.override_gems_op(operator, corrupted):
        with pytest.raises(AssertionError):
            checks[layout]()


@pytest.mark.parametrize("component", ["values", "rows"])
def test_csc_candidate_cannot_change_reference_through_shared_inputs(component):
    from . import test_sparse_csc_tensor as cases

    def corrupted(ccol, row, values, *args, **kwargs):
        if component == "values":
            values.add_(1)
        else:
            row.copy_((row + 1) % 4)
        return torch.ops.aten.sparse_csc_tensor(ccol, row, values, *args, **kwargs)

    with testing.override_gems_op("sparse_csc_tensor", corrupted):
        with pytest.raises(AssertionError):
            cases.test_sparse_csc_tensor(
                (4, 4), 4, torch.float32, torch.int64, ["0", "1"]
            )


@pytest.mark.parametrize("value", [0.5, False])
def test_version_rejects_nonintegral_tensor_results(value):
    from . import test__version as cases

    def invalid(inp):
        return torch.tensor(value, device=inp.device)

    with testing.override_gems_op("_version", invalid):
        with pytest.raises(AssertionError):
            cases.test__version_fresh((4,), torch.float32)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_version_accepts_integral_tensor_results(dtype):
    from . import test__version as cases

    def valid(inp):
        return torch.tensor(
            torch.ops.aten._version(inp), dtype=dtype, device=inp.device
        )

    with testing.override_gems_op("_version", valid):
        cases.test__version_fresh((4,), torch.float32)


@pytest.mark.parametrize("source", ["self", "other"])
def test_new_zeros_rejects_input_aliases(source):
    from . import test__new_zeros_with_same_feature_meta as cases

    def aliased(self_t, other_t, **kwargs):
        return (self_t if source == "self" else other_t).view_as(other_t)

    with testing.override_gems_op("_new_zeros_with_same_feature_meta", aliased):
        with pytest.raises(AssertionError):
            cases.test__new_zeros_with_same_feature_meta_value_ranges(
                (4, 5), (4, 5), 0, ["0", "0"], torch.float32
            )


@pytest.mark.parametrize("source", ["self", "other"])
def test_new_zeros_rejects_input_mutation(source):
    from . import test__new_zeros_with_same_feature_meta as cases

    def mutated(self_t, other_t, **kwargs):
        result = torch.ops.aten._new_zeros_with_same_feature_meta(
            self_t, other_t, **kwargs
        )
        (self_t if source == "self" else other_t).fill_(float("nan"))
        return result

    with testing.override_gems_op("_new_zeros_with_same_feature_meta", mutated):
        with pytest.raises(AssertionError):
            cases.test__new_zeros_with_same_feature_meta(
                (4, 5), (4, 5), 0, torch.float32
            )


def test_new_zeros_checks_output_device_with_cpu_reference(monkeypatch):
    from . import test__new_zeros_with_same_feature_meta as cases

    monkeypatch.setattr(utils, "TO_CPU", True)

    def wrong_device(*args, **kwargs):
        return torch.ops.aten._new_zeros_with_same_feature_meta(*args, **kwargs).cpu()

    with testing.override_gems_op("_new_zeros_with_same_feature_meta", wrong_device):
        with pytest.raises(AssertionError):
            cases.test__new_zeros_with_same_feature_meta(
                (4, 5), (4, 5), 0, torch.float32
            )


def test_new_zeros_rejects_spurious_autograd():
    from . import test__new_zeros_with_same_feature_meta as cases

    def differentiable(*args, **kwargs):
        return torch.ops.aten._new_zeros_with_same_feature_meta(
            *args, **kwargs
        ).requires_grad_()

    with testing.override_gems_op("_new_zeros_with_same_feature_meta", differentiable):
        with pytest.raises(AssertionError):
            cases.test__new_zeros_with_same_feature_meta(
                (4, 5), (4, 5), 0, torch.float32
            )


def test_new_zeros_out_preserves_storage_outside_the_view():
    from . import test__new_zeros_with_same_feature_meta as cases

    def overwritten(self_t, other_t, *, out, **kwargs):
        storage_size = out.untyped_storage().nbytes() // out.element_size()
        out.as_strided((storage_size,), (1,), 0).zero_()
        return out

    with testing.override_gems_op("_new_zeros_with_same_feature_meta", overwritten):
        with pytest.raises(AssertionError):
            cases.test__new_zeros_with_same_feature_meta_out_layouts(
                (4, 10), (10, 2), 0, torch.float32
            )


_BOOL_METADATA_CASES = [
    ("can_cast", "test_can_cast", (torch.float32, torch.float32)),
    ("can_cast", "test_can_cast", (torch.float32, torch.int32)),
    (
        "_has_same_storage_numel",
        "test__has_same_storage_numel_layouts",
        (("plain", (4,)), ("plain", (4,)), torch.float32),
    ),
    (
        "_has_same_storage_numel",
        "test__has_same_storage_numel_layouts",
        (("plain", (4,)), ("plain", (2,)), torch.float32),
    ),
]


@pytest.mark.parametrize("operator,test_name,args", _BOOL_METADATA_CASES)
@pytest.mark.parametrize("result_kind", ["list", "tuple", "vector", "matrix"])
def test_bool_metadata_rejects_container_results(
    operator, test_name, args, result_kind
):
    cases = importlib.import_module(f"tests.test_{operator}")

    def invalid(*inputs):
        value = getattr(torch.ops.aten, operator)(*inputs)
        if result_kind == "list":
            return [value]
        if result_kind == "tuple":
            return (value,)
        shape = (1,) if result_kind == "vector" else (1, 1)
        return torch.full(shape, value, dtype=torch.bool, device=flag_gems.device)

    with testing.override_gems_op(operator, invalid):
        with pytest.raises(AssertionError):
            getattr(cases, test_name)(*args)


@pytest.mark.parametrize("operator,test_name,args", _BOOL_METADATA_CASES)
def test_bool_metadata_accepts_scalar_bool_tensors(operator, test_name, args):
    cases = importlib.import_module(f"tests.test_{operator}")

    def valid(*inputs):
        value = getattr(torch.ops.aten, operator)(*inputs)
        return torch.tensor(value, dtype=torch.bool, device=flag_gems.device)

    with testing.override_gems_op(operator, valid):
        getattr(cases, test_name)(*args)


def test_quantization_params_reject_boolean_zero_point():
    from . import test__choose_qparams_per_tensor as cases

    def invalid(inp, reduce_range):
        scale, zero_point = torch.ops.aten._choose_qparams_per_tensor(inp, reduce_range)
        return scale, bool(zero_point)

    with testing.override_gems_op("_choose_qparams_per_tensor", invalid):
        with pytest.raises(AssertionError):
            cases.test__choose_qparams_per_tensor_constant(0.0, torch.float32, False)


@pytest.mark.parametrize(
    "operator",
    ["_empty_per_channel_affine_quantized", "_make_per_channel_quantized_tensor"],
)
@pytest.mark.parametrize("changed_input", ["scales", "zero_points"])
def test_quantized_factories_preserve_metadata_input_dtypes(operator, changed_input):
    cases = importlib.import_module(f".test_{operator}", package=__package__)

    def rewritten(*args, **kwargs):
        if operator == "_empty_per_channel_affine_quantized":
            scales, zero_points = kwargs["scales"], kwargs["zero_points"]
            result = getattr(torch.ops.aten, operator)(
                *args,
                **dict(kwargs, scales=scales.clone(), zero_points=zero_points.clone()),
            )
        else:
            inp, scales, zero_points, axis = args
            result = getattr(torch.ops.aten, operator)(
                inp, scales.clone(), zero_points.clone(), axis, **kwargs
            )
        value = scales if changed_input == "scales" else zero_points
        dtype = torch.float64 if changed_input == "scales" else torch.int64
        value.data = value.to(dtype)
        return result

    with testing.override_gems_op(operator, rewritten):
        with pytest.raises(AssertionError):
            if operator == "_empty_per_channel_affine_quantized":
                cases.test__empty_per_channel_affine_quantized(
                    (2, 3), 1, torch.qint8, torch.float32, torch.int32
                )
            else:
                cases.test__make_per_channel_quantized_tensor(
                    (2, 3), 1, torch.int8, torch.float32
                )


@pytest.mark.parametrize(
    "test_name,value",
    [
        ("test__empty_affine_quantized_non_finite_scale", float("nan")),
        ("test__empty_affine_quantized_wide_zero_point", 1 << 40),
    ],
)
def test_quantized_factory_special_cases_check_output_device(test_name, value):
    from . import test__empty_affine_quantized as cases

    def wrong_device(*args, **kwargs):
        return torch.ops.aten._empty_affine_quantized(
            *args, **dict(kwargs, device="cpu")
        )

    with testing.override_gems_op("_empty_affine_quantized", wrong_device):
        with pytest.raises(AssertionError):
            getattr(cases, test_name)((2, 3), torch.qint8, value)


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2, torch.bool])
@pytest.mark.parametrize("toggle", [False, True])
def test_negative_view_checks_stored_values_without_materialization(dtype, toggle):
    from . import test__neg_view as cases

    def corrupted(inp):
        stored = torch.ops.aten._neg_view(inp) if inp.is_neg() else inp
        stored.view(torch.uint8).bitwise_xor_(1)
        return torch.ops.aten._neg_view(inp)

    with testing.override_gems_op("_neg_view", corrupted):
        with pytest.raises(AssertionError):
            if toggle:
                cases.test__neg_view_toggle((4, 5), dtype)
            else:
                cases.test__neg_view_unmaterializable_dtypes((4, 5), dtype)


def test_negative_view_requires_exact_sign_flip_gradient():
    from . import test__neg_view as cases

    class BiasedGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp):
            return torch.ops.aten._neg_view(inp)

        @staticmethod
        def backward(ctx, grad):
            return -grad + 1e-6

    with testing.override_gems_op("_neg_view", BiasedGradient.apply):
        with pytest.raises(AssertionError):
            cases.test__neg_view_backward((4, 5), torch.float32)


def test_negative_view_requires_aliasing_for_empty_inputs():
    from . import test__neg_view as cases

    def copied(inp):
        return torch.ops.aten._neg_view(torch.empty_like(inp))

    with testing.override_gems_op("_neg_view", copied):
        with pytest.raises(AssertionError):
            cases.test__neg_view((0,), torch.float32)


def test_negative_view_checks_result_before_mutating_it():
    from . import test__neg_view as cases

    def corrupted(inp):
        inp.zero_()
        return torch.ops.aten._neg_view(inp)

    with testing.override_gems_op("_neg_view", corrupted):
        with pytest.raises(AssertionError):
            cases.test__neg_view_mutation((4, 5), torch.float32)


@pytest.mark.parametrize(
    "operator,args",
    [
        ("combinations", (8, 2, False, torch.float32)),
        ("diagflat", ((3, 4), 1, torch.float32)),
        ("dstack", ([(2, 3), (2, 3)], torch.float32)),
        ("flatten_dense_tensors", ([(2, 3), (2, 3)], torch.float32)),
        ("_fw_primal", ((3, 4), torch.float32)),
        ("_remove_batch_dim", ((1, 3), 0, 2, torch.float32)),
    ],
)
def test_gather_backward_cases_reject_small_forward_errors(operator, args):
    cases = importlib.import_module(f".test_{operator}", package=__package__)

    def corrupted(*args, **kwargs):
        return getattr(torch.ops.aten, operator)(*args, **kwargs) + 1e-6

    with testing.override_gems_op(operator, corrupted):
        with pytest.raises(AssertionError):
            getattr(cases, f"test_{operator}_backward")(*args)


def test_diagflat_rejects_small_gradient_errors():
    from . import test_diagflat as cases

    class CorruptedGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp, offset):
            ctx.shape = inp.shape
            ctx.offset = offset
            return torch.ops.aten.diagflat(inp, offset)

        @staticmethod
        def backward(ctx, grad):
            return torch.ops.aten.diag(grad, ctx.offset).reshape(ctx.shape) + 1e-6, None

    with testing.override_gems_op("diagflat", CorruptedGradient.apply):
        with pytest.raises(AssertionError):
            cases.test_diagflat_backward((3, 4), 1, torch.float32)


@pytest.mark.parametrize(
    "operator,case_name,args",
    [
        ("_fw_primal", "test__fw_primal_rejects_non_tensor", ()),
        (
            "_slow_conv2d_backward",
            "test__slow_conv2d_backward_negative_non_4d_grad_output",
            (),
        ),
        (
            "slow_conv_dilated2d",
            "test_slow_conv_dilated2d_rejects_unsupported_dtype",
            (torch.int32,),
        ),
    ],
)
def test_negative_cases_require_a_candidate(monkeypatch, operator, case_name, args):
    cases = importlib.import_module(f".test_{operator}", package=__package__)
    missing = Mock(side_effect=LookupError("candidate missing"))
    monkeypatch.setattr(testing, "resolve_gems_op", missing)
    with pytest.raises(LookupError, match="candidate missing"):
        getattr(cases, case_name)(*args)
    missing.assert_called_once()


@pytest.mark.parametrize(
    "operator,shape", [("atleast_1d", ()), ("atleast_2d", (3,)), ("atleast_3d", (2, 3))]
)
def test_atleast_backward_rejects_constant_gradient(operator, shape):
    cases = importlib.import_module(f".test_{operator}", package=__package__)

    class ConstantGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp):
            ctx.shape = inp.shape
            return getattr(torch.ops.aten, operator)(inp)

        @staticmethod
        def backward(ctx, grad):
            return torch.ones_like(grad).reshape(ctx.shape)

    with testing.override_gems_op(operator, ConstantGradient.apply):
        with pytest.raises(AssertionError):
            getattr(cases, f"test_{operator}_backward")(shape, torch.float32)


def test_detach_copy_rejects_an_implemented_backward():
    from . import test_detach_copy as cases

    with testing.override_gems_op("detach_copy", lambda inp: inp.clone()):
        with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE"):
            cases.test_detach_copy_no_backward((3, 4), torch.float32)


@pytest.mark.parametrize("operator", ["_nested_tensor_size", "_nested_tensor_strides"])
def test_nested_metadata_stays_on_cpu_with_cpu_reference(operator, monkeypatch):
    cases = importlib.import_module(f".test_{operator}", package=__package__)
    monkeypatch.setattr(utils, "TO_CPU", True)

    def misplaced(inp):
        return getattr(torch.ops.aten, operator)(inp).to(inp.device)

    with testing.override_gems_op(operator, misplaced):
        with pytest.raises(AssertionError):
            getattr(cases, f"test_{operator}_nan_inf_values")(torch.float32, "nan")


@pytest.mark.parametrize(
    "operator",
    [
        "ccol_indices",
        "crow_indices",
        "col_indices",
        "ccol_indices_copy",
        "crow_indices_copy",
        "col_indices_copy",
    ],
)
def test_compressed_indices_reject_widened_index_dtype(operator):
    cases = importlib.import_module(f".test_{operator}", package=__package__)

    def widened(inp):
        return getattr(torch.ops.aten, operator)(inp).to(torch.int64)

    with testing.override_gems_op(operator, widened):
        with pytest.raises(AssertionError):
            getattr(cases, f"test_{operator}_index_layouts")(
                cases._INDEX_LAYOUT_CASES[0], (), (), torch.float32, torch.int32
            )


@pytest.mark.parametrize(
    "n,r,with_replacement", [(8, 1, False), (8, 1, True), (1, 2, False)]
)
def test_combinations_nonreducing_backward_rejects_small_errors(n, r, with_replacement):
    from . import test_combinations as cases

    class CorruptedGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp, r, replacement):
            ctx.shape = inp.shape
            return torch.ops.aten.combinations(inp, r, replacement)

        @staticmethod
        def backward(ctx, grad):
            if grad.numel() == 0:
                return (
                    torch.full(ctx.shape, 1e-6, dtype=grad.dtype, device=grad.device),
                    None,
                    None,
                )
            return grad.reshape(ctx.shape) + 1e-6, None, None

    with testing.override_gems_op("combinations", CorruptedGradient.apply):
        with pytest.raises(AssertionError):
            cases.test_combinations_backward(n, r, with_replacement, torch.float32)


def test_combinations_zero_r_rejects_a_gradient_connection():
    from . import test_combinations as cases

    with testing.override_gems_op("combinations", lambda inp, r, replacement: inp[:0]):
        with pytest.raises(AssertionError):
            cases.test_combinations_zero_r_no_autograd(8, False, torch.float32)


def test_qparams_rejects_zero_tiny_scale():
    from tests import test__choose_qparams_per_tensor as cases

    with pytest.raises(AssertionError):
        cases._assert_pair((0.0, 0), (6e-5, 0))
    cases._assert_pair((6e-5, 0), (6e-5, 0))


def test_direct_candidate_override_restores_after_failure(monkeypatch):
    default = lambda value: value
    first = lambda value: value + 1
    second = lambda value: value + 2
    monkeypatch.setattr(flag_gems, "validation_candidate", default, raising=False)
    with testing.override_gems_op("validation_candidate", first):
        with pytest.raises(RuntimeError, match="candidate failed"):
            with testing.override_gems_op("validation_candidate", second):
                assert testing.resolve_gems_op("validation_candidate") is second
                raise RuntimeError("candidate failed")
        assert testing.resolve_gems_op("validation_candidate") is first
    assert testing.resolve_gems_op("validation_candidate") is default


@pytest.mark.parametrize("profile", [False, True])
def test_candidate_replay_preserves_master_case_contract(
    bench_config, monkeypatch, profile
):
    from benchmark.conftest import BenchConfig
    from benchmark.consts import BenchLevel

    config = BenchConfig()
    config.current_nodeid = "benchmark/test_validation.py::test_replay"
    config.bench_level = BenchLevel.CORE
    monkeypatch.setattr(bench_config, "Config", config)
    materialized = []
    calls = []

    def build(plan, dtype, device):
        materialized.append(plan.shape["input"])
        return (torch.ones(plan.shape["input"], dtype=dtype),)

    bench = bench_config.GenericBenchmark(
        op_name="validation_replay",
        torch_op=lambda *args: pytest.fail("native reference invoked"),
        gems_op=lambda value: calls.append(
            testing.current_gems_op_case("validation_replay")
        ),
        case_fn=lambda shape, dtype: iter(
            [bench_config.BenchmarkCasePlan(shape={"input": shape})]
        ),
        build_inputs_fn=build,
        dtypes=[torch.float32],
    )
    bench.shapes = [(2,), (3,)]
    monkeypatch.setattr(bench, "init_user_config", lambda: None)
    monkeypatch.setattr(
        bench, "_measure_input", lambda *args, **kwargs: pytest.fail("timing invoked")
    )
    events = []
    monkeypatch.setattr(
        bench, "_external_profiler_start", lambda: events.append("start")
    )
    monkeypatch.setattr(bench, "_external_profiler_stop", lambda: events.append("stop"))
    ids = [case.case_id for case in bench.list_cases().cases]
    assert materialized == []

    if profile:
        config.profile_only = True
        config.profile_warmup = 2
        config.profile_iterations = 3
        config.case_ids = [ids[1]]
        assert bench.run() == [ids[1]]
        assert calls == [ids[1]] * 5
        assert materialized == [(3,), (3,)]
        assert events == ["start", "stop"]
    else:
        config.preflight_only = True
        assert bench.run() == ids
        assert calls == ids
        assert materialized == [(2,), (3,)]
        assert events == []
    assert testing.current_gems_op_case() is None


def test_add_batch_dim_rejects_copied_storage():
    from tests import test__add_batch_dim as cases

    def copied(inp, dim, level):
        return torch.ops.aten._add_batch_dim(tu.to_reference(inp), dim, level)

    with flag_gems.testing.override_gems_op("_add_batch_dim", copied):
        with pytest.raises(AssertionError):
            cases.test__add_batch_dim((3, 5), 0, 0, torch.float32)


def test_remove_batch_dim_exercises_batched_input():
    from tests import test__remove_batch_dim as cases

    def plain_only(inp, *args):
        assert not torch._C._functorch.is_legacy_batchedtensor(inp), "batched input"
        return torch.ops.aten._remove_batch_dim(inp, *args)

    with flag_gems.testing.override_gems_op("_remove_batch_dim", plain_only):
        with pytest.raises(AssertionError, match="batched input"):
            cases.test__remove_batch_dim_batched(1, 2, 0, torch.float32)


def test_fw_primal_rejects_retained_tangent():
    from tests import test__fw_primal as cases

    with flag_gems.testing.override_gems_op(
        "_fw_primal", lambda inp, level: torch.ops.aten.alias(inp)
    ):
        with pytest.raises(AssertionError):
            cases.test__fw_primal_dual((3, 5), torch.float32)


def test_data_rejects_shared_version_counter():
    from tests import test_data as cases

    with flag_gems.testing.override_gems_op("data", lambda inp: inp.detach()):
        with pytest.raises(AssertionError):
            cases.test_data_independent_version_counter(0)


@pytest.mark.parametrize("level", ["CORE", "COMPREHENSIVE"])
def test_operator_shapes_override_defaults(bench_config, tmp_path, level):
    from benchmark import consts
    from benchmark.test_adjoint import AdjointBenchmark, _build_inputs_fn, _case_fn

    bench_config.Config.bench_level = getattr(consts.BenchLevel, level)
    bench_config.Config.query = False
    path = tmp_path / "shapes.yaml"
    path.write_text("adjoint:\n  shapes: [[7, 11]]\n  shape_desc: custom matrix\n")
    bench = AdjointBenchmark(
        "adjoint",
        torch.ops.aten.adjoint,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
    )
    bench.set_shapes(str(path))
    assert bench.shapes == [(7, 11)]
    assert bench.shape_desc == "custom matrix"


def test_composite_shapes_override_defaults(bench_config, tmp_path):
    from benchmark import consts
    from benchmark.test_sparse_bsr_tensor import (
        SparseBsrTensorBenchmark,
        _build_inputs_fn,
        _case_fn,
    )

    bench_config.Config.bench_level = consts.BenchLevel.COMPREHENSIVE
    bench_config.Config.query = False
    path = tmp_path / "shapes.yaml"
    path.write_text("sparse_bsr_tensor:\n  shapes: [[[2, 6, 12], [2, 3]]]\n")
    bench = SparseBsrTensorBenchmark(
        "sparse_bsr_tensor",
        torch.ops.aten.sparse_bsr_tensor,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
    )
    bench.set_shapes(str(path))
    assert bench.shapes == [((2, 6, 12), (2, 3))]
    plan = next(_case_fn(bench.shapes[0], torch.float32))
    crow, col, values, size, kwargs = _build_inputs_fn(plan, torch.float32, "cpu")
    with torch.sparse.check_sparse_tensor_invariants():
        result = torch.ops.aten.sparse_bsr_tensor(crow, col, values, size, **kwargs)
    assert result.dense_dim() == 0
    assert result.crow_indices().shape == (2, 4)
    assert result.values().shape == (2, 12, 2, 3)


@pytest.mark.parametrize(
    "module_name",
    [
        "tests.test__slow_conv2d_backward",
        "tests.test__slow_conv2d_forward",
        "tests.test_slow_conv_dilated2d",
        "tests.test_slow_conv_dilated3d",
        "tests.test_slow_conv_transpose2d",
        "tests.test_slow_conv_transpose3d",
        "tests.test_thnn_conv2d",
        "benchmark.test_thnn_conv2d",
    ],
)
def test_convolution_precision_restored_on_failure(module_name):
    module = importlib.import_module(module_name)
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        fixture = module.full_precision.__wrapped__()
        next(fixture)
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
        with pytest.raises(RuntimeError, match="test failure"):
            fixture.throw(RuntimeError("test failure"))
        assert torch.backends.cuda.matmul.allow_tf32
        assert torch.backends.cudnn.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


@pytest.mark.parametrize("operator", ["sparse_csc_tensor", "sparse_compressed_tensor"])
def test_special_value_sparse_fixtures_have_valid_indices(operator):
    cases = importlib.import_module(f"tests.test_{operator}")
    with torch.sparse.check_sparse_tensor_invariants():
        with flag_gems.testing.override_gems_op(
            operator, getattr(torch.ops.aten, operator)
        ):
            getattr(cases, f"test_{operator}_nan_inf_values")(torch.float32, "nan")


def test_csr_fixture_tables_have_valid_indices():
    from tests import test_sparse_csr_tensor as cases

    with torch.sparse.check_sparse_tensor_invariants():
        with flag_gems.testing.override_gems_op(
            "sparse_csr_tensor", torch.ops.aten.sparse_csr_tensor
        ):
            for case in cases._CSR_2D_CASES:
                cases.test_sparse_csr_tensor_crow_col_value_size(
                    case, torch.float32, ["-1", "1"]
                )
            for case in cases._CSR_3D_CASES:
                cases.test_sparse_csr_tensor_crow_col_value_size_batched(
                    case, torch.float32, ["-1", "1"]
                )

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Public ATen Linear correctness and bounded MetaX tuning selection."""

import importlib

import pytest
import torch

import flag_gems

pytestmark = [
    pytest.mark.linear,
    pytest.mark.skipif(
        flag_gems.vendor_name != "metax" or not torch.cuda.is_available(),
        reason="requires the MetaX Linear backend",
    ),
]


@pytest.mark.parametrize("shape", [(7,), (2, 7), (2, 3, 7)])
@pytest.mark.parametrize("weight_k", [6, 8])
def test_public_linear_rejects_mismatched_k_before_launch(monkeypatch, shape, weight_k):
    module = importlib.import_module(flag_gems.linear.__module__)
    x = torch.empty(shape, device="cuda")
    weight = torch.empty((5, weight_k), device="cuda")
    calls = []

    def reject(*args, **kwargs):
        calls.append(True)
        raise AssertionError("Invalid dimensions reached tuning or kernel launch")

    class RejectKernel:
        __getitem__ = reject

    monkeypatch.setattr(module, "_use_small_fp32_entry", reject)
    monkeypatch.setattr(module, "linear_kernel", RejectKernel())
    with pytest.raises(RuntimeError) as error:
        flag_gems.linear(x, weight)
    assert str(error.value) == f"weight (5, {weight_k}) is not an (N, 7) matrix"
    assert calls == []


@pytest.mark.parametrize("shape", [(32,), (2, 32), (2, 3, 32)])
@pytest.mark.parametrize("has_bias", [False, True])
def test_public_linear_valid_dimensions_match_reference(shape, has_bias):
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(732)
        x = torch.randn(shape, device="cuda")
        weight = torch.randn((17, 32), device="cuda")
        bias = torch.randn(17, device="cuda") if has_bias else None
        actual = flag_gems.linear(x, weight, bias)
        expected = torch.nn.functional.linear(
            x.cpu().double(),
            weight.cpu().double(),
            None if bias is None else bias.cpu().double(),
        )
        assert actual.shape == (*shape[:-1], 17)
        flag_gems.testing.assert_close(
            actual.cpu(), expected, dtype=torch.float32, reduce_dim=32
        )


@pytest.mark.parametrize(
    "rows,n,k,dtype,layout,parameter,tuned",
    [
        (1, 1024, 1024, torch.float32, "plain", False, True),
        (32, 1024, 1024, torch.float32, "plain", True, True),
        (32, 3072, 1024, torch.float32, "plain", False, True),
        (32, 4096, 1024, torch.float32, "plain", False, True),
        (32, 1024, 4096, torch.float32, "offset", False, True),
        (1024, 1024, 1024, torch.float32, "plain", False, True),
        (1025, 1024, 1024, torch.float32, "plain", False, False),
        (32, 17, 1024, torch.float32, "plain", False, False),
        (32, 1024, 1024, torch.float16, "plain", False, False),
        (32, 1024, 1024, torch.bfloat16, "plain", False, False),
        (32, 1024, 1024, torch.float32, "weight_stride", False, False),
    ],
)
@pytest.mark.parametrize("has_bias", [False, True])
def test_public_linear_matches_fp64_and_keeps_default_configs(
    monkeypatch, rows, n, k, dtype, layout, parameter, tuned, has_bias
):
    module = importlib.import_module(flag_gems.linear.__module__)
    configs_before = [
        dict(c.all_kwargs()) for c in module.runtime.get_tuned_config("linear")
    ]
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(731)
        x = torch.randn(rows + 1, k, device="cuda", dtype=dtype)
        x = x[1:] if layout == "offset" else x[:rows]
        weight = torch.randn(n, k, device="cuda", dtype=dtype)
        if layout == "weight_stride":
            weight = weight.t().contiguous().t()
        if parameter:
            weight = torch.nn.Parameter(weight, requires_grad=False)
            weight.loader_metadata = object()
        original_weight = weight
        original_storage = weight.data_ptr()
        bias = torch.randn(n, device="cuda", dtype=dtype) if has_bias else None
        selected = []
        real_select = module._small_fp32_entry

        def select():
            selected.append(True)
            return real_select()

        monkeypatch.setattr(module, "_small_fp32_entry", select)
        with torch.inference_mode():
            actual = flag_gems.linear(x, weight, bias)
        expected = torch.nn.functional.linear(
            x.cpu().double(),
            weight.cpu().double(),
            None if bias is None else bias.cpu().double(),
        )
        # Use FG's existing reduction-length-aware comparison, not a fixed
        # elementwise absolute threshold for near-cancelling long dot products.
        # Also bound the total relative error so the scaled absolute allowance
        # cannot conceal a loss of accumulator precision.
        host = actual.cpu()
        flag_gems.testing.assert_close(host, expected, dtype=dtype, reduce_dim=k)
        relative_error = torch.linalg.vector_norm(
            host.double() - expected
        ) / torch.linalg.vector_norm(expected)
        assert relative_error.item() <= max(1e-5, 2 * torch.finfo(dtype).eps)
        assert selected == ([True] if tuned else [])
        assert weight is original_weight and weight.data_ptr() == original_storage
        assert [
            dict(c.all_kwargs()) for c in module.runtime.get_tuned_config("linear")
        ] == configs_before


def test_public_linear_registered_aten_path(monkeypatch):
    x = torch.randn(16, 1024, device="cuda")
    w = torch.randn(1024, 1024, device="cuda")
    module = importlib.import_module(flag_gems.linear.__module__)
    selected = []
    real_select = module._small_fp32_entry

    def select():
        selected.append(True)
        return real_select()

    monkeypatch.setattr(module, "_small_fp32_entry", select)
    # Exercise the public registrar without enabling unrelated ATen overrides.
    previous = getattr(flag_gems, "current_work_registrar", None)
    with monkeypatch.context() as registration:
        registration.setattr(flag_gems, "current_work_registrar", None, raising=False)
        library = torch.library.Library("aten", "IMPL")
        try:
            flag_gems.only_enable(lib=library, include=["linear"])
            actual = torch.nn.functional.linear(x, w)
        finally:
            library._destroy()
    _assert_fp64_linear(actual, x, w, None)
    assert getattr(flag_gems, "current_work_registrar", None) is previous
    # A later native call must no longer enter the temporary registration.
    torch.nn.functional.linear(x, w)
    assert selected == [True]


def test_tuned_linear_capture_and_two_replays():
    with torch.inference_mode():
        x = torch.randn(32, 1024, device="cuda")
        w = torch.randn(1024, 1024, device="cuda")
        flag_gems.linear(x, w)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = flag_gems.linear(x, w)
        for shift in (0.5, -0.75):
            x.add_(shift)
            output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            expected = torch.nn.functional.linear(x.cpu().double(), w.cpu().double())
            torch.testing.assert_close(
                output.cpu().double(), expected, atol=2e-4, rtol=2e-4
            )


class _LoaderParameter(torch.nn.Parameter):
    """Exercise ordinary Parameter subclass metadata without a vLLM dependency."""


def _assert_fp64_linear(actual, x, weight, bias):
    expected = torch.nn.functional.linear(
        x.detach().cpu().double(),
        weight.detach().cpu().double(),
        None if bias is None else bias.detach().cpu().double(),
    )
    host = actual.detach().cpu()
    flag_gems.testing.assert_close(
        host, expected, dtype=x.dtype, reduce_dim=x.shape[-1]
    )
    norm = torch.linalg.vector_norm(expected)
    if norm == 0:
        assert torch.equal(host.double(), expected)
    else:
        relative = torch.linalg.vector_norm(host.double() - expected) / norm
        assert relative <= max(1e-5, 2 * torch.finfo(x.dtype).eps)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("features", [(1024, 1024), (17, 32)])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("subclass", [False, True])
def test_parameter_metadata_and_values_preserved(dtype, features, strided, subclass):
    n, k = features
    generator = torch.Generator(device="cuda").manual_seed(734)
    x = torch.randn(3, k, device="cuda", dtype=dtype, generator=generator)
    data = torch.randn(n, k, device="cuda", dtype=dtype, generator=generator)
    if strided:
        data = data.t().contiguous().t()
    cls = _LoaderParameter if subclass else torch.nn.Parameter
    weight = cls(data, requires_grad=False)
    metadata = {"shard": object()}
    weight.weight_loader = object()
    weight.loader_metadata = metadata
    attrs = dict(weight.__dict__)
    before = weight.detach().clone()
    pointer, stride, offset = (
        weight.data_ptr(),
        weight.stride(),
        weight.storage_offset(),
    )
    bias = torch.randn(n * 2, device="cuda", dtype=dtype, generator=generator)[::2]
    with torch.inference_mode():
        actual = flag_gems.linear(x, weight, bias)
    _assert_fp64_linear(actual, x, weight, bias)
    assert weight.data_ptr() == pointer
    assert weight.stride() == stride and weight.storage_offset() == offset
    assert torch.equal(weight.detach(), before)
    assert weight.__dict__.keys() == attrs.keys()
    assert all(weight.__dict__[key] is value for key, value in attrs.items())
    assert weight.loader_metadata is metadata


@pytest.mark.parametrize(
    "case,tuned",
    [
        ("one_dimensional", False),
        ("input_stride", False),
        ("grad_input", False),
        ("grad_weight", False),
        ("grad_bias", False),
        ("no_grad", True),
        ("inference", True),
    ],
)
def test_public_selection_exclusion_branches(monkeypatch, case, tuned):
    from contextlib import nullcontext

    module = importlib.import_module(flag_gems.linear.__module__)
    generator = torch.Generator(device="cuda").manual_seed(735)
    x = torch.randn(3, 1024, device="cuda", generator=generator)
    weight = torch.randn(1024, 1024, device="cuda", generator=generator)
    bias = torch.randn(1024, device="cuda", generator=generator)
    if case == "one_dimensional":
        x = x[0]
    elif case == "input_stride":
        x = torch.randn(3, 2048, device="cuda", generator=generator)[:, ::2]
    if case in ("grad_input", "no_grad", "inference"):
        x.requires_grad_(True)
    elif case == "grad_weight":
        weight.requires_grad_(True)
    elif case == "grad_bias":
        bias.requires_grad_(True)
    selected = []
    real = module._small_fp32_entry

    def select():
        selected.append(True)
        return real()

    monkeypatch.setattr(module, "_small_fp32_entry", select)
    context = (
        torch.no_grad()
        if case == "no_grad"
        else torch.inference_mode() if case == "inference" else nullcontext()
    )
    with context:
        actual = flag_gems.linear(x, weight, bias)
    _assert_fp64_linear(actual, x, weight, bias)
    assert selected == ([True] if tuned else [])
    # Forward routing only: this does not add or validate backward support.


@pytest.mark.parametrize("invalid", ["bias_dtype", "bias_cpu", "weight_cpu", "empty"])
def test_selection_metadata_rejections(invalid):
    module = importlib.import_module(flag_gems.linear.__module__)
    x = torch.empty(3, 1024, device="cuda")
    weight = torch.empty(1024, 1024, device="cuda")
    bias = torch.empty(1024, device="cuda")
    if invalid == "bias_dtype":
        bias = bias.half()
    elif invalid == "bias_cpu":
        bias = bias.cpu()
    elif invalid == "weight_cpu":
        weight = weight.cpu()
    else:
        x = x[:0]
    # Only the pre-execution capability predicate is tested for invalid metadata.
    # Do not launch a baseline kernel on mismatched device/dtype inputs.
    assert not module._use_small_fp32_entry(x, weight, bias, x.shape[0])


@pytest.mark.parametrize(
    "case", ["zero_input", "zero_weight", "cancellation", "bias_stride"]
)
def test_tuned_numerical_boundaries(case):
    x = torch.ones(3, 1024, device="cuda")
    weight = torch.ones(1024, 1024, device="cuda")
    bias = torch.zeros(2048, device="cuda")[::2]
    if case == "zero_input":
        x.zero_()
    elif case == "zero_weight":
        weight.zero_()
    elif case == "cancellation":
        weight[:, 1::2] = -1
    else:
        bias.copy_(torch.arange(1024, device="cuda"))
    actual = flag_gems.linear(x, weight, bias)
    _assert_fp64_linear(actual, x, weight, bias)


@pytest.mark.parametrize("tensor", ["input", "weight", "bias"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_tuned_nonfinite_inputs(tensor, value):
    x = torch.ones(3, 1024, device="cuda")
    weight = torch.ones(1024, 1024, device="cuda")
    bias = torch.zeros(1024, device="cuda")
    target = {"input": x, "weight": weight, "bias": bias}[tensor]
    target.view(-1)[0] = value
    actual = flag_gems.linear(x, weight, bias)
    expected = torch.nn.functional.linear(
        x.cpu().double(), weight.cpu().double(), bias.cpu().double()
    )
    torch.testing.assert_close(
        actual.cpu().double(), expected, atol=0, rtol=0, equal_nan=True
    )

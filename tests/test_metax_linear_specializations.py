# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import flag_gems
import pytest
import torch
from flag_gems import runtime

pytestmark = pytest.mark.skipif(
    runtime.device.vendor_name != "metax", reason="MetaX specialization"
)


def assert_reference(actual, expected):
    x, r = actual.detach().cpu().float(), expected.detach().cpu().float()
    delta = x - r
    assert torch.isfinite(x).all()
    assert (delta.norm() / r.norm().clamp_min(1e-20)).item() <= 0.004
    assert (delta.abs().max() / r.abs().max().clamp_min(1e-20)).item() <= 0.01


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 384, 6144),
        (1, 768, 6144),
        (8192, 1024, 6144),
        (2, 384, 6144),
        (1, 383, 6144),
        (1, 385, 6144),
        (1, 767, 6144),
        (1, 769, 6144),
        (1, 384, 6143),
        (1, 384, 6145),
        (8191, 1024, 6144),
    ],
)
def test_linear_specialization_and_neighbors(m, n, k):
    torch.manual_seed(560)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(n, k, dtype=x.dtype, device=x.device)
    result = flag_gems.linear(x, w)
    expected = torch.nn.functional.linear(x.cpu().float(), w.cpu().float())
    assert_reference(result, expected)
    assert result.shape == (m, n) and result.dtype == x.dtype


@pytest.mark.parametrize(
    "layout", ["input_stride", "weight_stride", "bias", "rank1", "rank3"]
)
def test_linear_interfaces(layout):
    torch.manual_seed(560)
    x = torch.randn(
        1,
        12288 if layout == "input_stride" else 6144,
        dtype=torch.bfloat16,
        device="cuda",
    )
    if layout == "input_stride":
        x = x[:, ::2]
    w = torch.randn(
        384,
        12288 if layout == "weight_stride" else 6144,
        dtype=x.dtype,
        device=x.device,
    )
    if layout == "weight_stride":
        w = w[:, ::2]
    bias = (
        torch.randn(384, dtype=x.dtype, device=x.device) if layout == "bias" else None
    )
    if layout == "rank1":
        x = x[0]
    if layout == "rank3":
        x = x.unsqueeze(0)
    result = flag_gems.linear(x, w, bias)
    expected = torch.nn.functional.linear(
        x.cpu().float(), w.cpu().float(), None if bias is None else bias.cpu().float()
    )
    assert result.shape == expected.shape
    assert_reference(result, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_other_dtypes_keep_native(dtype):
    x = torch.randn(1, 6144, dtype=dtype, device="cuda")
    w = torch.randn(384, 6144, dtype=dtype, device="cuda")
    actual = flag_gems.linear(x, w)
    assert_reference(
        actual, torch.nn.functional.linear(x.cpu().float(), w.cpu().float())
    )


def test_changed_input_graph_and_input_preservation():
    x = torch.randn(1, 6144, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(768, 6144, dtype=x.dtype, device=x.device)
    original = x.clone()
    for _ in range(3):
        flag_gems.linear(x, w)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = flag_gems.linear(x, w)
    graph.replay()
    assert torch.equal(x, original)
    x.mul_(0.25)
    graph.replay()
    torch.cuda.synchronize()
    assert_reference(
        output, torch.nn.functional.linear(x.cpu().float(), w.cpu().float())
    )


def test_empty_batch():
    x = torch.empty(0, 6144, dtype=torch.bfloat16, device="cuda")
    w = torch.empty(384, 6144, dtype=x.dtype, device=x.device)
    result = flag_gems.linear(x, w)
    assert result.shape == (0, 384) and result.numel() == 0


@pytest.mark.parametrize("special", [float("nan"), float("inf")])
def test_nonfinite_input(special):
    x = torch.zeros(1, 6144, dtype=torch.bfloat16, device="cuda")
    w = torch.ones(384, 6144, dtype=x.dtype, device=x.device)
    x[0, 0] = special
    result = flag_gems.linear(x, w)
    expected = torch.nn.functional.linear(x.cpu().float(), w.cpu().float()).bfloat16()
    torch.testing.assert_close(result.cpu(), expected, equal_nan=True)


def test_training_input_does_not_enter_inference_kernel(monkeypatch):
    import importlib

    impl = importlib.import_module(flag_gems.linear.__module__)

    class UnexpectedLaunch:
        def __getitem__(self, grid):
            raise AssertionError("inference specialization used for training input")

    monkeypatch.setattr(impl, "linear_gemv_untransposed_kernel", UnexpectedLaunch())
    x = torch.randn(1, 6144, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    w = torch.randn(384, 6144, dtype=x.dtype, device=x.device)
    result = flag_gems.linear(x, w)
    assert_reference(
        result, torch.nn.functional.linear(x.detach().cpu().float(), w.cpu().float())
    )

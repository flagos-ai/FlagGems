# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import importlib

import pytest
import torch

import flag_gems

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _reference_scaled_mm(a, b, sa, sb, bias, dtype):
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        out = (a.float() @ b.float()) * sa * sb
        if bias is not None:
            out += bias
        return out.to(dtype)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


@pytest.mark.scaled_mm
@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 1536, 6144),
        (64, 1536, 6144),
        (4096, 1536, 6144),
        (5089, 3072, 6144),
        (8192, 6144, 1024),
        (4096, 6144, 1536),
        (5089, 768, 6144),
    ],
)
def test_hopper_opt_in_tiles(m, n, k, monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Hopper opt-in tiles")
    torch.manual_seed(53)
    a = torch.randint(-32, 32, (m, k), device="cuda", dtype=torch.int8)
    # Row-major B selects the public Triton fallback, rather than the
    # existing specialized scaled-MM entrypoint.
    b = torch.randint(-32, 32, (k, n), device="cuda", dtype=torch.int8)
    sa = torch.rand((m, 1), device="cuda") * 0.01
    sb = torch.rand((1, n), device="cuda") * 0.01
    bias = torch.randn((n,), device="cuda", dtype=torch.bfloat16)
    monkeypatch.setenv("FLAGGEMS_I8_SCALED_MM_SHAPE_TILES", "1")
    actual = flag_gems.scaled_mm_int8(a, b, sa, sb, bias=bias, out_dtype=torch.bfloat16)
    expected = _reference_scaled_mm(a, b, sa, sb, bias, torch.bfloat16)
    torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0.016)


@pytest.mark.scaled_mm
@pytest.mark.parametrize("m,k,n", [(0, 32, 16), (3, 32, 0), (3, 0, 17), (3, 35, 17)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_int8_empty_tail_stride_graph(m, k, n, dtype):
    a = torch.randint(-32, 32, (m, 2 * k), device="cuda", dtype=torch.int8)[:, ::2]
    b = torch.randint(-32, 32, (k, 2 * n), device="cuda", dtype=torch.int8)[:, ::2]
    sa = torch.ones((m, 1), device="cuda") * 0.001
    sb = torch.ones((1, n), device="cuda") * 0.01
    bias = torch.randn(n, device="cuda", dtype=dtype)

    def candidate():
        return flag_gems.scaled_mm_int8(a, b, sa, sb, bias, out_dtype=dtype)

    actual = candidate()
    expected = _reference_scaled_mm(a, b, sa, sb, bias, dtype)
    tolerance = 1e-6 if dtype == torch.float32 else 0.016
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    if m and k and n:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            actual = candidate()
        a.fill_(2)
        g.replay()
        expected = _reference_scaled_mm(a, b, sa, sb, bias, dtype)
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.scaled_mm
def test_int8_specialized_route_and_fp8_schema(monkeypatch):
    module = importlib.import_module("flag_gems.fused.scaled_mm_int8")
    original = module.cutlass_scaled_mm
    calls = []

    def tracked(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(module, "cutlass_scaled_mm", tracked)
    a = torch.randint(-32, 32, (7, 128), device="cuda", dtype=torch.int8)
    b = torch.randint(-32, 32, (32, 128), device="cuda", dtype=torch.int8).t()
    sa, sb = (
        torch.ones((7, 1), device="cuda") * 0.001,
        torch.ones((1, 32), device="cuda") * 0.01,
    )
    actual = flag_gems.scaled_mm_int8(a, b, sa, sb)
    # Only Hopper uses the specialized column-major entrypoint. Other CUDA
    # architectures must keep the generic fallback and its numeric checks.
    assert bool(calls) == (torch.cuda.get_device_capability(a.device)[0] == 9)
    expected = _reference_scaled_mm(a, b, sa, sb, None, torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0.016, atol=0.03125)
    with pytest.raises(RuntimeError, match="Float8"):
        flag_gems.scaled_mm(a, b, sa, sb, out_dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="FP32"):
        flag_gems.scaled_mm_int8(a, b, sa.half(), sb)
    with pytest.raises(NotImplementedError):
        flag_gems.scaled_mm_int8(a.half(), b.half(), sa, sb)

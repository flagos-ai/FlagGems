# _ADAPTIVE_AVG_POOL2D operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops._adaptive_avg_pool2d import (
    _adaptive_avg_pool2d as gems__adaptive_avg_pool2d,
)
from flag_gems.experimental_ops._adaptive_avg_pool2d import (
    _adaptive_avg_pool2d_out as gems__adaptive_avg_pool2d_out,
)

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
try:
    from tests.accuracy_utils import gems_assert_close
except ImportError:
    # Fallback values when running outside pytest

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


@pytest.mark.adaptive_avg_pool2d
@pytest.mark.parametrize(
    "case",
    [
        ((2, 3, 8, 7), (1, 1)),
        ((2, 3, 8, 7), (8, 7)),
        ((2, 3, 8, 7), (4, 3)),
        ((2, 3, 8, 7), (9, 5)),
        ((4, 16, 64, 48), (1, 1)),
        ((4, 16, 64, 48), (32, 24)),
        ((4, 16, 64, 48), (80, 60)),
        ((2, 32, 128, 96), (64, 48)),
        ((2, 32, 128, 96), (128, 96)),
        ((2, 32, 128, 96), (160, 120)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test__adaptive_avg_pool2d_tensor(case, dtype):
    shape, output_size = case
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()

    ref_out = torch.ops.aten._adaptive_avg_pool2d(ref_x, output_size)

    with flag_gems.use_gems():
        act_out = gems__adaptive_avg_pool2d(x, output_size)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.adaptive_avg_pool2d
@pytest.mark.parametrize(
    "case",
    [
        ((2, 3, 8, 7), (1, 1)),
        ((2, 3, 8, 7), (8, 7)),
        ((2, 3, 8, 7), (4, 3)),
        ((2, 3, 8, 7), (9, 5)),
        ((4, 16, 64, 48), (1, 1)),
        ((4, 16, 64, 48), (32, 24)),
        ((4, 16, 64, 48), (80, 60)),
        ((2, 32, 128, 96), (64, 48)),
        ((2, 32, 128, 96), (128, 96)),
        ((2, 32, 128, 96), (160, 120)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test__adaptive_avg_pool2d_out(case, dtype):
    shape, output_size = case
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()

    out_shape = (shape[0], shape[1], output_size[0], output_size[1])
    ref_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    act_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)

    ref_out = torch.ops.aten._adaptive_avg_pool2d.out(
        ref_x, output_size, out=ref_out_buf
    )

    with flag_gems.use_gems():
        act_out = gems__adaptive_avg_pool2d_out(x, output_size, act_out_buf)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.adaptive_avg_pool2d
@pytest.mark.parametrize(
    "case",
    [
        ((2, 3, 8, 7), (1, 1)),
        ((2, 3, 8, 7), (8, 7)),
        ((2, 3, 8, 7), (4, 3)),
        ((2, 3, 8, 7), (9, 5)),
        ((4, 16, 64, 48), (1, 1)),
        ((4, 16, 64, 48), (32, 24)),
        ((4, 16, 64, 48), (80, 60)),
        ((2, 32, 128, 96), (64, 48)),
        ((2, 32, 128, 96), (128, 96)),
        ((2, 32, 128, 96), (160, 120)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test__adaptive_avg_pool2d_benchmark_tensor(case, dtype):
    quantiles = [0.5, 0.2, 0.8]

    shape, output_size = case
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten._adaptive_avg_pool2d(ref_x, output_size),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems__adaptive_avg_pool2d(x, output_size),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"_adaptive_avg_pool2d {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.adaptive_avg_pool2d
@pytest.mark.parametrize(
    "case",
    [
        ((2, 3, 8, 7), (1, 1)),
        ((2, 3, 8, 7), (8, 7)),
        ((2, 3, 8, 7), (4, 3)),
        ((2, 3, 8, 7), (9, 5)),
        ((4, 16, 64, 48), (1, 1)),
        ((4, 16, 64, 48), (32, 24)),
        ((4, 16, 64, 48), (80, 60)),
        ((2, 32, 128, 96), (64, 48)),
        ((2, 32, 128, 96), (128, 96)),
        ((2, 32, 128, 96), (160, 120)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test__adaptive_avg_pool2d_benchmark_out(case, dtype):
    quantiles = [0.5, 0.2, 0.8]

    shape, output_size = case
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()

    out_shape = (shape[0], shape[1], output_size[0], output_size[1])
    ref_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    act_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten._adaptive_avg_pool2d.out(
            ref_x, output_size, out=ref_out_buf
        ),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems__adaptive_avg_pool2d_out(x, output_size, act_out_buf),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"_adaptive_avg_pool2d {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")

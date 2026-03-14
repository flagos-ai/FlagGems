# REFLECTION_PAD2D operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.reflection_pad2d import (
    reflection_pad2d as gems_reflection_pad2d,
)
from flag_gems.experimental_ops.reflection_pad2d import (
    reflection_pad2d_out as gems_reflection_pad2d_out,
)

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


@pytest.mark.reflection_pad2d
@pytest.mark.parametrize(
    "shape",
    [
        (3, 33, 33),
        (2, 4, 32, 64),
        (8, 16, 64, 64),
        (32, 64, 128, 256),
        (16, 32, 64, 128),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "padding",
    [
        (1, 1, 1, 1),
        (2, 3, 2, 3),
        (3, 5, 3, 5),
        (0, 4, 0, 4),
    ],
)
def test_reflection_pad2d_benchmark_tensor(shape, dtype, padding):
    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.reflection_pad2d(ref_x, padding),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_reflection_pad2d(x, padding), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"reflection_pad2d {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.reflection_pad2d
@pytest.mark.parametrize(
    "shape",
    [
        (3, 33, 33),
        (2, 4, 32, 64),
        (8, 16, 64, 64),
        (32, 64, 128, 256),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "padding",
    [
        (1, 1, 1, 1),
        (2, 3, 2, 3),
        (3, 5, 3, 5),
        (0, 4, 0, 4),
    ],
)
def test_reflection_pad2d_benchmark_out(shape, dtype, padding):
    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()

    out_shape = list(shape)
    out_shape[-2] = out_shape[-2] + padding[2] + padding[3]
    out_shape[-1] = out_shape[-1] + padding[0] + padding[1]
    out_shape = tuple(out_shape)

    ref_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    act_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.reflection_pad2d.out(ref_x, padding, out=ref_out_buf),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_reflection_pad2d_out(x, padding, act_out_buf),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"reflection_pad2d {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")

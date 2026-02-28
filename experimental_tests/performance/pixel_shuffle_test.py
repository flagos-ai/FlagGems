# PIXEL_SHUFFLE operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.pixel_shuffle import pixel_shuffle as gems_pixel_shuffle
from flag_gems.experimental_ops.pixel_shuffle import (
    pixel_shuffle_out as gems_pixel_shuffle_out,
)

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from tests.accuracy_utils import TO_CPU, gems_assert_close
except ImportError:
    # Fallback values when running outside pytest
    TO_CPU = False  # fallback

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


def to_reference(inp, upcast=False):
    if inp is None:
        return None
    if TO_CPU:
        ref_inp = inp.to("cpu")
    else:
        ref_inp = inp.clone()
    if upcast:
        if ref_inp.is_complex():
            ref_inp = ref_inp.to(torch.complex128)
        else:
            ref_inp = ref_inp.to(torch.float64)
    return ref_inp


@pytest.mark.pixel_shuffle
@pytest.mark.parametrize(
    "shape_r",
    [
        ((1, 4, 2, 3), 2),
        ((2, 9, 4, 4), 3),
        ((4, 64, 32, 32), 2),
        ((2, 128, 64, 64), 2),
        ((1, 64, 16, 16), 4),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_pixel_shuffle_benchmark(shape_r, dtype):
    quantiles = [0.5, 0.2, 0.8]
    shape, r = shape_r
    n, c, h, w = shape
    x = torch.randn((n, c, h, w), dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()

    # PyTorch reference
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.pixel_shuffle(ref_x, r), rep=100, quantiles=quantiles
    )

    # FlagGems implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_pixel_shuffle(x, r), rep=100, quantiles=quantiles
        )

    speedup = ms_torch / ms_triton
    print(f"pixel_shuffle {shape} r={r} {dtype}:")
    print(f"  PyTorch: {ms_torch:.3f}ms")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.pixel_shuffle
@pytest.mark.parametrize(
    "shape_r",
    [
        ((1, 4, 2, 3), 2),
        ((2, 9, 4, 4), 3),
        ((4, 64, 32, 32), 2),
        ((2, 128, 64, 64), 2),
        ((1, 64, 16, 16), 4),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_pixel_shuffle_out_benchmark(shape_r, dtype):
    quantiles = [0.5, 0.2, 0.8]
    shape, r = shape_r
    n, c, h, w = shape
    x = torch.randn((n, c, h, w), dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()
    out_shape = (n, c // (r * r), h * r, w * r)
    ref_out = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    act_out = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)

    # PyTorch reference
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.pixel_shuffle.out(ref_x, r, out=ref_out),
        rep=100,
        quantiles=quantiles,
    )

    # FlagGems implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_pixel_shuffle_out(x, r, act_out), rep=100, quantiles=quantiles
        )

    speedup = ms_torch / ms_triton
    print(f"pixel_shuffle.out {shape} r={r} {dtype}:")
    print(f"  PyTorch: {ms_torch:.3f}ms")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")

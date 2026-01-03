# PIXEL_UNSHUFFLE operator test

import os
import sys

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
try:
    from tests.accuracy_utils import gems_assert_close
except ImportError:
    # Fallback values when running outside pytest

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.pixel_unshuffle import (
    pixel_unshuffle as gems_pixel_unshuffle,
)
from flag_gems.experimental_ops.pixel_unshuffle import (
    pixel_unshuffle_out as gems_pixel_unshuffle_out,
)


@pytest.mark.pixel_unshuffle
@pytest.mark.parametrize(
    "shape_factor", [((1, 3, 8, 8), 2), ((2, 4, 12, 6), 3), ((4, 16, 64, 48), 4)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_pixel_unshuffle_tensor(shape_factor, dtype):
    shape, downscale_factor = shape_factor
    input_tensor = torch.randn(shape, dtype=torch.float32, device=flag_gems.device).to(
        dtype
    )

    ref_input = input_tensor.clone()
    ref_out = torch.ops.aten.pixel_unshuffle(ref_input, downscale_factor)

    with flag_gems.use_gems():
        act_out = gems_pixel_unshuffle(input_tensor, downscale_factor)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.pixel_unshuffle
@pytest.mark.parametrize(
    "shape_factor", [((1, 3, 8, 8), 2), ((2, 4, 12, 6), 3), ((4, 16, 64, 48), 4)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_pixel_unshuffle_out(shape_factor, dtype):
    shape, downscale_factor = shape_factor
    N, C, H, W = shape
    r = downscale_factor
    out_shape = (N, C * (r * r), H // r, W // r)

    input_tensor = torch.randn(shape, dtype=torch.float32, device=flag_gems.device).to(
        dtype
    )
    ref_input = input_tensor.clone()

    out_ref = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    ref_out = torch.ops.aten.pixel_unshuffle.out(
        ref_input, downscale_factor, out=out_ref
    )

    out_act = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        act_out = gems_pixel_unshuffle_out(input_tensor, downscale_factor, out_act)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.pixel_unshuffle
@pytest.mark.parametrize(
    "shape_factor", [((1, 3, 8, 8), 2), ((2, 4, 12, 6), 3), ((4, 16, 64, 48), 4)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_pixel_unshuffle_benchmark_tensor(shape_factor, dtype):
    import torch.utils.benchmark as benchmark

    quantiles = [0.5, 0.2, 0.8]

    shape, downscale_factor = shape_factor
    input_tensor = torch.randn(shape, dtype=torch.float32, device=flag_gems.device).to(
        dtype
    )

    ref_input = input_tensor.clone()
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.pixel_unshuffle(ref_input, downscale_factor),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_pixel_unshuffle(input_tensor, downscale_factor),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"pixel_unshuffle {shape_factor} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.pixel_unshuffle
@pytest.mark.parametrize(
    "shape_factor", [((1, 3, 8, 8), 2), ((2, 4, 12, 6), 3), ((4, 16, 64, 48), 4)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_pixel_unshuffle_benchmark_out(shape_factor, dtype):
    quantiles = [0.5, 0.2, 0.8]

    shape, downscale_factor = shape_factor
    N, C, H, W = shape
    r = downscale_factor
    out_shape = (N, C * (r * r), H // r, W // r)

    input_tensor = torch.randn(shape, dtype=torch.float32, device=flag_gems.device).to(
        dtype
    )
    ref_input = input_tensor.clone()

    out_ref = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.pixel_unshuffle.out(
            ref_input, downscale_factor, out=out_ref
        ),
        rep=100,
        quantiles=quantiles,
    )

    out_act = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_pixel_unshuffle_out(input_tensor, downscale_factor, out_act),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"pixel_unshuffle {shape_factor} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")

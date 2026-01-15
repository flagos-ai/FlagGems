# _ADAPTIVE_AVG_POOL3D operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops._adaptive_avg_pool3d import (
    _adaptive_avg_pool3d as gems__adaptive_avg_pool3d,
)
from flag_gems.experimental_ops._adaptive_avg_pool3d import (
    _adaptive_avg_pool3d_out as gems__adaptive_avg_pool3d_out,
)

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from tests.accuracy_utils import TO_CPU, gems_assert_close  # noqa: E402
except ImportError:
    # Fallback values when running outside pytest
    TO_CPU = False  # fallback

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison aligned with flag_gems.testing.assert_close
        from flag_gems.testing import assert_close as fg_assert_close  # noqa: E402

        kwargs = dict(kwargs)
        reduce_dim = kwargs.pop("reduce_dim", 1)
        equal_nan = kwargs.pop("equal_nan", False)
        fg_assert_close(res, ref, dtype, equal_nan=equal_nan, reduce_dim=reduce_dim)


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


@pytest.mark.adaptive_avg_pool3d
@pytest.mark.parametrize(
    "shape,output_size",
    [
        ((1, 1, 4, 4, 4), (1, 1, 1)),
        ((1, 1, 4, 4, 4), (2, 2, 2)),
        ((1, 1, 4, 4, 4), (4, 4, 4)),
        ((2, 3, 8, 7, 6), (1, 1, 1)),
        ((2, 3, 8, 7, 6), (2, 3, 3)),
        ((2, 3, 8, 7, 6), (4, 5, 6)),
        ((4, 16, 16, 16, 16), (1, 1, 1)),
        ((4, 16, 16, 16, 16), (4, 4, 4)),
        ((4, 16, 16, 16, 16), (8, 8, 8)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test__adaptive_avg_pool3d_tensor(shape, output_size, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # Upcast reference for better numerical stability on CPU path
    ref_x = to_reference(x, upcast=True)

    ref_out = torch.ops.aten._adaptive_avg_pool3d(ref_x, output_size)

    with flag_gems.use_gems():
        act_out = gems__adaptive_avg_pool3d(x, output_size)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.adaptive_avg_pool3d
@pytest.mark.parametrize(
    "shape,output_size",
    [
        ((1, 1, 4, 4, 4), (1, 1, 1)),
        ((1, 1, 4, 4, 4), (2, 2, 2)),
        ((1, 1, 4, 4, 4), (4, 4, 4)),
        ((2, 3, 8, 7, 6), (1, 1, 1)),
        ((2, 3, 8, 7, 6), (2, 3, 3)),
        ((2, 3, 8, 7, 6), (4, 5, 6)),
        ((4, 16, 16, 16, 16), (1, 1, 1)),
        ((4, 16, 16, 16, 16), (4, 4, 4)),
        ((4, 16, 16, 16, 16), (8, 8, 8)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test__adaptive_avg_pool3d_out(shape, output_size, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # Upcast reference for better numerical stability on CPU path
    ref_x = to_reference(x, upcast=True)

    out_shape = (shape[0], shape[1], output_size[0], output_size[1], output_size[2])
    # Keep reference buffer aligned with upcast dtype
    ref_out_buf = torch.empty(out_shape, dtype=ref_x.dtype, device=ref_x.device)
    ref_out = torch.ops.aten._adaptive_avg_pool3d.out(
        ref_x, output_size, out=ref_out_buf
    )

    act_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        act_out = gems__adaptive_avg_pool3d_out(x, output_size, act_out_buf)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.adaptive_avg_pool3d
@pytest.mark.parametrize(
    "shape,output_size",
    [
        ((1, 1, 4, 4, 4), (1, 1, 1)),
        ((1, 1, 4, 4, 4), (2, 2, 2)),
        ((1, 1, 4, 4, 4), (4, 4, 4)),
        ((2, 3, 8, 7, 6), (1, 1, 1)),
        ((2, 3, 8, 7, 6), (2, 3, 3)),
        ((2, 3, 8, 7, 6), (4, 5, 6)),
        ((4, 16, 16, 16, 16), (1, 1, 1)),
        ((4, 16, 16, 16, 16), (4, 4, 4)),
        ((4, 16, 16, 16, 16), (8, 8, 8)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test__adaptive_avg_pool3d_benchmark_tensor(shape, output_size, dtype):
    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten._adaptive_avg_pool3d(ref_x, output_size),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems__adaptive_avg_pool3d(x, output_size),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"_adaptive_avg_pool3d {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.adaptive_avg_pool3d
@pytest.mark.parametrize(
    "shape,output_size",
    [
        ((1, 1, 4, 4, 4), (1, 1, 1)),
        ((1, 1, 4, 4, 4), (2, 2, 2)),
        ((1, 1, 4, 4, 4), (4, 4, 4)),
        ((2, 3, 8, 7, 6), (1, 1, 1)),
        ((2, 3, 8, 7, 6), (2, 3, 3)),
        ((2, 3, 8, 7, 6), (4, 5, 6)),
        ((4, 16, 16, 16, 16), (1, 1, 1)),
        ((4, 16, 16, 16, 16), (4, 4, 4)),
        ((4, 16, 16, 16, 16), (8, 8, 8)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test__adaptive_avg_pool3d_benchmark_out(shape, output_size, dtype):
    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()

    out_shape = (shape[0], shape[1], output_size[0], output_size[1], output_size[2])
    ref_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten._adaptive_avg_pool3d.out(
            ref_x, output_size, out=ref_out_buf
        ),
        rep=100,
        quantiles=quantiles,
    )

    act_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems__adaptive_avg_pool3d_out(x, output_size, act_out_buf),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"_adaptive_avg_pool3d {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")

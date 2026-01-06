# _UPSAMPLE_NEAREST_EXACT3D operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops._upsample_nearest_exact3d import (
    _upsample_nearest_exact3d as gems__upsample_nearest_exact3d,
)
from flag_gems.experimental_ops._upsample_nearest_exact3d import (
    _upsample_nearest_exact3d_out as gems__upsample_nearest_exact3d_out,
)

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from tests.accuracy_utils import gems_assert_close
except ImportError:
    # Fallback values when running outside pytest

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


@pytest.mark.upsample_nearest_exact3d
@pytest.mark.parametrize(
    "shape", [(1, 1, 3, 4, 5), (2, 3, 8, 9, 10), (4, 8, 16, 32, 32)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mults", [(1, 1, 1), (2, 2, 2)])
def test__upsample_nearest_exact3d_tensor(shape, dtype, mults):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    n, c, d, h, w = shape
    md, mh, mw = mults
    output_size = [d * md, h * mh, w * mw]

    ref_x = x.clone()
    ref_out = torch.ops.aten._upsample_nearest_exact3d(
        ref_x, output_size, None, None, None
    )

    with flag_gems.use_gems():
        act_out = gems__upsample_nearest_exact3d(x, output_size, None, None, None)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.upsample_nearest_exact3d
@pytest.mark.parametrize("shape", [(1, 4, 5, 6, 7), (2, 8, 6, 7, 8)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mults", [(2, 2, 2), (3, 1, 2)])
def test__upsample_nearest_exact3d_out(shape, dtype, mults):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    n, c, d, h, w = shape
    md, mh, mw = mults
    out_size = [d * md, h * mh, w * mw]

    ref_x = x.clone()
    ref_out_buf = torch.empty(
        (n, c, out_size[0], out_size[1], out_size[2]),
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_out = torch.ops.aten._upsample_nearest_exact3d.out(
        ref_x, out_size, None, None, None, out=ref_out_buf
    )

    act_out_buf = torch.empty_like(ref_out_buf)
    with flag_gems.use_gems():
        act_out = gems__upsample_nearest_exact3d_out(
            x, out_size, None, None, None, act_out_buf
        )

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.upsample_nearest_exact3d
@pytest.mark.parametrize("shape", [(1, 2, 5, 6, 7), (2, 4, 8, 10, 12)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", ["size", "scales"])
def test__upsample_nearest_exact3d_vec(shape, dtype, mode):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    n, c, d, h, w = shape

    if mode == "size":
        output_size = [d * 2, h * 3, w * 2]
        scale_factors = None
    else:
        output_size = None
        scale_factors = [2.0, 2.0, 2.0]

    ref_x = x.clone()
    ref_out = torch.ops.aten._upsample_nearest_exact3d.vec(
        ref_x, output_size, scale_factors
    )

    with flag_gems.use_gems():
        act_out = torch.ops.aten._upsample_nearest_exact3d.vec(
            x, output_size, scale_factors
        )

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.upsample_nearest_exact3d
@pytest.mark.parametrize(
    "shape", [(1, 1, 3, 4, 5), (2, 3, 8, 9, 10), (4, 8, 16, 32, 32)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mults", [(1, 1, 1), (2, 2, 2)])
def test__upsample_nearest_exact3d_benchmark_tensor(shape, dtype, mults):
    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    n, c, d, h, w = shape
    md, mh, mw = mults
    output_size = [d * md, h * mh, w * mw]

    ref_x = x.clone()
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten._upsample_nearest_exact3d(
            ref_x, output_size, None, None, None
        ),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems__upsample_nearest_exact3d(x, output_size, None, None, None),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"_upsample_nearest_exact3d {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.upsample_nearest_exact3d
@pytest.mark.parametrize("shape", [(1, 4, 5, 6, 7), (2, 8, 6, 7, 8)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mults", [(2, 2, 2), (3, 1, 2)])
def test__upsample_nearest_exact3d_benchmark_out(shape, dtype, mults):
    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    n, c, d, h, w = shape
    md, mh, mw = mults
    out_size = [d * md, h * mh, w * mw]

    ref_x = x.clone()
    ref_out_buf = torch.empty(
        (n, c, out_size[0], out_size[1], out_size[2]),
        dtype=dtype,
        device=flag_gems.device,
    )
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten._upsample_nearest_exact3d.out(
            ref_x, out_size, None, None, None, out=ref_out_buf
        ),
        rep=100,
        quantiles=quantiles,
    )

    act_out_buf = torch.empty_like(ref_out_buf)

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems__upsample_nearest_exact3d_out(
                x, out_size, None, None, None, act_out_buf
            ),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"_upsample_nearest_exact3d {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.upsample_nearest_exact3d
@pytest.mark.parametrize("shape", [(1, 2, 5, 6, 7), (2, 4, 8, 10, 12)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", ["size", "scales"])
def test__upsample_nearest_exact3d_vec_performance(shape, dtype, mode):
    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    n, c, d, h, w = shape

    if mode == "size":
        output_size = [d * 2, h * 3, w * 2]
        scale_factors = None
    else:
        output_size = None
        scale_factors = [2.0, 2.0, 2.0]

    ref_x = x.clone()
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten._upsample_nearest_exact3d.vec(
            ref_x, output_size, scale_factors
        ),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: torch.ops.aten._upsample_nearest_exact3d.vec(
                x, output_size, scale_factors
            ),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"_upsample_nearest_exact3d {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")

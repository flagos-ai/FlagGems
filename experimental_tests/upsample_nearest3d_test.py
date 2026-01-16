# UPSAMPLE_NEAREST3D operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.upsample_nearest3d import (
    upsample_nearest3d as gems_upsample_nearest3d,
)
from flag_gems.experimental_ops.upsample_nearest3d import (
    upsample_nearest3d_out as gems_upsample_nearest3d_out,
)

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
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


@pytest.mark.upsample_nearest3d
@pytest.mark.parametrize("shape", [(1, 1, 2, 3, 4), (2, 3, 4, 8, 8), (4, 8, 8, 16, 16)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("output_size", [(2, 3, 4), (5, 7, 9), (16, 32, 32)])
def test_upsample_nearest3d_base(shape, dtype, output_size):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.ops.aten.upsample_nearest3d(ref_inp, output_size)

    with flag_gems.use_gems():
        act_out = gems_upsample_nearest3d(inp, output_size)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.upsample_nearest3d
@pytest.mark.parametrize("shape", [(1, 1, 2, 3, 4), (2, 3, 4, 8, 8), (4, 8, 8, 16, 16)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_size", [True, False])
@pytest.mark.parametrize("output_size", [(3, 5, 6)])
@pytest.mark.parametrize("scale_factors", [[2.0, 2.0, 2.0], [1.5, 2.0, 3.0]])
def test_upsample_nearest3d_vec(shape, dtype, use_size, output_size, scale_factors):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    if use_size:
        ref_out = torch.ops.aten.upsample_nearest3d.vec(ref_inp, output_size, None)
        with flag_gems.use_gems():
            act_out = torch.ops.aten.upsample_nearest3d.vec(inp, output_size, None)
    else:
        ref_out = torch.ops.aten.upsample_nearest3d.vec(ref_inp, None, scale_factors)
        with flag_gems.use_gems():
            act_out = torch.ops.aten.upsample_nearest3d.vec(inp, None, scale_factors)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.upsample_nearest3d
@pytest.mark.parametrize("shape", [(1, 1, 2, 3, 4), (2, 3, 4, 8, 8), (4, 8, 8, 16, 16)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("output_size", [(2, 3, 4), (6, 10, 12), (12, 24, 24)])
def test_upsample_nearest3d_out(shape, dtype, output_size):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    N, C, _, _, _ = shape
    out_shape = (N, C, output_size[0], output_size[1], output_size[2])

    ref_out_buf = torch.empty(out_shape, dtype=dtype, device=ref_inp.device)
    act_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)

    ref_out = torch.ops.aten.upsample_nearest3d.out(
        ref_inp, output_size, None, None, None, out=ref_out_buf
    )

    with flag_gems.use_gems():
        act_out = gems_upsample_nearest3d_out(
            inp, output_size, None, None, None, act_out_buf
        )

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.upsample_nearest3d
@pytest.mark.parametrize("shape", [(1, 1, 2, 3, 4), (2, 3, 4, 8, 8), (4, 8, 8, 16, 16)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("output_size", [(2, 3, 4), (5, 7, 9), (16, 32, 32)])
def test_upsample_nearest3d_base_performance(shape, dtype, output_size):
    quantiles = [0.5, 0.2, 0.8]

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = inp.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.upsample_nearest3d(ref_inp, output_size),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_upsample_nearest3d(inp, output_size),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"upsample_nearest3d {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.upsample_nearest3d
@pytest.mark.parametrize("shape", [(1, 1, 2, 3, 4), (2, 3, 4, 8, 8), (4, 8, 8, 16, 16)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_size", [True, False])
@pytest.mark.parametrize("output_size", [(3, 5, 6)])
@pytest.mark.parametrize("scale_factors", [[2.0, 2.0, 2.0], [1.5, 2.0, 3.0]])
def test_upsample_nearest3d_vec_performance(
    shape, dtype, use_size, output_size, scale_factors
):
    quantiles = [0.5, 0.2, 0.8]

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = inp.clone()

    if use_size:
        # PyTorch reference implementation
        ms_torch, _, _ = triton.testing.do_bench(
            lambda: torch.ops.aten.upsample_nearest3d.vec(ref_inp, output_size, None),
            rep=100,
            quantiles=quantiles,
        )

        # Triton implementation
        with flag_gems.use_gems():
            ms_triton, _, _ = triton.testing.do_bench(
                lambda: torch.ops.aten.upsample_nearest3d.vec(inp, output_size, None),
                rep=100,
                quantiles=quantiles,
            )
    else:
        # PyTorch reference implementation
        ms_torch, _, _ = triton.testing.do_bench(
            lambda: torch.ops.aten.upsample_nearest3d.vec(ref_inp, None, scale_factors),
            rep=100,
            quantiles=quantiles,
        )

        # Triton implementation
        with flag_gems.use_gems():
            ms_triton, _, _ = triton.testing.do_bench(
                lambda: torch.ops.aten.upsample_nearest3d.vec(inp, None, scale_factors),
                rep=100,
                quantiles=quantiles,
            )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"upsample_nearest3d {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.upsample_nearest3d
@pytest.mark.parametrize("shape", [(1, 1, 2, 3, 4), (2, 3, 4, 8, 8), (4, 8, 8, 16, 16)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("output_size", [(2, 3, 4), (6, 10, 12), (12, 24, 24)])
def test_upsample_nearest3d_benchmark_out(shape, dtype, output_size):
    quantiles = [0.5, 0.2, 0.8]

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = inp.clone()

    N, C, _, _, _ = shape
    out_shape = (N, C, output_size[0], output_size[1], output_size[2])

    ref_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    act_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.upsample_nearest3d.out(
            ref_inp, output_size, None, None, None, out=ref_out_buf
        ),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_upsample_nearest3d_out(
                inp, output_size, None, None, None, act_out_buf
            ),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"upsample_nearest3d {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")

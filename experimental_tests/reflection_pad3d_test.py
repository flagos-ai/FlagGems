# REFLECTION_PAD3D operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.reflection_pad3d import (
    reflection_pad3d as gems_reflection_pad3d,
)
from flag_gems.experimental_ops.reflection_pad3d import (
    reflection_pad3d_out as gems_reflection_pad3d_out,
)

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from tests.accuracy_utils import TO_CPU, gems_assert_close
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


@pytest.mark.reflection_pad3d
@pytest.mark.parametrize(
    "shape", [(1, 1, 4, 5, 6), (2, 3, 17, 18, 19), (4, 8, 32, 33, 34)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "padding",
    [(0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1, 1), (0, 0, 2, 1, 3, 0), (2, 0, 1, 3, 0, 2)],
)
def test_reflection_pad3d_tensor(shape, dtype, padding):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_input = to_reference(input_tensor)
    ref_out = torch.ops.aten.reflection_pad3d(ref_input, padding)

    with flag_gems.use_gems():
        act_out = gems_reflection_pad3d(input_tensor, padding)

    gems_assert_close(act_out, ref_out, dtype=dtype, equal_nan=False, reduce_dim=1)


@pytest.mark.reflection_pad3d
@pytest.mark.parametrize(
    "shape", [(1, 1, 4, 5, 6), (2, 3, 17, 18, 19), (4, 8, 32, 33, 34)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "padding",
    [(0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1, 1), (0, 0, 2, 1, 3, 0), (2, 0, 1, 3, 0, 2)],
)
def test_reflection_pad3d_out_tensor(shape, dtype, padding):
    N, C, D, H, W = shape
    pad_w = padding[0] + padding[1]
    pad_h = padding[2] + padding[3]
    pad_d = padding[4] + padding[5]
    out_shape = (N, C, D + pad_d, H + pad_h, W + pad_w)

    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_input = to_reference(input_tensor)
    ref_out_buf = torch.empty(out_shape, dtype=dtype, device=ref_input.device)
    ref_out = torch.ops.aten.reflection_pad3d.out(ref_input, padding, out=ref_out_buf)

    with flag_gems.use_gems():
        act_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
        act_out = gems_reflection_pad3d_out(input_tensor, padding, act_out_buf)

    gems_assert_close(act_out, ref_out, dtype=dtype, equal_nan=False, reduce_dim=1)


@pytest.mark.reflection_pad3d
@pytest.mark.parametrize(
    "shape", [(1, 1, 4, 5, 6), (2, 3, 17, 18, 19), (4, 8, 32, 33, 34)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "padding",
    [(0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1, 1), (0, 0, 2, 1, 3, 0), (2, 0, 1, 3, 0, 2)],
)
def test_reflection_pad3d_benchmark_tensor(shape, dtype, padding):
    quantiles = [0.5, 0.2, 0.8]

    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_input = input_tensor.clone()
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.reflection_pad3d(ref_input, padding),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_reflection_pad3d(input_tensor, padding),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"reflection_pad3d {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.reflection_pad3d
@pytest.mark.parametrize(
    "shape", [(1, 1, 4, 5, 6), (2, 3, 17, 18, 19), (4, 8, 32, 33, 34)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "padding",
    [(0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1, 1), (0, 0, 2, 1, 3, 0), (2, 0, 1, 3, 0, 2)],
)
def test_reflection_pad3d_out_benchmark_tensor(shape, dtype, padding):
    quantiles = [0.5, 0.2, 0.8]

    N, C, D, H, W = shape
    pad_w = padding[0] + padding[1]
    pad_h = padding[2] + padding[3]
    pad_d = padding[4] + padding[5]
    out_shape = (N, C, D + pad_d, H + pad_h, W + pad_w)

    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_input = input_tensor.clone()
    ref_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.reflection_pad3d.out(
            ref_input, padding, out=ref_out_buf
        ),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        act_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_reflection_pad3d_out(input_tensor, padding, act_out_buf),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"reflection_pad3d {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")

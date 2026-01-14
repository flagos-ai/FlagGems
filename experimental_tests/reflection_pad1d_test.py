# REFLECTION_PAD1D operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.reflection_pad1d import (
    reflection_pad1d as gems_reflection_pad1d,
)
from flag_gems.experimental_ops.reflection_pad1d import (
    reflection_pad1d_out as gems_reflection_pad1d_out,
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


@pytest.mark.reflection_pad1d
@pytest.mark.parametrize("shape", [(3, 33), (2, 4, 64), (8, 16, 256), (32, 64, 2048)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("padding", [(1, 1), (3, 5), (8, 8)])
def test_reflection_pad1d_tensor(shape, dtype, padding):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x)

    ref_out = torch.ops.aten.reflection_pad1d(ref_x, padding)

    with flag_gems.use_gems():
        act_out = gems_reflection_pad1d(x, padding)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.reflection_pad1d
@pytest.mark.parametrize("shape", [(3, 33), (2, 4, 64), (32, 64, 2048)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("padding", [(1, 1), (3, 5), (8, 8)])
def test_reflection_pad1d_out(shape, dtype, padding):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x)

    out_shape = list(shape)
    out_shape[-1] = out_shape[-1] + padding[0] + padding[1]
    out_shape = tuple(out_shape)

    ref_out_buf = torch.empty(out_shape, dtype=dtype, device=ref_x.device)
    act_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)

    ref_out = torch.ops.aten.reflection_pad1d.out(ref_x, padding, out=ref_out_buf)

    with flag_gems.use_gems():
        act_out = gems_reflection_pad1d_out(x, padding, act_out_buf)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.reflection_pad1d
@pytest.mark.parametrize("shape", [(3, 33), (2, 4, 64), (8, 16, 256), (32, 64, 2048)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("padding", [(1, 1), (3, 5), (8, 8)])
def test_reflection_pad1d_benchmark_tensor(shape, dtype, padding):
    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.reflection_pad1d(ref_x, padding),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_reflection_pad1d(x, padding), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"reflection_pad1d {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.reflection_pad1d
@pytest.mark.parametrize("shape", [(3, 33), (2, 4, 64), (32, 64, 2048)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("padding", [(1, 1), (3, 5), (8, 8)])
def test_reflection_pad1d_benchmark_out(shape, dtype, padding):
    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()

    out_shape = list(shape)
    out_shape[-1] = out_shape[-1] + padding[0] + padding[1]
    out_shape = tuple(out_shape)

    ref_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    act_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.reflection_pad1d.out(ref_x, padding, out=ref_out_buf),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_reflection_pad1d_out(x, padding, act_out_buf),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"reflection_pad1d {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")

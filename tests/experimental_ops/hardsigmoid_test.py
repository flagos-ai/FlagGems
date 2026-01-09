# HARDSIGMOID operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.hardsigmoid import hardsigmoid as gems_hardsigmoid
from flag_gems.experimental_ops.hardsigmoid import (
    hardsigmoid_out as gems_hardsigmoid_out,
)

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from tests.accuracy_utils import gems_assert_close, TO_CPU
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


@pytest.mark.hardsigmoid
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_hardsigmoid_tensor(shape, dtype):
    x = torch.randn(shape, device=flag_gems.device, dtype=dtype)
    ref_x = to_reference(x)

    ref_out = torch.ops.aten.hardsigmoid(ref_x)

    with flag_gems.use_gems():
        act_out = gems_hardsigmoid(x)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.hardsigmoid
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_hardsigmoid_out(shape, dtype):
    x = torch.randn(shape, device=flag_gems.device, dtype=dtype)
    ref_x = to_reference(x)

    ref_out_buf = torch.empty_like(ref_x)
    act_out_buf = torch.empty_like(x)

    ref_out = torch.ops.aten.hardsigmoid.out(ref_x, out=ref_out_buf)

    with flag_gems.use_gems():
        act_out = gems_hardsigmoid_out(x, act_out_buf)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.hardsigmoid
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_hardsigmoid_benchmark_tensor(shape, dtype):
    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(shape, device=flag_gems.device, dtype=dtype)
    ref_x = x.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.hardsigmoid(ref_x), rep=100, quantiles=quantiles
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_hardsigmoid(x), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"hardsigmoid {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.hardsigmoid
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_hardsigmoid_benchmark_out(shape, dtype):
    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(shape, device=flag_gems.device, dtype=dtype)
    ref_x = x.clone()

    ref_out_buf = torch.empty_like(ref_x)
    act_out_buf = torch.empty_like(x)

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.hardsigmoid.out(ref_x, out=ref_out_buf),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_hardsigmoid_out(x, act_out_buf), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"hardsigmoid {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")

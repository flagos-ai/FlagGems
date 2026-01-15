# CEIL_ operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.ceil_ import ceil_ as gems_ceil_

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


@pytest.mark.ceil_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_ceil__tensor(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x)
    act_x = x.clone()

    ref_out = torch.ops.aten.ceil_(ref_x)

    with flag_gems.use_gems():
        act_out = gems_ceil_(act_x)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.ceil_
@pytest.mark.parametrize("shape", [(3, 2), (256, 128), (512, 256)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_ceil__tensor_noncontig(shape, dtype):
    base = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # Ensure reference follows TO_CPU behavior
    ref_view = to_reference(base).transpose(0, 1)
    act_base = base.clone()
    act_view = act_base.transpose(0, 1)

    ref_out = torch.ops.aten.ceil_(ref_view)

    with flag_gems.use_gems():
        act_out = gems_ceil_(act_view)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.ceil_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_ceil__benchmark_tensor(shape, dtype):
    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()
    act_x = x.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.ceil_(ref_x), rep=100, quantiles=quantiles
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_ceil_(act_x), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"ceil_ {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.ceil_
@pytest.mark.parametrize("shape", [(3, 2), (256, 128), (512, 256)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_ceil__tensor_noncontig_performance(shape, dtype):
    quantiles = [0.5, 0.2, 0.8]

    base = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_view = base.transpose(0, 1)
    act_base = base.clone()
    act_view = act_base.transpose(0, 1)

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.ceil_(ref_view), rep=100, quantiles=quantiles
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_ceil_(act_view), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"ceil_ {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")

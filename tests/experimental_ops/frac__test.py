# FRAC_ operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.frac_ import frac_ as gems_frac_

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from tests.accuracy_utils import gems_assert_close
except ImportError:
    # Fallback values when running outside pytest

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


@pytest.mark.frac_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_frac__tensor(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()

    ref_out = torch.ops.aten.frac_(ref_x)
    with flag_gems.use_gems():
        act_out = gems_frac_(x)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.frac_
@pytest.mark.parametrize("shape", [(64, 33), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_frac__tensor_noncontiguous(shape, dtype):
    base = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    base2 = base.clone()

    ref_inp = base.transpose(0, 1)
    act_inp = base2.transpose(0, 1)

    ref_out = torch.ops.aten.frac_(ref_inp)
    with flag_gems.use_gems():
        act_out = gems_frac_(act_inp)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.frac_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_frac__benchmark_tensor(shape, dtype):
    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.frac_(ref_x), rep=100, quantiles=quantiles
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_frac_(x), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"frac_ {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.frac_
@pytest.mark.parametrize("shape", [(64, 33), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_frac__tensor_noncontiguous_performance(shape, dtype):
    quantiles = [0.5, 0.2, 0.8]

    base = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    base2 = base.clone()

    ref_inp = base.transpose(0, 1)
    act_inp = base2.transpose(0, 1)

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.frac_(ref_inp), rep=100, quantiles=quantiles
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_frac_(act_inp), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"frac_ {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")

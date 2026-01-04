# ASINH_ operator test

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
import triton

import flag_gems
from flag_gems.experimental_ops.asinh_ import asinh_ as gems_asinh_

import torch


@pytest.mark.asinh_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_asinh__tensor(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = inp.clone()
    act_inp = inp.clone()

    ref_out = torch.ops.aten.asinh_(ref_inp)

    with flag_gems.use_gems():
        act_out = gems_asinh_(act_inp)

    gems_assert_close(act_out, ref_out, dtype=dtype)

@pytest.mark.asinh_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_asinh__benchmark_tensor(shape, dtype):
    import torch.utils.benchmark as benchmark

    quantiles = [0.5, 0.2, 0.8]

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = inp.clone()
    act_inp = inp.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.asinh_(ref_inp),
        rep=100,
        quantiles=quantiles
    )


    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_asinh_(act_inp),
            rep=100,
            quantiles=quantiles
        )


    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"asinh_ {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
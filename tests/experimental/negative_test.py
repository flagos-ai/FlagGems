# NEGATIVE operator test

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
from flag_gems.experimental_ops.negative import negative as gems_negative, negative_out as gems_negative_out

import torch

@pytest.mark.negative
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_negative_tensor(shape, dtype):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_input = input_tensor.clone()

    ref_out = torch.ops.aten.negative(ref_input)

    with flag_gems.use_gems():
        act_out = gems_negative(input_tensor)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.negative
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_negative_out(shape, dtype):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_input = input_tensor.clone()

    ref_out = torch.empty_like(ref_input, device=flag_gems.device)
    act_out = torch.empty_like(input_tensor, device=flag_gems.device)

    ref_res = torch.ops.aten.negative.out(ref_input, out=ref_out)

    with flag_gems.use_gems():
        act_res = gems_negative_out(input_tensor, act_out)

    gems_assert_close(act_res, ref_res, dtype=dtype)

@pytest.mark.negative
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_negative_benchmark_tensor(shape, dtype):
    import torch.utils.benchmark as benchmark

    quantiles = [0.5, 0.2, 0.8]

    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_input = input_tensor.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.negative(ref_input),
        rep=100,
        quantiles=quantiles
    )


    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_negative(input_tensor),
            rep=100,
            quantiles=quantiles
        )


    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"negative {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.negative
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_negative_benchmark_out(shape, dtype):

    quantiles = [0.5, 0.2, 0.8]

    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_input = input_tensor.clone()

    ref_out = torch.empty_like(ref_input, device=flag_gems.device)
    act_out = torch.empty_like(input_tensor, device=flag_gems.device)

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.negative.out(ref_input, out=ref_out),
        rep=100,
        quantiles=quantiles
    )


    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_negative_out(input_tensor, act_out),
            rep=100,
            quantiles=quantiles
        )


    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"negative {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
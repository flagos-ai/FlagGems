# RELU operator test

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
from flag_gems.experimental_ops.relu import relu as gems_relu, relu_out as gems_relu_out

import torch


@pytest.mark.relu
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_relu_tensor(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()
    ref_out = torch.ops.aten.relu(ref_x)
    with flag_gems.use_gems():
        act_out = gems_relu(x)
    gems_assert_close(act_out, ref_out, dtype=dtype)

@pytest.mark.relu
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_relu_out(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()
    ref_out_buf = torch.empty_like(ref_x)
    act_out_buf = torch.empty_like(x)
    ref_out = torch.ops.aten.relu.out(ref_x, out=ref_out_buf)
    with flag_gems.use_gems():
        act_out = gems_relu_out(x, act_out_buf)
    gems_assert_close(act_out, ref_out, dtype=dtype)

@pytest.mark.relu
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_relu_benchmark_tensor(shape, dtype):
    import torch.utils.benchmark as benchmark

    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.relu(ref_x),
        rep=100,
        quantiles=quantiles
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_relu(x),
            rep=100,
            quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"relu {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")

@pytest.mark.relu
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_relu_benchmark_out(shape, dtype):

    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()
    ref_out_buf = torch.empty_like(ref_x)
    act_out_buf = torch.empty_like(x)
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.relu.out(ref_x, out=ref_out_buf),
        rep=100,
        quantiles=quantiles
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_relu_out(x, act_out_buf),
            rep=100,
            quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"relu {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
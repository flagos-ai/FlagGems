# SOFTPLUS operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.softplus import softplus as gems_softplus
from flag_gems.experimental_ops.softplus import softplus_out as gems_softplus_out

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from tests.accuracy_utils import gems_assert_close
except ImportError:
    # Fallback values when running outside pytest

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


@pytest.mark.softplus
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("beta", [1.0, 2.0])
@pytest.mark.parametrize("threshold", [20.0, 10.0])
def test_softplus_tensor(shape, dtype, beta, threshold):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_input = input_tensor.clone()

    ref_out = torch.ops.aten.softplus(ref_input, beta, threshold)

    with flag_gems.use_gems():
        act_out = gems_softplus(input_tensor, beta, threshold)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.softplus
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("beta", [1.0, 2.0])
@pytest.mark.parametrize("threshold", [20.0, 10.0])
def test_softplus_out(shape, dtype, beta, threshold):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    out_ref = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    out_act = torch.empty(shape, dtype=dtype, device=flag_gems.device)

    ref_input = input_tensor.clone()

    ref_out = torch.ops.aten.softplus.out(ref_input, beta, threshold, out=out_ref)

    with flag_gems.use_gems():
        act_out = gems_softplus_out(input_tensor, beta, threshold, out_act)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.softplus
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("beta", [1.0, 2.0])
@pytest.mark.parametrize("threshold", [20.0, 10.0])
def test_softplus_benchmark_tensor(shape, dtype, beta, threshold):
    quantiles = [0.5, 0.2, 0.8]

    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_input = input_tensor.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.softplus(ref_input, beta, threshold),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_softplus(input_tensor, beta, threshold),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"softplus {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.softplus
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("beta", [1.0, 2.0])
@pytest.mark.parametrize("threshold", [20.0, 10.0])
def test_softplus_benchmark_out(shape, dtype, beta, threshold):
    quantiles = [0.5, 0.2, 0.8]

    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    out_ref = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    out_act = torch.empty(shape, dtype=dtype, device=flag_gems.device)

    ref_input = input_tensor.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.softplus.out(ref_input, beta, threshold, out=out_ref),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_softplus_out(input_tensor, beta, threshold, out_act),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"softplus {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")

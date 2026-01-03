# NATIVE_DROPOUT_BACKWARD operator test

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
from flag_gems.experimental_ops.native_dropout_backward import native_dropout_backward as gems_native_dropout_backward, native_dropout_backward_out as gems_native_dropout_backward_out

import torch



@pytest.mark.native_dropout_backward
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale", [1.0, 0.5, 2.0])
def test_native_dropout_backward_tensor(shape, dtype, scale):
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = (torch.rand(shape, device=flag_gems.device) > 0.3)

    ref_grad = grad_output.clone()
    ref_mask = mask.clone()
    ref_out = torch.ops.aten.native_dropout_backward(ref_grad, ref_mask, float(scale))

    act_grad = grad_output.clone()
    act_mask = mask.clone()
    with flag_gems.use_gems():
        act_out = gems_native_dropout_backward(act_grad, act_mask, float(scale))

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.native_dropout_backward
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale", [1.0, 0.5, 2.0])
def test_native_dropout_backward_out(shape, dtype, scale):
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = (torch.rand(shape, device=flag_gems.device) > 0.6)

    ref_grad = grad_output.clone()
    ref_mask = mask.clone()
    ref_out_tensor = torch.empty_like(ref_grad)
    ref_out = torch.ops.aten.native_dropout_backward.out(ref_grad, ref_mask, float(scale), out=ref_out_tensor)

    act_grad = grad_output.clone()
    act_mask = mask.clone()
    act_out_tensor = torch.empty_like(act_grad)
    with flag_gems.use_gems():
        act_out = gems_native_dropout_backward_out(act_grad, act_mask, float(scale), act_out_tensor)

    gems_assert_close(act_out, ref_out, dtype=dtype)
    gems_assert_close(act_out_tensor, ref_out_tensor, dtype=dtype)

@pytest.mark.native_dropout_backward
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale", [1.0, 0.5, 2.0])
def test_native_dropout_backward_benchmark_tensor(shape, dtype, scale):
    import torch.utils.benchmark as benchmark

    quantiles = [0.5, 0.2, 0.8]

    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = (torch.rand(shape, device=flag_gems.device) > 0.3)

    ref_grad = grad_output.clone()
    ref_mask = mask.clone()
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.native_dropout_backward(ref_grad, ref_mask, float(scale)),
        rep=100,
        quantiles=quantiles
    )

    act_grad = grad_output.clone()
    act_mask = mask.clone()

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_native_dropout_backward(act_grad, act_mask, float(scale)),
            rep=100,
            quantiles=quantiles
        )


    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"native_dropout_backward {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.native_dropout_backward
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale", [1.0, 0.5, 2.0])
def test_native_dropout_backward_benchmark_out(shape, dtype, scale):

    quantiles = [0.5, 0.2, 0.8]

    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = (torch.rand(shape, device=flag_gems.device) > 0.6)

    ref_grad = grad_output.clone()
    ref_mask = mask.clone()
    ref_out_tensor = torch.empty_like(ref_grad)
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.native_dropout_backward.out(ref_grad, ref_mask, float(scale), out=ref_out_tensor),
        rep=100,
        quantiles=quantiles
    )

    act_grad = grad_output.clone()
    act_mask = mask.clone()
    act_out_tensor = torch.empty_like(act_grad)

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_native_dropout_backward_out(act_grad, act_mask, float(scale), act_out_tensor),
            rep=100,
            quantiles=quantiles
        )


    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"native_dropout_backward {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"native_dropout_backward {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
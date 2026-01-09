# RRELU_WITH_NOISE_BACKWARD operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.rrelu_with_noise_backward import (
    rrelu_with_noise_backward as gems_rrelu_with_noise_backward,
)
from flag_gems.experimental_ops.rrelu_with_noise_backward import (
    rrelu_with_noise_backward_out as gems_rrelu_with_noise_backward_out,
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
    ref_inp = inp
    if TO_CPU:
        ref_inp = ref_inp.to("cpu")
    if upcast:
        if ref_inp.is_complex():
            ref_inp = ref_inp.to(torch.complex128)
        else:
            ref_inp = ref_inp.to(torch.float64)
    return ref_inp


@pytest.mark.rrelu_with_noise_backward
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("training", [True, False])
def test_rrelu_with_noise_backward_tensor(shape, dtype, training):
    lower = 0.1
    upper = 0.3
    self = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    noise = lower + (upper - lower) * torch.rand(
        shape, dtype=dtype, device=flag_gems.device
    )
    self_is_result = False

    ref_self = to_reference(self)
    ref_grad_output = to_reference(grad_output)
    ref_noise = to_reference(noise)

    act_self = self.clone()
    act_grad_output = grad_output.clone()
    act_noise = noise.clone()

    ref_out = torch.ops.aten.rrelu_with_noise_backward(
        ref_grad_output, ref_self, ref_noise, lower, upper, training, self_is_result
    )

    with flag_gems.use_gems():
        act_out = gems_rrelu_with_noise_backward(
            act_grad_output, act_self, act_noise, lower, upper, training, self_is_result
        )

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.rrelu_with_noise_backward
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("training", [True, False])
def test_rrelu_with_noise_backward_out(shape, dtype, training):
    lower = 0.1
    upper = 0.3
    self = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    noise = lower + (upper - lower) * torch.rand(
        shape, dtype=dtype, device=flag_gems.device
    )
    self_is_result = True

    ref_self = to_reference(self)
    ref_grad_output = to_reference(grad_output)
    ref_noise = to_reference(noise)
    ref_out_buf = torch.empty_like(ref_self)

    act_self = self.clone()
    act_grad_output = grad_output.clone()
    act_noise = noise.clone()
    act_out_buf = torch.empty_like(act_self)

    ref_out = torch.ops.aten.rrelu_with_noise_backward.out(
        ref_grad_output,
        ref_self,
        ref_noise,
        lower,
        upper,
        training,
        self_is_result,
        out=ref_out_buf,
    )

    with flag_gems.use_gems():
        act_out = gems_rrelu_with_noise_backward_out(
            act_grad_output,
            act_self,
            act_noise,
            lower,
            upper,
            training,
            self_is_result,
            act_out_buf,
        )

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.rrelu_with_noise_backward
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("training", [True, False])
def test_rrelu_with_noise_backward_benchmark_tensor(shape, dtype, training):
    quantiles = [0.5, 0.2, 0.8]

    lower = 0.1
    upper = 0.3
    self = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    noise = lower + (upper - lower) * torch.rand(
        shape, dtype=dtype, device=flag_gems.device
    )
    self_is_result = False

    ref_self = self.clone()
    ref_grad_output = grad_output.clone()
    ref_noise = noise.clone()

    act_self = self.clone()
    act_grad_output = grad_output.clone()
    act_noise = noise.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.rrelu_with_noise_backward(
            ref_grad_output, ref_self, ref_noise, lower, upper, training, self_is_result
        ),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_rrelu_with_noise_backward(
                act_grad_output,
                act_self,
                act_noise,
                lower,
                upper,
                training,
                self_is_result,
            ),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"rrelu_with_noise_backward {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.rrelu_with_noise_backward
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("training", [True, False])
def test_rrelu_with_noise_backward_benchmark_out(shape, dtype, training):
    quantiles = [0.5, 0.2, 0.8]

    lower = 0.1
    upper = 0.3
    self = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    noise = lower + (upper - lower) * torch.rand(
        shape, dtype=dtype, device=flag_gems.device
    )
    self_is_result = True

    ref_self = self.clone()
    ref_grad_output = grad_output.clone()
    ref_noise = noise.clone()
    ref_out_buf = torch.empty_like(ref_self)

    act_self = self.clone()
    act_grad_output = grad_output.clone()
    act_noise = noise.clone()
    act_out_buf = torch.empty_like(act_self)

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.rrelu_with_noise_backward.out(
            ref_grad_output,
            ref_self,
            ref_noise,
            lower,
            upper,
            training,
            self_is_result,
            out=ref_out_buf,
        ),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_rrelu_with_noise_backward_out(
                act_grad_output,
                act_self,
                act_noise,
                lower,
                upper,
                training,
                self_is_result,
                act_out_buf,
            ),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"rrelu_with_noise_backward {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")

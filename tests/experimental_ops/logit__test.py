# LOGIT_ operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.logit_ import logit_ as gems_logit_

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


@pytest.mark.logit_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_logit__inplace_no_eps(shape, dtype):
    base = torch.empty(shape, device=flag_gems.device, dtype=torch.float32).uniform_(
        -4.0, 4.0
    )
    input_tensor = torch.sigmoid(base).to(dtype=dtype)

    ref_input = to_reference(input_tensor, upcast=True)
    act_input = input_tensor.clone()

    ref_out = torch.ops.aten.logit_(ref_input)
    with flag_gems.use_gems():
        act_out = gems_logit_(act_input)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.logit_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("eps", [1e-3, 1e-2, 0.2])
def test_logit__inplace_with_eps(shape, dtype, eps):
    base = torch.empty(shape, device=flag_gems.device, dtype=torch.float32).uniform_(
        -0.5, 1.5
    )
    input_tensor = base.to(dtype=dtype)

    ref_input = to_reference(input_tensor, upcast=True)
    act_input = input_tensor.clone()

    ref_out = torch.ops.aten.logit_(ref_input, eps)
    with flag_gems.use_gems():
        act_out = gems_logit_(act_input, eps)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.logit_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_logit__inplace_no_eps_performance(shape, dtype):
    quantiles = [0.5, 0.2, 0.8]

    base = torch.empty(shape, device=flag_gems.device, dtype=torch.float32).uniform_(
        -4.0, 4.0
    )
    input_tensor = torch.sigmoid(base).to(dtype=dtype)

    ref_input = input_tensor.clone()
    act_input = input_tensor.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.logit_(ref_input), rep=100, quantiles=quantiles
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_logit_(act_input), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"logit_ {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.logit_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("eps", [1e-3, 1e-2, 0.2])
def test_logit__inplace_with_eps_performance(shape, dtype, eps):
    quantiles = [0.5, 0.2, 0.8]

    base = torch.empty(shape, device=flag_gems.device, dtype=torch.float32).uniform_(
        -0.5, 1.5
    )
    input_tensor = base.to(dtype=dtype)

    ref_input = input_tensor.clone()
    act_input = input_tensor.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.logit_(ref_input, eps), rep=100, quantiles=quantiles
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_logit_(act_input, eps), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"logit_ {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")

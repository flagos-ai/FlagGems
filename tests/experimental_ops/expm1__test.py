# EXPM1_ operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.expm1_ import expm1_ as gems_expm1_

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
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


@pytest.mark.expm1_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("noncontig", [False, True])
def test_expm1__tensor(shape, dtype, noncontig):
    base = torch.randn(shape, device=flag_gems.device, dtype=dtype) * 0.5
    base_ref = to_reference(base)
    base_act = base.clone()

    ref_input = base_ref.transpose(0, 1) if noncontig else base_ref
    act_input = base_act.transpose(0, 1) if noncontig else base_act

    ref_out = torch.ops.aten.expm1_(ref_input)

    with flag_gems.use_gems():
        act_out = gems_expm1_(act_input)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.expm1_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("noncontig", [False, True])
def test_expm1__benchmark_tensor(shape, dtype, noncontig):
    quantiles = [0.5, 0.2, 0.8]

    base = torch.randn(shape, device=flag_gems.device, dtype=dtype) * 0.5
    base_ref = base.clone()
    base_act = base.clone()

    ref_input = base_ref.transpose(0, 1) if noncontig else base_ref
    act_input = base_act.transpose(0, 1) if noncontig else base_act

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.expm1_(ref_input), rep=100, quantiles=quantiles
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_expm1_(act_input), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"expm1_ {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")

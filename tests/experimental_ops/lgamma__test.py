# LGAMMA_ operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.lgamma_ import lgamma_ as gems_lgamma_

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


@pytest.mark.lgamma_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("noncontig", [False, True])
def test_lgamma__tensor(shape, dtype, noncontig):
    # create stable inputs avoiding singularities (0, negative integers)
    u1 = torch.rand(shape, device=flag_gems.device, dtype=dtype)
    u2 = torch.rand(shape, device=flag_gems.device, dtype=dtype)
    pos = 0.1 + 2.9 * u1
    neg = -0.1 - 0.8 * u2
    mask = torch.rand(shape, device=flag_gems.device) > 0.5
    base = torch.where(mask, pos, neg)

    if noncontig:
        base = base.t()

    ref_input = to_reference(base)
    act_input = base.clone()

    ref_out = torch.ops.aten.lgamma_(ref_input)

    with flag_gems.use_gems():
        act_out = gems_lgamma_(act_input)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.lgamma_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("noncontig", [False, True])
def test_lgamma__benchmark_tensor(shape, dtype, noncontig):
    quantiles = [0.5, 0.2, 0.8]

    # create stable inputs avoiding singularities (0, negative integers)
    u1 = torch.rand(shape, device=flag_gems.device, dtype=dtype)
    u2 = torch.rand(shape, device=flag_gems.device, dtype=dtype)
    pos = 0.1 + 2.9 * u1
    neg = -0.1 - 0.8 * u2
    mask = torch.rand(shape, device=flag_gems.device) > 0.5
    base = torch.where(mask, pos, neg)

    if noncontig:
        base = base.t()

    ref_input = base.clone()
    act_input = base.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.lgamma_(ref_input), rep=100, quantiles=quantiles
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_lgamma_(act_input), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"lgamma_ {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")

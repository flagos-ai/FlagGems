# VIEW_AS_REAL operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.view_as_real import view_as_real as gems_view_as_real

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


@pytest.mark.view_as_real
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("cdtype", [torch.complex64, torch.complex128])
def test_view_as_real_tensor(shape, cdtype):
    if cdtype == torch.complex64:
        real_dtype = torch.float32
    else:
        real_dtype = torch.float64

    real_part = torch.randn(shape, dtype=real_dtype, device=flag_gems.device)
    imag_part = torch.randn(shape, dtype=real_dtype, device=flag_gems.device)
    input_tensor = torch.complex(real_part, imag_part)

    ref_input = to_reference(input_tensor)
    act_input = input_tensor.clone()

    ref_out = torch.ops.aten.view_as_real(ref_input)

    with flag_gems.use_gems():
        act_out = gems_view_as_real(act_input)

    gems_assert_close(act_out, ref_out, dtype=real_dtype)


@pytest.mark.view_as_real
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("cdtype", [torch.complex64, torch.complex128])
def test_view_as_real_benchmark_tensor(shape, cdtype):
    quantiles = [0.5, 0.2, 0.8]

    if cdtype == torch.complex64:
        real_dtype = torch.float32
    else:
        real_dtype = torch.float64

    real_part = torch.randn(shape, dtype=real_dtype, device=flag_gems.device)
    imag_part = torch.randn(shape, dtype=real_dtype, device=flag_gems.device)
    input_tensor = torch.complex(real_part, imag_part)

    ref_input = input_tensor.clone()
    act_input = input_tensor.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.view_as_real(ref_input), rep=100, quantiles=quantiles
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_view_as_real(act_input), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"view_as_real {shape} {cdtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")

# ADDCDIV operator test

import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops.addcdiv import addcdiv as gems_addcdiv
from flag_gems.experimental_ops.addcdiv import addcdiv_out as gems_addcdiv_out

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from tests.accuracy_utils import TO_CPU, gems_assert_close  # noqa: E402
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


@pytest.mark.addcdiv
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("value", [1.0, 0.5, 2.0, -1.25])
def test_addcdiv_tensor(shape, dtype, value):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    a = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    mags = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 1.0 + 0.5
    sgn = (torch.randint(0, 2, shape, device=flag_gems.device) * 2 - 1).to(dtype)
    b = mags * sgn

    ref_x = to_reference(x)
    ref_a = to_reference(a)
    ref_b = to_reference(b)

    ref_out = torch.ops.aten.addcdiv(ref_x, ref_a, ref_b, value=value)

    with flag_gems.use_gems():
        act_out = gems_addcdiv(x, a, b, value=value)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.addcdiv
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("value", [1.0, 0.5, 2.0, -1.25])
def test_addcdiv_out(shape, dtype, value):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    a = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    mags = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 1.0 + 0.5
    sgn = (torch.randint(0, 2, shape, device=flag_gems.device) * 2 - 1).to(dtype)
    b = mags * sgn

    ref_x = to_reference(x)
    ref_a = to_reference(a)
    ref_b = to_reference(b)
    ref_out_buf = torch.empty_like(ref_x)

    ref_out = torch.ops.aten.addcdiv.out(
        ref_x, ref_a, ref_b, value=value, out=ref_out_buf
    )

    act_out_buf = torch.empty_like(x)
    with flag_gems.use_gems():
        act_out = gems_addcdiv_out(x, a, b, value=value, out=act_out_buf)

    gems_assert_close(act_out, ref_out, dtype=dtype)

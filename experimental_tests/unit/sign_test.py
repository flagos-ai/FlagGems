# SIGN operator test

import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops.sign import sign as gems_sign
from flag_gems.experimental_ops.sign import sign_out as gems_sign_out

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


@pytest.mark.sign
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_sign_tensor(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x)
    ref_out = torch.ops.aten.sign(ref_x)
    with flag_gems.use_gems():
        act_out = gems_sign(x)
    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.sign
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_sign_out_tensor(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x)
    ref_out_buf = torch.empty_like(ref_x)
    ref_out = torch.ops.aten.sign.out(ref_x, out=ref_out_buf)
    out_buf = torch.empty_like(x)
    with flag_gems.use_gems():
        act_out = gems_sign_out(x, out_buf)
    gems_assert_close(act_out, ref_out, dtype=dtype)

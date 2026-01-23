# I0 operator test

import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops.i0 import i0 as gems_i0
from flag_gems.experimental_ops.i0 import i0_out as gems_i0_out

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


@pytest.mark.i0
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_i0_tensor(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = to_reference(x)
    ref_out = torch.ops.aten.i0(ref_x)

    with flag_gems.use_gems():
        act_out = gems_i0(x)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.i0
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_i0_out(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = to_reference(x)
    ref_out = torch.empty_like(ref_x)
    torch.ops.aten.i0.out(ref_x, out=ref_out)

    act_out = torch.empty_like(x)
    with flag_gems.use_gems():
        gems_i0_out(x, act_out)

    gems_assert_close(act_out, ref_out, dtype=dtype)

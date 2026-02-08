# HEAVISIDE operator test

import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops.heaviside import heaviside as gems_heaviside
from flag_gems.experimental_ops.heaviside import heaviside_out as gems_heaviside_out

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


@pytest.mark.heaviside
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_heaviside_tensor(shape, dtype):
    self_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    values_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.rand(shape, device=flag_gems.device) < 0.1
    self_tensor[mask] = 0.0

    ref_self = to_reference(self_tensor)
    ref_values = to_reference(values_tensor)

    ref_out = torch.ops.aten.heaviside(ref_self, ref_values)

    with flag_gems.use_gems():
        act_out = gems_heaviside(self_tensor, values_tensor)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.heaviside
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_heaviside_out(shape, dtype):
    self_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    values_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.rand(shape, device=flag_gems.device) < 0.1
    self_tensor[mask] = 0.0

    ref_self = to_reference(self_tensor)
    ref_values = to_reference(values_tensor)
    ref_out_buf = torch.empty_like(ref_self)

    ref_out = torch.ops.aten.heaviside.out(ref_self, ref_values, out=ref_out_buf)

    act_out_buf = torch.empty_like(self_tensor)
    with flag_gems.use_gems():
        act_out = gems_heaviside_out(self_tensor, values_tensor, act_out_buf)

    gems_assert_close(act_out, ref_out, dtype=dtype)

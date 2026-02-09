# DEG2RAD_ operator test

import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops.deg2rad_ import deg2rad_ as gems_deg2rad_

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


@pytest.mark.deg2rad_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_deg2rad__tensor(shape, dtype):
    base = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_input = to_reference(base)
    act_input = base.clone()

    ref_out = torch.ops.aten.deg2rad_(ref_input)

    with flag_gems.use_gems():
        act_out = gems_deg2rad_(act_input)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.deg2rad_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_deg2rad__noncontig(shape, dtype):
    base_ref = torch.randn((shape[1], shape[0]), dtype=dtype, device=flag_gems.device)
    base_act = base_ref.clone()
    # Place reference on CPU when TO_CPU is set
    ref_view = to_reference(base_ref).transpose(0, 1)
    act_view = base_act.transpose(0, 1)

    ref_out = torch.ops.aten.deg2rad_(ref_view)

    with flag_gems.use_gems():
        act_out = gems_deg2rad_(act_view)

    gems_assert_close(act_out, ref_out, dtype=dtype)

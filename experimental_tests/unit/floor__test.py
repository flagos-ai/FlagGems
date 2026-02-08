# FLOOR_ operator test

import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops.floor_ import floor_ as gems_floor_

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


@pytest.mark.floor_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_floor__tensor(shape, dtype):
    base = torch.randn(shape, dtype=dtype, device=flag_gems.device) * 5.3

    ref_input = to_reference(base)
    act_input = base.clone()

    ref_out = torch.ops.aten.floor_(ref_input)

    with flag_gems.use_gems():
        act_out = gems_floor_(act_input)

    gems_assert_close(act_out, ref_out, dtype=dtype)

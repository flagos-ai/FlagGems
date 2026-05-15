# LOGICAL_XOR_ operator test

import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops.logical_xor_ import logical_xor_ as gems_logical_xor_

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


@pytest.mark.logical_xor_
@pytest.mark.parametrize("dtype", [torch.bool])
@pytest.mark.parametrize(
    "shapes",
    [
        ((2, 3), (2, 3)),
        ((2, 3), (1, 3)),
        ((2, 3), (3,)),
        ((128, 256), (128, 256)),
        ((128, 256), (1, 1)),
        ((128, 256), (256,)),
        ((512, 512), (512, 512)),
        ((512, 512), (1,)),
    ],
)
def test_logical_xor__tensor(shapes, dtype):
    self_shape, other_shape = shapes
    self_input = (torch.rand(self_shape, device=flag_gems.device) > 0.5).to(dtype)
    other_input = (torch.rand(other_shape, device=flag_gems.device) > 0.5).to(dtype)

    ref_self = to_reference(self_input)
    ref_other = to_reference(other_input)
    ref_out = torch.ops.aten.logical_xor_(ref_self, ref_other)

    act_self = self_input.clone()
    act_other = other_input.clone()
    with flag_gems.use_gems():
        act_out = gems_logical_xor_(act_self, act_other)

    gems_assert_close(act_out, ref_out, dtype)

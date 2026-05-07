import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


# Issue #2840
@pytest.mark.polar
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_polar(shape, dtype):
    abs = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 5
    angle = (torch.rand(shape, dtype=dtype, device=flag_gems.device) - 0.5) * (
        8 * math.pi
    )
    ref_abs = utils.to_reference(abs)
    ref_angle = utils.to_reference(angle)
    ref_out = torch.polar(ref_abs, ref_angle)
    with flag_gems.use_gems():
        res_out = torch.polar(abs, angle)

    utils.gems_assert_close(res_out.real, ref_out.real, dtype)
    utils.gems_assert_close(res_out.imag, ref_out.imag, dtype)

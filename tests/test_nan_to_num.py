import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.nan_to_num
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("nan", [None, 0.0, 2.3])
@pytest.mark.parametrize("posinf", [None, 999.0])
@pytest.mark.parametrize("neginf", [None, -999.0])
def test_nan_to_num(shape, dtype, nan, posinf, neginf):
    base = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    base.view(-1)[0] = float("nan")
    if base.numel() > 1:
        base.view(-1)[1] = float("inf")
    if base.numel() > 2:
        base.view(-1)[2] = float("-inf")

    ref_input = utils.to_reference(base)
    ref_out = torch.nan_to_num(ref_input, nan=nan, posinf=posinf, neginf=neginf)

    with flag_gems.use_gems():
        res_out = torch.nan_to_num(base, nan=nan, posinf=posinf, neginf=neginf)

    utils.gems_assert_equal(res_out, ref_out)

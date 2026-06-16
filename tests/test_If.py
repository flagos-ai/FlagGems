import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.If
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_If(shape, dtype):
    condition = torch.randint(0, 2, shape, dtype=torch.bool, device=flag_gems.device)
    then_val = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else_val = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_condition = utils.to_reference(condition)
    ref_then_val = utils.to_reference(then_val)
    ref_else_val = utils.to_reference(else_val)

    ref_out = torch.where(ref_condition, ref_then_val, ref_else_val)
    with flag_gems.use_gems():
        res_out = flag_gems.If(condition, then_val, else_val)

    utils.gems_assert_equal(res_out, ref_out)

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.lshift
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_lshift(shape, dtype):
    inp1 = torch.randint(low=0, high=0x00FF, size=shape, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    inp2 = torch.randint(low=0, high=8, size=shape, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    # Reference: using bitwise_left_shift which is equivalent to __lshift__
    ref_out = torch.bitwise_left_shift(ref_inp1, ref_inp2)
    # FlagGems: using << operator which dispatches to aten.__lshift__
    with flag_gems.use_gems():
        res_out = inp1 << inp2

    utils.gems_assert_equal(res_out, ref_out)

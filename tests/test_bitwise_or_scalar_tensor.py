import random

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.bitwise_or_scalar_tensor
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES + utils.BOOL_TYPES)
def test_bitwise_or_scalar_tensor(shape, dtype):
    if dtype in utils.BOOL_TYPES:
        inp1 = bool(random.randint(0, 2))
        inp2 = torch.randint(0, 2, size=shape, dtype=torch.bool, device="cpu").to(
            flag_gems.device
        )
    else:
        inp1 = 0x00FF
        inp2 = torch.randint(
            low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cpu"
        ).to(flag_gems.device)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.bitwise_or(inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.bitwise_or(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)

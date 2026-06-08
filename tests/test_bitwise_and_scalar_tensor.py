import random

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

INT_DTYPES = [torch.int32] if QUICK_MODE else utils.INT_DTYPES + utils.BOOL_TYPES


@pytest.mark.bitwise_and_scalar_tensor
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_bitwise_and_scalar_tensor(shape, dtype):
    if dtype in utils.BOOL_TYPES:
        inp1 = bool(random.randint(0, 2))
        inp2 = torch.randint(0, 2, size=shape, dtype=dtype, device=flag_gems.device)
    else:
        inp1 = 0x00FF
        inp2 = torch.randint(
            low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cpu"
        ).to(flag_gems.device)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.bitwise_and(inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.bitwise_and(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)

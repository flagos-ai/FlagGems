import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

INT_DTYPES = [torch.int32] if QUICK_MODE else utils.INT_DTYPES + utils.BOOL_TYPES


@pytest.mark.bitwise_not
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_bitwise_not(shape, dtype):
    if dtype in utils.BOOL_TYPES:
        inp = torch.randint(0, 2, size=shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    else:
        inp = torch.randint(
            low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cpu"
        ).to(flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.bitwise_not(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.bitwise_not(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.bitwise_not_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_bitwise_not_(shape, dtype):
    if dtype in utils.BOOL_TYPES:
        res_inp = torch.randint(0, 2, size=shape, dtype=dtype, device=flag_gems.device)
    else:
        res_inp = torch.randint(
            low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cpu"
        ).to(flag_gems.device)
    ref_inp = utils.to_reference(res_inp.clone())

    ref_out = ref_inp.bitwise_not_()  # NOTE: there is no torch.bitwse_not_
    with flag_gems.use_gems():
        res_out = res_inp.bitwise_not_()

    utils.gems_assert_equal(res_out, ref_out)

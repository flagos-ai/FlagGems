import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.diag
@pytest.mark.parametrize("shape", utils.UT_SHAPES_1D + utils.UT_SHAPES_2D)
@pytest.mark.parametrize("diagonal", [-2, -1, 0, 1, 2])
@pytest.mark.parametrize(
    "dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES + utils.BOOL_TYPES
)
def test_diag(shape, diagonal, dtype):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    if dtype in utils.FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    elif dtype in utils.BOOL_TYPES:
        inp = torch.randint(0, 2, size=shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    else:
        inp = torch.randint(0, 0x7FFF, size=shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.diag(ref_inp, diagonal)
    with flag_gems.use_gems():
        res_out = torch.diag(inp, diagonal)

    utils.gems_assert_equal(res_out, ref_out)

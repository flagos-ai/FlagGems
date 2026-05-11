import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.logical_not
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype",
    utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES,
)
def test_logical_not(shape, dtype):
    if dtype in utils.ALL_FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    elif dtype in utils.ALL_INT_DTYPES:
        inp = torch.randint(-1000, 1000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    elif dtype in utils.BOOL_TYPES:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.logical_not(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.logical_not(inp)

    utils.gems_assert_equal(res_out, ref_out)

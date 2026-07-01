import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.reduce_min
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.ALL_INT_DTYPES)
def test_reduce_min(shape, dtype):
    if dtype in utils.FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.amin(ref_inp)
    with flag_gems.use_gems():
        res_out = flag_gems.reduce_min(inp)

    utils.gems_assert_equal(res_out, ref_out)

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

device = flag_gems.device


@pytest.mark.ones_like
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_ones_like(shape, dtype):
    inp = torch.empty(size=shape, dtype=dtype, device=device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.ones_like(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.ones_like(inp)

    utils.gems_assert_equal(res_out, ref_out)

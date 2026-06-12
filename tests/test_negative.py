import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.negative
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_negative(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.negative(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.negative(inp)

    utils.gems_assert_equal(res_out, ref_out)

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.squeeze_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_squeeze_copy(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.squeeze_copy(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.squeeze_copy(inp)
    utils.gems_assert_equal(res_out, ref_out)

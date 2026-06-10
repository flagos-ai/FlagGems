import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.new_ones
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_new_ones(shape, dtype):
    inp = torch.empty(size=shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_out = ref_inp.new_ones(shape)
    with flag_gems.use_gems():
        res_out = inp.new_ones(shape)
    utils.gems_assert_equal(res_out, ref_out)

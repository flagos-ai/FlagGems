import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.reduce_max
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_reduce_max(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.max(ref_inp)
    with flag_gems.use_gems():
        res_out = flag_gems.reduce_max(inp)

    utils.gems_assert_equal(res_out, ref_out)

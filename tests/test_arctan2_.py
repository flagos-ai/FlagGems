import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.arctan2_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arctan2_(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    y = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = ref_x.arctan2_(ref_y)

    with flag_gems.use_gems():
        res_out = x.arctan2_(y)

    utils.gems_assert_close(res_out, ref_out, dtype)

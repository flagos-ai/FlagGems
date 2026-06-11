import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.arctan2
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arctan2(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    y = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = torch.arctan2(ref_x, ref_y)

    with flag_gems.use_gems():
        res_out = torch.arctan2(x, y)

    utils.gems_assert_close(res_out, ref_out, dtype)

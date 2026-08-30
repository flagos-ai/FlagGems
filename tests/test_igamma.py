import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.igamma
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_igamma(shape, dtype):
    # igamma requires a > 0, x >= 0
    a = torch.randn(shape, dtype=dtype, device=flag_gems.device).abs() + 0.5
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device).abs()
    ref_a = utils.to_reference(a, True)
    ref_x = utils.to_reference(x, True)

    ref_out = torch.igamma(ref_a, ref_x)
    with flag_gems.use_gems():
        res_out = torch.igamma(a, x)

    utils.gems_assert_close(res_out, ref_out, dtype)

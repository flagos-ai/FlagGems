import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

device = flag_gems.device


@pytest.mark.rand_like
@pytest.mark.parametrize("shape", utils.DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_rand_like(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device=device)

    with flag_gems.use_gems():
        res_out = torch.rand_like(x)

    ref_out = utils.to_reference(res_out)

    assert (ref_out <= 1.0).all()
    assert (ref_out >= 0.0).all()

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.randint_like
@pytest.mark.parametrize("shape", utils.DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_randint_like(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res_out = torch.randint_like(x, 10)
    ref_out = utils.to_reference(res_out)
    assert (ref_out >= 0).all()
    assert (ref_out < 10).all()

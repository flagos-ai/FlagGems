import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.randint
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES)
def test_randint(shape, dtype):
    high = 100
    with flag_gems.use_gems():
        res_out = torch.randint(
            high=high, size=shape, dtype=dtype, device=flag_gems.device
        )
    assert res_out.shape == shape
    assert res_out.dtype == dtype
    assert (res_out >= 0).all()
    assert (res_out < high).all()

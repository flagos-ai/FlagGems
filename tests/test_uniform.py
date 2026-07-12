import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.uniform_
@pytest.mark.parametrize("shape", utils.DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_uniform(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        x.uniform_(-3, 3)

    assert (x <= 3.0).all()
    assert (x >= -3.0).all()

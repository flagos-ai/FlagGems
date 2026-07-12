import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.rand
@pytest.mark.parametrize("shape", utils.DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_rand(shape, dtype):
    with flag_gems.use_gems():
        res_out = torch.rand(shape, dtype=dtype, device=flag_gems.device)

    ref_out = utils.to_reference(res_out)

    assert (ref_out <= 1.0).all()
    assert (ref_out >= 0.0).all()

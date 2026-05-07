import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.randn_like
@pytest.mark.parametrize("shape", utils.DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_randn_like(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        res_out = torch.randn_like(x)

    ref_out = utils.to_reference(res_out)
    mean = torch.mean(ref_out)
    std = torch.std(ref_out)

    assert torch.abs(mean) < 0.01
    assert torch.abs(std - 1) < 0.01

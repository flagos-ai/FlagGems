import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

device = flag_gems.device


@pytest.mark.log_normal_
@pytest.mark.parametrize("shape", utils.DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_log_normal_(shape, dtype):
    mean_param = 1.0
    std_param = 2.0
    x = torch.empty(size=shape, dtype=dtype, device=device)
    with flag_gems.use_gems():
        x.log_normal_(mean_param, std_param)

    assert x.min() > 0

    x_ref = utils.to_reference(x)
    log_x = torch.log(x_ref.to(torch.float32))
    log_mean = torch.mean(log_x)
    log_std = torch.std(log_x)

    assert torch.abs(log_mean - mean_param) < 0.1
    assert torch.abs(log_std - std_param) < 0.2

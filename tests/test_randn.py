import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.randn
@pytest.mark.parametrize("shape", utils.DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_randn(shape, dtype):
    if flag_gems.vendor_name in ["cambricon", "iluvatar"]:
        torch.manual_seed(42)

    with flag_gems.use_gems():
        res_out = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_out = utils.to_reference(res_out).float()
    mean = torch.mean(ref_out)
    std = torch.std(ref_out)

    assert torch.abs(mean) < 0.01
    assert torch.abs(std - 1) < 0.01

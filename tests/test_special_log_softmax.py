import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_log_softmax
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_log_softmax(dtype):
    # Test with dim=1, which is the most common case
    x = torch.randn(32, 64, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = torch.special.log_softmax(ref_x, dim=1)
    with flag_gems.use_gems():
        res_out = torch.special.log_softmax(x, dim=1)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_log_softmax
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_log_softmax_large_n(dtype):
    # Test large N cases
    x = torch.randn(1, 8192, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = torch.special.log_softmax(ref_x, dim=1)
    with flag_gems.use_gems():
        res_out = torch.special.log_softmax(x, dim=1)
    utils.gems_assert_close(res_out, ref_out, dtype)

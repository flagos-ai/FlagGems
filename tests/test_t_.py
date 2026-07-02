import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.t_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_t_(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten.t_(ref_x)
    with flag_gems.use_gems():
        act_out = torch.ops.aten.t_(x)
    utils.gems_assert_close(act_out, ref_out, dtype)


@pytest.mark.t_
@pytest.mark.parametrize("shape", [(5,), (1024,), (8192,)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_t_1d(shape, dtype):
    # For 1D tensors, t_ should be a no-op
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten.t_(ref_x)
    with flag_gems.use_gems():
        act_out = torch.ops.aten.t_(x)
    utils.gems_assert_close(act_out, ref_out, dtype)

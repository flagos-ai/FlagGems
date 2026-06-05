import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES

# Make sure every thread has same seed.
random.seed(time.time() // 100)


@pytest.mark.mse_loss
@pytest.mark.parametrize("reduction", ["mean", "none", "sum"])
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mse_loss(shape, dtype, reduction):
    if flag_gems.vendor_name == "metax":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    dim = 1
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target, True)

    ref_out = torch.nn.functional.mse_loss(ref_inp, ref_target, reduction=reduction)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.mse_loss(inp, target, reduction=reduction)

    utils.gems_assert_close(
        res_out, ref_out, dtype, equal_nan=True, reduce_dim=shape[dim]
    )

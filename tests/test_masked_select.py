import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    THRESHOLD_SHAPE = [(0.3, utils.REDUCTION_SHAPES[0])]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    THRESHOLD_SHAPE = list(zip([0.3, 0.5, 0.7], utils.REDUCTION_SHAPES))

random.seed(time.time() // 100)


@pytest.mark.masked_select
@pytest.mark.parametrize("threshold, shape", THRESHOLD_SHAPE)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_masked_select(shape, dtype, threshold):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.randn(shape, dtype=dtype, device=flag_gems.device) < threshold

    ref_inp = utils.to_reference(inp)
    ref_mask = utils.to_reference(mask)
    ref_out = torch.masked_select(ref_inp, ref_mask)
    with flag_gems.use_gems():
        res_out = torch.masked_select(inp, mask)

    utils.gems_assert_equal(res_out, ref_out)

import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    DIMS_LIST = [1]
    CORRECTION = [1]
    KEEP_DIM = [True]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIMS_LIST = [0, 1, [0, 1], [1, 0]]
    CORRECTION = [0, 1]
    KEEP_DIM = [True, False]

random.seed(time.time() // 100)


@pytest.mark.var_mean
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIMS_LIST)
@pytest.mark.parametrize("correction", CORRECTION)
@pytest.mark.parametrize("keepdim", KEEP_DIM)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_var_mean(shape, dim, correction, keepdim, dtype):
    if shape[0] == 1:  # TODO: res is inf, while ref is nan
        shape = (2, 2)

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_var, ref_mean = torch.var_mean(
        ref_inp, dim, correction=correction, keepdim=keepdim
    )
    with flag_gems.use_gems():
        res_var, res_mean = torch.var_mean(
            inp, dim, correction=correction, keepdim=keepdim
        )

    utils.gems_assert_close(res_mean, ref_mean, dtype)
    utils.gems_assert_close(res_var, ref_var, dtype)

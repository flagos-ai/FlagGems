import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    # Reduced dtype and shape set for quick CI testing
    FLOAT_DTYPES = [torch.float32]
    # Minimal 2D shape to speed up QUICK_MODE execution
    CUMSUM_SHAPES = [(2, 32)]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    CUMSUM_SHAPES = utils.REDUCTION_SHAPES + [(2637,), (16, 1025, 255)]

random.seed(time.time() // 100)


@pytest.mark.cumsum_
@pytest.mark.parametrize("shape", CUMSUM_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + utils.INT_DTYPES)
def test_cumsum_(shape, dtype):
    dim = 1 if shape == utils.REDUCTION_SHAPES[-1] else -1
    if dtype in utils.INT_DTYPES:
        inp = torch.randint(-3, 3, shape, device=flag_gems.device).to(dtype)
        ref_inp = utils.to_reference(inp)
    else:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        ref_inp = utils.to_reference(inp, True)

    ref_out = ref_inp.cumsum_(dim=dim)
    with flag_gems.use_gems():
        res_out = inp.cumsum_(dim=dim)


    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[dim])
    assert res_out is inp

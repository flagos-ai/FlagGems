import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    NONZERO_SHAPES = [(2, 32)]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    NONZERO_SHAPES = utils.REDUCTION_SHAPES + [(2637,)]

random.seed(time.time() // 100)


@pytest.mark.nonzero
@pytest.mark.parametrize("shape", NONZERO_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + utils.INT_DTYPES + [torch.bool])
def test_nonzero(shape, dtype):
    if dtype == torch.bool:
        inp = torch.randint(0, 2, shape, dtype=torch.int, device=flag_gems.device).to(
            dtype
        )
    elif dtype in utils.INT_DTYPES:
        inp = torch.randint(-3, 3, shape, device=flag_gems.device).to(dtype)
    else:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, False)
    ref_out = torch.nonzero(ref_inp)

    with flag_gems.use_gems():
        res_out = torch.nonzero(inp)

    utils.gems_assert_equal(res_out, ref_out)

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

TRACE_SHAPES = [
    (1, 1),
    (5, 5),
    (10, 20),
    (30, 15),
    (1, 100),
    (100, 1),
    (128, 256),
    (256, 128),
    (0, 10),  # empty diagonal
    (10, 0),  # empty diagonal
    (1500, 1200),  # Larger shape
]

random.seed(time.time() // 100)


@pytest.mark.trace
@pytest.mark.parametrize("shape", TRACE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + utils.INT_DTYPES + [torch.bool])
def test_trace(shape, dtype):
    if dtype == torch.bool:
        inp = torch.randint(0, 2, size=shape, device=flag_gems.device).to(dtype)
    elif dtype in utils.INT_DTYPES:
        inp = torch.randint(-100, 100, size=shape, device=flag_gems.device).to(dtype)
    else:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    if ref_inp.device.type == "cpu" and dtype in [
        torch.half,
        torch.bfloat16,
        torch.bool,
    ]:
        ref_out = torch.sum(torch.diagonal(ref_inp))
    else:
        ref_out = torch.trace(ref_inp)

    with flag_gems.use_gems():
        res_out = torch.trace(inp)

    if dtype in FLOAT_DTYPES:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)

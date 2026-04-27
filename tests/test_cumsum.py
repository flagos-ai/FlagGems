import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    CUMSUM_SHAPES = [(2, 32)]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    CUMSUM_SHAPES = utils.REDUCTION_SHAPES + [(2637,), (16, 1025, 255)]

random.seed(time.time() // 100)


@pytest.mark.cumsum
@pytest.mark.parametrize("shape", CUMSUM_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + utils.INT_DTYPES)
def test_cumsum(shape, dtype):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    dim = 1 if shape == utils.REDUCTION_SHAPES[-1] else -1
    if dtype in utils.INT_DTYPES:
        inp = torch.randint(-3, 3, shape, device=flag_gems.device).to(dtype)
        ref_inp = utils.to_reference(inp)
    else:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        ref_inp = utils.to_reference(inp, True)

    ref_out = torch.cumsum(ref_inp, dim=dim)
    if flag_gems.vendor_name == "kunlunxin":
        from flag_gems.runtime.backend._kunlunxin import ops as kl_ops

        res_out = kl_ops.cumsum(inp, dim=dim)
    else:
        with flag_gems.use_gems():
            res_out = torch.cumsum(inp, dim=dim)

    # we should use ref's output type, since cumsum of int dtype results in int64
    if flag_gems.vendor_name in ["cambricon", "enflame"]:
        check_dtype = dtype
    elif dtype in utils.INT_DTYPES:
        check_dtype = ref_out.dtype
    else:
        check_dtype = dtype

    utils.gems_assert_close(res_out, ref_out, check_dtype, reduce_dim=shape[dim])

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    DIM_LIST = [1]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIM_LIST = [0, 1]


@pytest.mark.argmin
@pytest.mark.parametrize("shape", utils.REDUCTION_SMALL_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST + [None])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + utils.INT_DTYPES)
def test_argmin(shape, dim, keepdim, dtype):
    if dtype in utils.INT_DTYPES:
        inp = torch.randint(-1024, 1024, size=shape, device=flag_gems.device).to(dtype)
    else:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.argmin(ref_inp, dim=dim, keepdim=keepdim)

    with flag_gems.use_gems():
        res_out = torch.argmin(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(res_out, ref_out)

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    DIM_LIST = [1]
    KEEP_DIM = [True]
    ORD_LIST = [2]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIM_LIST = [0, 1, [0, 1], [1, 0]]
    KEEP_DIM = [True, False]
    ORD_LIST = [2, float("inf"), -float("inf"), 0, 1]


@pytest.mark.vector_norm
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("ord", ORD_LIST)
@pytest.mark.parametrize("keepdim", KEEP_DIM)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_vectornorm(shape, ord, dim, keepdim, dtype):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.linalg.vector_norm(ref_inp, ord, dim, keepdim)
    with flag_gems.use_gems():
        res_out = torch.linalg.vector_norm(inp, ord, dim, keepdim)

    utils.gems_assert_close(res_out, ref_out, dtype)

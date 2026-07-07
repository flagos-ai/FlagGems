import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    DIM_LIST = [None, 1]
    KEEP_DIM = [True]
    P_LIST = [2]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIM_LIST = [None, 0, 1, (0, 1), (1, 0)]
    KEEP_DIM = [True, False]
    P_LIST = [2, float("inf"), -float("inf"), 0, 1]


@pytest.mark.norm
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("p", P_LIST)
@pytest.mark.parametrize("keepdim", KEEP_DIM)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_norm(shape, p, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.norm(ref_inp, p=p, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.norm(inp, p=p, dim=dim, keepdim=keepdim)

    reduce_dim = inp.numel()
    if dim is not None:
        dims = [dim] if isinstance(dim, int) else list(dim)
        reduce_dim = 1
        for d in dims:
            reduce_dim *= shape[d % inp.ndim]
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim)

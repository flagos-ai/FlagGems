import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

# Quick mode uses minimal parameters; full mode covers various dims and keepdim
if QUICK_MODE:
    DIM_LIST = [[1]]
    KEEP_DIM = [True]
else:
    DIM_LIST = [[0], [1], [-1], [0, 1]]
    KEEP_DIM = [True, False]


@pytest.mark.frobenius_norm
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("keepdim", KEEP_DIM)
@pytest.mark.parametrize("dim", DIM_LIST)
def test_frobenius_norm(shape, dtype, keepdim, dim):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    # Filter invalid dim combinations
    if any(d >= len(shape) or d < -len(shape) for d in dim):
        pytest.skip("dim out of range for shape")

    ref_out = torch.ops.aten.frobenius_norm.dim(ref_inp, dim, keepdim)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.frobenius_norm.dim(inp, dim, keepdim)

    utils.gems_assert_close(res_out, ref_out, dtype)

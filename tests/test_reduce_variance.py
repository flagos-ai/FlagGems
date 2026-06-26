import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.reduce_variance
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", [0, 1, [0, 1], [1, 0]])
@pytest.mark.parametrize("correction", [0, 1])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_reduce_variance(shape, dim, correction, keepdim, dtype):
    # Avoid shapes where result is inf while reference is nan
    if shape[0] == 1:
        shape = (2, 2)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_var = torch.var(ref_inp, dim, correction=correction, keepdim=keepdim)
    with flag_gems.use_gems():
        res_var = torch.var(inp, dim, correction=correction, keepdim=keepdim)

    utils.gems_assert_close(res_var, ref_var, dtype)

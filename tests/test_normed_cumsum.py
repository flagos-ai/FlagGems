import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.normed_cumsum
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [0, -1])
def test_normed_cumsum(shape, dtype, dim):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.cumsum(ref_inp, dim=dim) / torch.sum(ref_inp, dim=dim, keepdim=True)
    with flag_gems.use_gems():
        res_out = flag_gems.normed_cumsum(inp, dim=dim)

    utils.gems_assert_close(res_out, ref_out, dtype)

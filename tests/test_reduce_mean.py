import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.reduce_mean
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_reduce_mean_without_dim(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.mean(ref_inp)
    with flag_gems.use_gems():
        res_out = flag_gems.reduce_mean(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.reduce_mean
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("keepdim, dim", [(True, 0), (False, 1)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_reduce_mean_dim(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.mean(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = flag_gems.reduce_mean(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_close(res_out, ref_out, dtype)

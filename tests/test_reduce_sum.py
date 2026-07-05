import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

KEEPDIM_DIM = list(zip([True, False], [0, 1]))


@pytest.mark.reduce_sum
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_reduce_sum_without_dim(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.sum(ref_inp)
    with flag_gems.use_gems():
        res_out = flag_gems.reduce_sum(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=inp.numel())


@pytest.mark.reduce_sum
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("keepdim, dim", KEEPDIM_DIM)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_reduce_sum_dim(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.sum(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = flag_gems.reduce_sum(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=inp.numel())

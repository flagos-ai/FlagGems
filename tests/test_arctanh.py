import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.arctanh
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arctanh(shape, dtype):
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(-0.99, 0.99)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.arctanh(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.arctanh(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.arctanh_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arctanh_(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 1.8 - 0.9
    ref_inp = utils.to_reference(inp.clone())

    ref_out = ref_inp.arctanh_()
    with flag_gems.use_gems():
        res_out = inp.arctanh_()

    utils.gems_assert_close(res_out, ref_out, dtype)

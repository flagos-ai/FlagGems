import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.erfinv
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_erfinv(shape, dtype):
    # erfinv input must be in range (-1, 1)
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(-0.99, 0.99)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.erfinv(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.erfinv(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.erfinv_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_erfinv_(shape, dtype):
    # erfinv_ input must be in range (-1, 1)
    torch.manual_seed(0)
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(-0.99, 0.99)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = ref_inp.erfinv_()
    with flag_gems.use_gems():
        res_out = inp.erfinv_()

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(inp, ref_inp, dtype)

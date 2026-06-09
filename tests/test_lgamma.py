import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.lgamma
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_lgamma(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    # lgamma is undefined for 0 and negative integers, so use positive values
    inp = inp + 0.1  # shift to avoid edge cases

    ref_inp = utils.to_reference(inp)
    ref_out = torch.lgamma(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.lgamma(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.lgamma_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_lgamma_(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    inp = inp + 0.1  # shift to avoid edge cases

    ref_inp = utils.to_reference(inp.clone())
    ref_out = ref_inp.lgamma_()
    with flag_gems.use_gems():
        res_out = inp.lgamma_()

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(inp, ref_inp, dtype)

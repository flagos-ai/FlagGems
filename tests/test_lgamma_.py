import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.lgamma
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_lgamma(shape, dtype):
    torch.manual_seed(0)
    inp = (
        torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.1
    )  # lgamma requires positive values
    ref_inp = utils.to_reference(inp)
    ref_out = ref_inp.lgamma()
    with flag_gems.use_gems():
        res_out = inp.lgamma()
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.lgamma_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_lgamma_(shape, dtype):
    torch.manual_seed(0)
    inp = (
        torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.1
    )  # lgamma requires positive values
    ref_inp = utils.to_reference(inp.clone())
    ref_out = ref_inp.lgamma_()
    with flag_gems.use_gems():
        res_out = inp.lgamma_()
    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(inp, ref_inp, dtype)

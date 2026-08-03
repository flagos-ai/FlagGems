import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.lgamma
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_lgamma(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device).abs() + 0.1
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.lgamma(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.lgamma(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.lgamma_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_lgamma_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device).abs() + 0.1
    ref_inp = utils.to_reference(inp.clone(), True)

    ref_out = torch.lgamma_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.lgamma_(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.sinc
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sinc(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.sinc(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.sinc(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.sinc_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sinc_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

    ref_out = torch.sinc_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.sinc_(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)

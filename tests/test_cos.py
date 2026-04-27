import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.cos
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cos(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.cos(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.cos(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cos_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cos_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

    ref_out = torch.cos_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.cos_(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)

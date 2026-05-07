import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.abs
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_abs(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.abs(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.abs(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.abs_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_abs_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.abs_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.abs_(inp)

    utils.gems_assert_equal(res_out, ref_out)

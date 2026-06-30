import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.reciprocal
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_reciprocal(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.reciprocal(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.reciprocal(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.reciprocal_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_reciprocal_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

    ref_out = torch.reciprocal_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.reciprocal_(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)

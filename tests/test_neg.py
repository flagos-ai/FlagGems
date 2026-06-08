import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

FLOAT_DTYPES = [torch.float32] if QUICK_MODE else utils.FLOAT_DTYPES


@pytest.mark.neg
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_neg(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.neg(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.neg(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.neg_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_neg_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.neg_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.neg_(inp)

    utils.gems_assert_equal(res_out, ref_out)
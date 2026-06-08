import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

FLOAT_DTYPES = [torch.float32] if QUICK_MODE else utils.FLOAT_DTYPES


@pytest.mark.sin
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_sin(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.sin(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.sin(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.sin_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_sin_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

    ref_out = torch.sin_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.sin_(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)

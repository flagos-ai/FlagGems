import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.sqrt
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_sqrt(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.sqrt(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.sqrt(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.sqrt_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_sqrt_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

    ref_out = torch.sqrt_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.sqrt_(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)

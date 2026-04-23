import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.square
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_square(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.square(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.square(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.square_
@pytest.mark.inplace
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_square_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.square_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.square_(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.square_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_square_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    out = torch.empty_like(inp)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.empty_like(ref_inp)

    torch.square(ref_inp, out=ref_out)
    with flag_gems.use_gems():
        torch.square(inp, out=out)

    utils.gems_assert_equal(out, ref_out)

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.ceil
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_ceil(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ceil(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.ceil(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.ceil_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_ceil_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = ref_inp.ceil_()
    with flag_gems.use_gems():
        res_out = inp.ceil_()

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.ceil_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_ceil_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    out = torch.empty_like(inp)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.empty_like(ref_inp)

    torch.ceil(ref_inp, out=ref_out)
    with flag_gems.use_gems():
        torch.ceil(inp, out=out)

    utils.gems_assert_equal(out, ref_out)

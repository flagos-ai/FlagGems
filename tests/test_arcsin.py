import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.arcsin
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arcsin(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.arcsin(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.arcsin(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, True)


@pytest.mark.arcsin_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arcsin_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.arcsin(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.arcsin_(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, True)
    utils.gems_assert_close(inp, ref_out, dtype, True)


@pytest.mark.arcsin_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arcsin_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.empty_like(ref_inp)
    torch.arcsin(ref_inp, out=ref_out)
    with flag_gems.use_gems():
        res_out = torch.empty_like(inp)
        torch.arcsin(inp, out=res_out)

    utils.gems_assert_close(res_out, ref_out, dtype, True)

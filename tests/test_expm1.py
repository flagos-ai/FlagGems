import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.expm1
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_expm1(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.expm1(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.expm1(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.expm1_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_expm1_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

    ref_out = torch.expm1_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.expm1_(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.expm1_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_expm1_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.empty_like(ref_inp)
    torch.expm1(ref_inp, out=ref_out)
    with flag_gems.use_gems():
        res_out = torch.empty_like(inp)
        torch.expm1(inp, out=res_out)

    utils.gems_assert_close(res_out, ref_out, dtype)

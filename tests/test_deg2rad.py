import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.deg2rad
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_deg2rad(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.deg2rad(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.deg2rad(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.deg2rad_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_deg2rad_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_inp.deg2rad_()
    with flag_gems.use_gems():
        inp.deg2rad_()

    utils.gems_assert_close(inp, ref_inp, dtype)


@pytest.mark.deg2rad_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_deg2rad_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.empty_like(ref_inp)
    torch.deg2rad(ref_inp, out=ref_out)

    res_out = torch.empty_like(inp)
    with flag_gems.use_gems():
        torch.deg2rad(inp, out=res_out)

    utils.gems_assert_close(res_out, ref_out, dtype)

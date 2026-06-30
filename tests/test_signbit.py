import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.signbit
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
def test_signbit(shape, dtype):
    inp = (
        torch.randn(shape, dtype=dtype, device=flag_gems.device)
        if dtype not in utils.INT_DTYPES
        else torch.randint(
            low=-100, high=100, size=shape, dtype=dtype, device=flag_gems.device
        )
    )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.signbit(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.signbit(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.signbit_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
def test_signbit_out(shape, dtype):
    inp = (
        torch.randn(shape, dtype=dtype, device=flag_gems.device)
        if dtype not in utils.INT_DTYPES
        else torch.randint(
            low=-100, high=100, size=shape, dtype=dtype, device=flag_gems.device
        )
    )
    ref_inp = utils.to_reference(inp)
    out = torch.empty_like(inp, dtype=torch.bool)
    ref_out = torch.empty_like(ref_inp, dtype=torch.bool)

    torch.signbit(ref_inp, out=ref_out)
    with flag_gems.use_gems():
        torch.signbit(inp, out=out)

    utils.gems_assert_equal(out, ref_out)

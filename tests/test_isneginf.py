import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.isneginf
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_isneginf(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = torch.masked_fill(inp, inp > 1.0, -float("inf"))
    ref_inp = utils.to_reference(inp)

    ref_out = torch.isneginf(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.isneginf(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.isneginf_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_isneginf_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = torch.masked_fill(inp, inp > 1.0, -float("inf"))
    ref_inp = utils.to_reference(inp)
    out = torch.empty_like(inp, dtype=torch.bool)
    ref_out = torch.empty_like(ref_inp, dtype=torch.bool)

    torch.isneginf(ref_inp, out=ref_out)
    with flag_gems.use_gems():
        torch.isneginf(inp, out=out)

    utils.gems_assert_equal(out, ref_out)

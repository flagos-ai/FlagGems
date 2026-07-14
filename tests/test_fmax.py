import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.fmax
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fmax(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.fmax(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.fmax(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.fmax_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fmax_out(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.empty_like(ref_inp1)
    torch.fmax(ref_inp1, ref_inp2, out=ref_out)
    with flag_gems.use_gems():
        res_out = torch.empty_like(inp1)
        torch.fmax(inp1, inp2, out=res_out)

    utils.gems_assert_equal(res_out, ref_out)

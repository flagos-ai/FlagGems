import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.greater
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_greater(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.greater(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.greater(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.greater_scalar
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_greater_scalar(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1)
    inp2 = 0

    ref_out = torch.greater(ref_inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.greater(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.greater_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_greater_out(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    out = torch.empty_like(inp1, dtype=torch.bool)

    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)
    ref_out = torch.empty_like(ref_inp1, dtype=torch.bool)

    torch.greater(ref_inp1, ref_inp2, out=ref_out)
    with flag_gems.use_gems():
        torch.greater(inp1, inp2, out=out)

    utils.gems_assert_equal(out, ref_out)


@pytest.mark.greater_scalar_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_greater_scalar_out(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = 0
    out = torch.empty_like(inp1, dtype=torch.bool)

    ref_inp1 = utils.to_reference(inp1)
    ref_out = torch.empty_like(ref_inp1, dtype=torch.bool)

    torch.greater(ref_inp1, inp2, out=ref_out)
    with flag_gems.use_gems():
        torch.greater(inp1, inp2, out=out)

    utils.gems_assert_equal(out, ref_out)

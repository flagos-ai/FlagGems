import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.greater_equal
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_greater_equal(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.greater_equal(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.greater_equal(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.greater_equal_scalar
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_greater_equal_scalar(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = 0
    ref_inp1 = utils.to_reference(inp1)

    ref_out = torch.greater_equal(ref_inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.greater_equal(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.greater_equal_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_greater_equal_(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1.clone())
    ref_inp2 = utils.to_reference(inp2.clone())

    ref_out = ref_inp1.greater_equal_(ref_inp2)
    with flag_gems.use_gems():
        res_out = inp1.greater_equal_(inp2)

    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(inp1, ref_inp1)

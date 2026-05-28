import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.true_divide
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_true_divide(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = torch.true_divide(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.true_divide(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.true_divide_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_true_divide_(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1.clone(), False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_inp1.true_divide_(ref_inp2)
    with flag_gems.use_gems():
        inp1.true_divide_(inp2)

    utils.gems_assert_close(inp1, ref_inp1, dtype, equal_nan=True)


@pytest.mark.true_divide_tensor_scalar
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_true_divide_tensor_scalar(shape, scalar, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar
    ref_inp1 = utils.to_reference(inp1, False)

    ref_out = torch.true_divide(ref_inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.true_divide(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.true_divide_scalar_tensor
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_true_divide_scalar_tensor(shape, scalar, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = torch.true_divide(inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.true_divide(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)

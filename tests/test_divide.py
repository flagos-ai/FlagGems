import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.divide
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_divide(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # Avoid division by zero
    inp2 = torch.where(inp2 == 0, torch.ones_like(inp2), inp2)

    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.divide(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.divide(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.inplace
@pytest.mark.divide_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_divide_(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # Avoid division by zero
    inp2 = torch.where(inp2 == 0, torch.ones_like(inp2), inp2)

    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = ref_inp1.divide_(ref_inp2)
    with flag_gems.use_gems():
        res_out = inp1.divide_(inp2)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.divide
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
def test_divide_tensor_scalar(shape, dtype, scalar):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # Avoid division by zero
    if scalar == 0:
        scalar = 1.0

    ref_inp1 = utils.to_reference(inp1, True)

    ref_out = torch.divide(ref_inp1, scalar)
    with flag_gems.use_gems():
        res_out = torch.divide(inp1, scalar)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.divide
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
def test_divide_scalar_tensor(shape, dtype, scalar):
    # Avoid division by zero
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.where(inp2 == 0, torch.ones_like(inp2), inp2)

    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.divide(scalar, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.divide(scalar, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.divide
# Scalar-scalar always returns float32 regardless of input
@pytest.mark.parametrize("dtype", [torch.float32])
def test_divide_scalar_scalar(dtype):
    # Avoid division by zero
    scalar1 = 5.0
    scalar2 = 2.0

    ref_out = torch.divide(scalar1, scalar2)
    with flag_gems.use_gems():
        res_out = torch.divide(scalar1, scalar2)

    utils.gems_assert_close(res_out, ref_out, dtype)

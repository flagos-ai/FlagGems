import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.hardtanh
@pytest.mark.parametrize("min_val,max_val", [(-1.0, 1.0), (0.0, 1.0), (-2.0, 2.0)])
@pytest.mark.parametrize("shape", [(64,), (16, 16), (8, 16, 16)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_hardtanh(shape, min_val, max_val, dtype):
    """Test hardtanh with various shapes and min/max values."""
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input)
    ref_output = torch.ops.aten.hardtanh(ref_input, min_val, max_val)

    with flag_gems.use_gems():
        res_output = torch.ops.aten.hardtanh(input, min_val, max_val)

    utils.gems_assert_close(res_output, ref_output, dtype)


@pytest.mark.hardtanh
@pytest.mark.parametrize("min_val,max_val", [(-1.0, 1.0), (0.0, 1.0)])
@pytest.mark.parametrize("shape", [(64,), (16, 16)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_hardtanh_inplace(shape, min_val, max_val, dtype):
    """Test hardtanh_ in-place operation."""
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    input_copy = input.clone()

    ref_input = utils.to_reference(input_copy)
    ref_output = torch.ops.aten.hardtanh_(ref_input, min_val, max_val)

    with flag_gems.use_gems():
        res_output = torch.ops.aten.hardtanh_(input, min_val, max_val)

    utils.gems_assert_close(res_output, ref_output, dtype)


@pytest.mark.hardtanh_backward
@pytest.mark.parametrize("min_val,max_val", [(-1.0, 1.0), (0.0, 1.0), (-2.0, 2.0)])
@pytest.mark.parametrize("shape", [(64,), (16, 16), (8, 16, 16)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_hardtanh_backward(shape, min_val, max_val, dtype):
    """Test hardtanh_backward with various shapes and min/max values."""
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input)
    ref_grad = utils.to_reference(grad_output)

    ref_grad_input = torch.ops.aten.hardtanh_backward(ref_grad, ref_input, min_val, max_val)

    with flag_gems.use_gems():
        res_grad_input = torch.ops.aten.hardtanh_backward(grad_output, input, min_val, max_val)

    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype)

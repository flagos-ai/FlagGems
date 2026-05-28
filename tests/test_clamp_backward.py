import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.clamp_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clamp_tensor_backward(shape, dtype):
    """Test clamp_tensor_backward with various shapes."""
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    min_val = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    max_val = min_val + torch.abs(torch.randn(shape, dtype=dtype, device=flag_gems.device)) + 0.5
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_min = utils.to_reference(min_val)
    ref_max = utils.to_reference(max_val)
    ref_grad = utils.to_reference(grad_output)

    ref_grad_input = torch.ops.aten.clamp.Tensor_backward(ref_grad, ref_x, ref_min, ref_max)

    with flag_gems.use_gems():
        res_grad_input = torch.ops.aten.clamp.Tensor_backward(grad_output, x, min_val, max_val)

    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype)


@pytest.mark.clamp_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clamp_min_tensor_backward(shape, dtype):
    """Test clamp_min_tensor_backward with various shapes."""
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    min_val = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_min = utils.to_reference(min_val)
    ref_grad = utils.to_reference(grad_output)

    ref_grad_input = torch.ops.aten.clamp_min.Tensor_backward(ref_grad, ref_x, ref_min)

    with flag_gems.use_gems():
        res_grad_input = torch.ops.aten.clamp_min.Tensor_backward(grad_output, x, min_val)

    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype)


@pytest.mark.clamp_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clamp_max_tensor_backward(shape, dtype):
    """Test clamp_max_tensor_backward with various shapes."""
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    max_val = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_max = utils.to_reference(max_val)
    ref_grad = utils.to_reference(grad_output)

    ref_grad_input = torch.ops.aten.clamp_max.Tensor_backward(ref_grad, ref_x, ref_max)

    with flag_gems.use_gems():
        res_grad_input = torch.ops.aten.clamp_max.Tensor_backward(grad_output, x, max_val)

    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype)

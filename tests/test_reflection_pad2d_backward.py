import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.reflection_pad2d_backward
@pytest.mark.parametrize("padding", [(1, 1, 1, 1), (2, 2, 0, 0), (0, 0, 3, 3), (3, 1, 2, 4)])
@pytest.mark.parametrize("shape", [(16, 16), (32, 64), (64, 32)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_reflection_pad2d_backward(shape, padding, dtype):
    """Test reflection_pad2d_backward with various shapes and padding."""
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad_output = torch.randn(
        shape[0] + padding[0] + padding[1],
        shape[1] + padding[2] + padding[3],
        dtype=dtype,
        device=flag_gems.device,
    )

    ref_input = utils.to_reference(input)
    ref_grad = utils.to_reference(grad_output)

    ref_grad_input = torch.ops.aten.reflection_pad2d_backward(ref_grad, ref_input, padding)

    with flag_gems.use_gems():
        res_grad_input = torch.ops.aten.reflection_pad2d_backward(grad_output, input, padding)

    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype)


@pytest.mark.reflection_pad2d_backward
@pytest.mark.parametrize("padding", [(1, 1, 1, 1), (2, 2, 0, 0), (0, 0, 3, 3)])
@pytest.mark.parametrize("shape", [(2, 8, 16, 16), (4, 16, 32, 32)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_reflection_pad2d_backward_batched(shape, padding, dtype):
    """Test reflection_pad2d_backward with batched input."""
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad_output = torch.randn(
        shape[0],
        shape[1],
        shape[2] + padding[0] + padding[1],
        shape[3] + padding[2] + padding[3],
        dtype=dtype,
        device=flag_gems.device,
    )

    ref_input = utils.to_reference(input)
    ref_grad = utils.to_reference(grad_output)

    ref_grad_input = torch.ops.aten.reflection_pad2d_backward(ref_grad, ref_input, padding)

    with flag_gems.use_gems():
        res_grad_input = torch.ops.aten.reflection_pad2d_backward(grad_output, input, padding)

    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype)

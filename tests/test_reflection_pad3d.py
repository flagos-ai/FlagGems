import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.reflection_pad3d
@pytest.mark.parametrize(
    "padding",
    [(1, 1, 1, 1, 1, 1), (2, 2, 0, 0, 0, 0), (0, 0, 3, 3, 0, 0), (3, 1, 2, 4, 1, 2)],
)
@pytest.mark.parametrize("shape", [(8, 8, 8), (16, 16, 16)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_reflection_pad3d(shape, padding, dtype):
    """Test reflection_pad3d with various shapes and padding."""
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input)
    ref_output = torch.ops.aten.reflection_pad3d(ref_input, padding)

    with flag_gems.use_gems():
        res_output = torch.ops.aten.reflection_pad3d(input, padding)

    utils.gems_assert_close(res_output, ref_output, dtype)


@pytest.mark.reflection_pad3d
@pytest.mark.parametrize("padding", [(1, 1, 1, 1, 1, 1), (2, 2, 0, 0, 0, 0), (0, 0, 3, 3, 0, 0)])
@pytest.mark.parametrize("shape", [(2, 8, 8, 8), (4, 16, 16, 16)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_reflection_pad3d_batched(shape, padding, dtype):
    """Test reflection_pad3d with batched input."""
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    pad_d, pad_h, pad_w = padding[4:], padding[2:4], padding[:2]
    output_shape = (
        shape[0],
        shape[1] + padding[4] + padding[5],
        shape[2] + padding[2] + padding[3],
        shape[3] + padding[0] + padding[1],
    )

    ref_input = utils.to_reference(input)
    ref_output = torch.ops.aten.reflection_pad3d(ref_input, padding)

    with flag_gems.use_gems():
        res_output = torch.ops.aten.reflection_pad3d(input, padding)

    utils.gems_assert_close(res_output, ref_output, dtype)


@pytest.mark.reflection_pad3d_backward
@pytest.mark.parametrize(
    "padding",
    [(1, 1, 1, 1, 1, 1), (2, 2, 0, 0, 0, 0), (0, 0, 3, 3, 0, 0), (3, 1, 2, 4, 1, 2)],
)
@pytest.mark.parametrize("shape", [(8, 8, 8), (16, 16, 16)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_reflection_pad3d_backward(shape, padding, dtype):
    """Test reflection_pad3d_backward with various shapes and padding."""
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad_output = torch.randn(
        shape[0] + padding[4] + padding[5],
        shape[1] + padding[2] + padding[3],
        shape[2] + padding[0] + padding[1],
        dtype=dtype,
        device=flag_gems.device,
    )

    ref_input = utils.to_reference(input)
    ref_grad = utils.to_reference(grad_output)

    ref_grad_input = torch.ops.aten.reflection_pad3d_backward(ref_grad, ref_input, padding)

    with flag_gems.use_gems():
        res_grad_input = torch.ops.aten.reflection_pad3d_backward(grad_output, input, padding)

    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype)

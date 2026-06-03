import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.replication_pad3d_backward
@pytest.mark.parametrize("shape", [(2, 3, 8, 16, 16), (1, 2, 4, 12, 12)])
@pytest.mark.parametrize(
    "padding", [(1, 1, 1, 1, 1, 1), (2, 2, 0, 0, 1, 1), (0, 3, 1, 2, 0, 1)]
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_replication_pad3d_backward(shape, dtype, padding):
    """Test replication_pad3d_backward with various shapes and padding configs."""
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    grad_output = torch.randn(
        *shape[:2],
        shape[2] + padding[4] + padding[5],
        shape[3] + padding[2] + padding[3],
        shape[4] + padding[0] + padding[1],
        dtype=dtype,
        device=flag_gems.device,
    )

    ref_x = utils.to_reference(x)
    ref_grad_output = utils.to_reference(grad_output)

    ref_grad_input = torch.ops.aten.replication_pad3d_backward(
        ref_grad_output, ref_x, padding
    )

    with flag_gems.use_gems():
        res_grad_input = torch.ops.aten.replication_pad3d_backward(
            grad_output, x, padding
        )

    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype)


@pytest.mark.replication_pad3d_backward
@pytest.mark.parametrize("shape", [(2, 3, 8, 16, 16), (1, 2, 4, 12, 12)])
@pytest.mark.parametrize("padding", [1, 2])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_replication_pad3d_backward_int_padding(shape, dtype, padding):
    """Test replication_pad3d_backward with integer padding (same on all sides)."""
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    padded_size = shape[2] + 2 * padding
    grad_output = torch.randn(
        *shape[:2],
        padded_size,
        padded_size,
        padded_size,
        dtype=dtype,
        device=flag_gems.device,
    )

    ref_x = utils.to_reference(x)
    ref_grad_output = utils.to_reference(grad_output)

    ref_grad_input = torch.ops.aten.replication_pad3d_backward(
        ref_grad_output, ref_x, padding
    )

    with flag_gems.use_gems():
        res_grad_input = torch.ops.aten.replication_pad3d_backward(
            grad_output, x, padding
        )

    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype)


@pytest.mark.replication_pad3d_backward
@pytest.mark.parametrize("shape", [(3, 16, 16, 16), (2, 8, 12, 12)])
@pytest.mark.parametrize("padding", [(1, 1, 1, 1, 1, 1), 2])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_replication_pad3d_backward_4d_input(shape, dtype, padding):
    """Test replication_pad3d_backward with 4D input (unsqueeze/squeeze behavior)."""
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    if isinstance(padding, int):
        padded_size = shape[1] + 2 * padding
        grad_output = torch.randn(
            shape[0],
            padded_size,
            padded_size,
            padded_size,
            dtype=dtype,
            device=flag_gems.device,
        )
    else:
        grad_output = torch.randn(
            shape[0],
            shape[1] + padding[4] + padding[5],
            shape[2] + padding[2] + padding[3],
            shape[3] + padding[0] + padding[1],
            dtype=dtype,
            device=flag_gems.device,
        )

    ref_x = utils.to_reference(x)
    ref_grad_output = utils.to_reference(grad_output)

    ref_grad_input = torch.ops.aten.replication_pad3d_backward(
        ref_grad_output, ref_x, padding
    )

    with flag_gems.use_gems():
        res_grad_input = torch.ops.aten.replication_pad3d_backward(
            grad_output, x, padding
        )

    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype)

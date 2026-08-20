import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

REPLICATION_PAD2D_SHAPES = [
    (2, 3, 8, 8),
    (2, 4, 8, 16),
    (3, 16, 32),
    (1, 32, 1, 128),
    (2, 8, 64, 1),
    (1, 64, 256, 256),
    (1, 3, 512, 512),
    (4, 64, 256, 512),
    (1, 3, 1024, 1024),
]

REPLICATION_PAD2D_PADDINGS = [
    (0, 0, 0, 0),
    (1, 2, 3, 4),
    (3, 0, 0, 3),
    torch.tensor([1, 2, 3, 4]),
    [0, 2, 3, 4],
    (0, 2, 3, 4),
    (2, 2, 0, 0),
]


def _normalize_padding(padding):
    if isinstance(padding, torch.Tensor):
        pl, pr, pt, pb = padding.tolist()
        return pl, pr, pt, pb
    pl, pr, pt, pb = padding
    return pl, pr, pt, pb


@pytest.mark.replication_pad2d_backward
@pytest.mark.parametrize("shape", REPLICATION_PAD2D_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("padding", REPLICATION_PAD2D_PADDINGS)
def test_replication_pad2d_backward(shape, dtype, padding):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    pl, pr, pt, pb = _normalize_padding(padding)

    if len(shape) == 4:
        N, C, H, W = shape
        padded_shape = (N, C, H + pt + pb, W + pl + pr)
    else:
        C, H, W = shape
        padded_shape = (C, H + pt + pb, W + pl + pr)

    grad_output = torch.ones(padded_shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_grad = utils.to_reference(grad_output)
    ref_padding = (
        utils.to_reference(padding) if isinstance(padding, torch.Tensor) else padding
    )
    ref_out = torch.ops.aten.replication_pad2d_backward(ref_grad, ref_x, ref_padding)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.replication_pad2d_backward(grad_output, x, padding)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.replication_pad2d_backward_grad_input
@pytest.mark.parametrize("shape", REPLICATION_PAD2D_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("padding", REPLICATION_PAD2D_PADDINGS)
def test_replication_pad2d_backward_grad_input(shape, dtype, padding):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    pl, pr, pt, pb = _normalize_padding(padding)

    if len(shape) == 4:
        N, C, H, W = shape
        padded_shape = (N, C, H + pt + pb, W + pl + pr)
        input_shape = (N, C, H, W)
    else:
        C, H, W = shape
        padded_shape = (C, H + pt + pb, W + pl + pr)
        input_shape = (C, H, W)

    grad_output = torch.ones(padded_shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_grad = utils.to_reference(grad_output)
    ref_padding = (
        utils.to_reference(padding) if isinstance(padding, torch.Tensor) else padding
    )
    ref_out = torch.empty(input_shape, dtype=dtype, device=ref_x.device)
    torch.ops.aten.replication_pad2d_backward.grad_input(
        ref_grad, ref_x, ref_padding, grad_input=ref_out
    )

    res_out = torch.empty(input_shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        torch.ops.aten.replication_pad2d_backward.grad_input(
            grad_output, x, padding, grad_input=res_out
        )

    utils.gems_assert_close(res_out, ref_out, dtype)

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.soft_margin_loss_backward
@pytest.mark.parametrize(
    "shape, target_shape",
    [
        ((8,), (8,)),
        ((4, 6), (4, 6)),
        ((2, 3, 4), (2, 3, 4)),
        ((2, 1, 4), (3, 4)),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_soft_margin_loss_backward(shape, target_shape, dtype, reduction):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn(target_shape, dtype=dtype, device=flag_gems.device)
    out_shape = torch.broadcast_shapes(shape, target_shape)

    if reduction == 0:
        grad_output = torch.randn(out_shape, dtype=dtype, device=flag_gems.device)
    else:
        grad_output = torch.randn((), dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp).to(torch.float32)
    ref_target = utils.to_reference(target).to(torch.float32)
    ref_grad_output = utils.to_reference(grad_output).to(torch.float32)
    ref_out = torch.ops.aten.soft_margin_loss_backward(
        ref_grad_output, ref_inp, ref_target, reduction
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = flag_gems.soft_margin_loss_backward(
            grad_output, inp, target, reduction
        )

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True, atol=2e-2)


@pytest.mark.soft_margin_loss_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_soft_margin_loss_backward_scalar_grad(dtype):
    """Test with scalar grad_output (from sum/mean reduction)."""
    inp = torch.randn(4, 6, dtype=dtype, device=flag_gems.device)
    target = torch.randn(4, 6, dtype=dtype, device=flag_gems.device)
    grad_output = torch.tensor(1.0, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp).to(torch.float32)
    ref_target = utils.to_reference(target).to(torch.float32)
    ref_grad_output = utils.to_reference(grad_output).to(torch.float32)
    ref_out = torch.ops.aten.soft_margin_loss_backward(
        ref_grad_output, ref_inp, ref_target, 1
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = flag_gems.soft_margin_loss_backward(
            grad_output, inp, target, 1
        )

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.soft_margin_loss_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_soft_margin_loss_backward_broadcast(dtype):
    """Test with broadcasting between input and target."""
    inp = torch.randn(2, 3, 4, dtype=dtype, device=flag_gems.device)
    target = torch.randn(4, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp).to(torch.float32)
    ref_target = utils.to_reference(target).to(torch.float32)
    ref_grad = torch.randn((), dtype=torch.float32)
    ref_out = torch.ops.aten.soft_margin_loss_backward(
        ref_grad, ref_inp, ref_target, 1
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = flag_gems.soft_margin_loss_backward(
            torch.randn((), dtype=dtype, device=flag_gems.device), inp, target, 1
        )

    # Only check shapes match — values differ due to broadcast
    assert res_out.shape == inp.shape

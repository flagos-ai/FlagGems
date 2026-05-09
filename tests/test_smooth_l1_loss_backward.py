import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.smooth_l1_loss_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
@pytest.mark.parametrize(
    "shape",
    [
        (8,),
        (64,),
        (32, 32),
        (128, 256),
        (4, 16, 32),
        (64, 64, 64),
        (2, 3, 4, 5),
        (8, 16, 32, 64),
        (2, 3, 4, 5, 6),
        (4, 8, 16, 32, 64),
    ],
)
def test_smooth_l1_loss_backward(shape, dtype, reduction, beta):
    """Test backward via autograd: compare gem gradients with reference."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device).requires_grad_()
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True).clone().detach().requires_grad_()
    ref_target = utils.to_reference(target, True).clone().detach()

    ref_out = torch.ops.aten.smooth_l1_loss(ref_inp, ref_target, reduction, beta)
    ref_loss = ref_out.sum() if reduction == 0 else ref_out
    ref_loss.backward()
    ref_grad = ref_inp.grad.to(dtype)

    with flag_gems.use_gems():
        gem_out = torch.ops.aten.smooth_l1_loss(inp, target, reduction, beta)
    gem_loss = gem_out.sum() if reduction == 0 else gem_out
    gem_loss.backward()
    gem_grad = inp.grad

    utils.gems_assert_close(gem_grad, ref_grad, dtype, equal_nan=True)

# ===========================================================================
# smooth_l1_loss_backward.grad_input — out variant
# ===========================================================================
@pytest.mark.smooth_l1_loss_backward_out
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("beta", [1.0])
@pytest.mark.parametrize("shape", [(64,), (32, 32), (128, 256)])
def test_smooth_l1_loss_backward_out(shape, dtype, reduction, beta):
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_go = utils.to_reference(grad_output, True)
    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target, True)

    ref_grad_input = torch.empty(shape, dtype=dtype, device=ref_inp.device)
    torch.ops.aten.smooth_l1_loss_backward.grad_input(
        ref_go, ref_inp, ref_target, reduction, beta, grad_input=ref_grad_input
    )

    grad_input = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        torch.ops.aten.smooth_l1_loss_backward.grad_input(
            grad_output, inp, target, reduction, beta, grad_input=grad_input
        )
    utils.gems_assert_close(grad_input, ref_grad_input, dtype, equal_nan=True)

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

FLOAT_DTYPES = utils.FLOAT_DTYPES


def _run_backward_out_test(shape, dtype, reduction, beta):
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_go = utils.to_reference(grad_output, True)
    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target, True)
    ref_grad_input = torch.empty(shape, dtype=dtype, device=ref_inp.device)
    ref_grad_input = ref_grad_input.resize_(0)
    torch.ops.aten.smooth_l1_loss_backward.grad_input(
        ref_go, ref_inp, ref_target, reduction, beta, grad_input=ref_grad_input
    )
    grad_input = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        torch.ops.aten.smooth_l1_loss_backward.grad_input(
            grad_output, inp, target, reduction, beta, grad_input=grad_input
        )
    utils.gems_assert_close(grad_input, ref_grad_input, dtype, equal_nan=True)


# ===========================================================================
# smooth_l1_loss_backward
# ===========================================================================
@pytest.mark.smooth_l1_loss_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("shape", [(64,), (32, 32), (128, 256)])
def test_accuracy_smooth_l1_loss_backward(shape, dtype, reduction, beta):
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


@pytest.mark.smooth_l1_loss_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [1, 2])
@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("shape", [(0,), (3, 0)])
def test_accuracy_smooth_l1_loss_backward_empty(shape, dtype, reduction, beta):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device).requires_grad_()
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True).clone().detach().requires_grad_()
    ref_target = utils.to_reference(target, True).clone().detach()
    ref_out = torch.ops.aten.smooth_l1_loss(ref_inp, ref_target, reduction, beta)
    ref_loss = ref_out.sum() if reduction == 0 else ref_out
    ref_loss.backward()
    ref_grad = ref_inp.grad
    if ref_grad is not None:
        ref_grad = ref_grad.to(dtype)
    with flag_gems.use_gems():
        gem_out = torch.ops.aten.smooth_l1_loss(inp, target, reduction, beta)
    gem_loss = gem_out.sum() if reduction == 0 else gem_out
    gem_loss.backward()
    gem_grad = inp.grad
    if gem_grad is not None and ref_grad is not None:
        utils.gems_assert_close(gem_grad, ref_grad, dtype, equal_nan=True)


@pytest.mark.smooth_l1_loss_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [1, 2])
@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("shape", [(64, 64)])
def test_accuracy_smooth_l1_loss_backward_identical(shape, dtype, reduction, beta):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device).requires_grad_()
    target = inp.detach().clone()
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


@pytest.mark.smooth_l1_loss_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [1, 2])
@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("shape", [(64, 64)])
def test_accuracy_smooth_l1_loss_backward_large_diff(shape, dtype, reduction, beta):
    inp = torch.full(shape, 100.0, dtype=dtype, device=flag_gems.device).requires_grad_()
    target = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
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


@pytest.mark.smooth_l1_loss_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [1, 2])
@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_accuracy_smooth_l1_loss_backward_small_diff(dtype, reduction, beta):
    shape = (64, 64)
    inp = torch.full(shape, 0.01, dtype=dtype, device=flag_gems.device).requires_grad_()
    target = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
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
# smooth_l1_loss_backward_out
# ===========================================================================
@pytest.mark.smooth_l1_loss_backward_out
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("shape", [(64,), (32, 32), (128, 256)])
def test_accuracy_smooth_l1_loss_backward_out(shape, dtype, reduction, beta):
    _run_backward_out_test(shape, dtype, reduction, beta)


@pytest.mark.smooth_l1_loss_backward_out
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [1, 2])
@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("shape", [(0,), (3, 0)])
def test_accuracy_smooth_l1_loss_backward_out_empty(shape, dtype, reduction, beta):
    _run_backward_out_test(shape, dtype, reduction, beta)


@pytest.mark.smooth_l1_loss_backward_out
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [1, 2])
@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("shape", [(64, 64)])
def test_accuracy_smooth_l1_loss_backward_out_identical(shape, dtype, reduction, beta):
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = inp.detach().clone()
    ref_go = utils.to_reference(grad_output, True)
    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target, True)
    ref_grad_input = torch.empty(shape, dtype=dtype, device=ref_inp.device)
    ref_grad_input = ref_grad_input.resize_(0)
    torch.ops.aten.smooth_l1_loss_backward.grad_input(
        ref_go, ref_inp, ref_target, reduction, beta, grad_input=ref_grad_input
    )
    grad_input = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        torch.ops.aten.smooth_l1_loss_backward.grad_input(
            grad_output, inp, target, reduction, beta, grad_input=grad_input
        )
    utils.gems_assert_close(grad_input, ref_grad_input, dtype, equal_nan=True)


@pytest.mark.smooth_l1_loss_backward_out
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [1, 2])
@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("shape", [(64, 64)])
def test_accuracy_smooth_l1_loss_backward_out_large_diff(
    shape, dtype, reduction, beta
):
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = torch.full(shape, 100.0, dtype=dtype, device=flag_gems.device)
    target = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    ref_go = utils.to_reference(grad_output, True)
    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target, True)
    ref_grad_input = torch.empty(shape, dtype=dtype, device=ref_inp.device)
    ref_grad_input = ref_grad_input.resize_(0)
    torch.ops.aten.smooth_l1_loss_backward.grad_input(
        ref_go, ref_inp, ref_target, reduction, beta, grad_input=ref_grad_input
    )
    grad_input = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        torch.ops.aten.smooth_l1_loss_backward.grad_input(
            grad_output, inp, target, reduction, beta, grad_input=grad_input
        )
    utils.gems_assert_close(grad_input, ref_grad_input, dtype, equal_nan=True)


@pytest.mark.smooth_l1_loss_backward_out
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [1, 2])
@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_accuracy_smooth_l1_loss_backward_out_small_diff(dtype, reduction, beta):
    shape = (64, 64)
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = torch.full(shape, 0.01, dtype=dtype, device=flag_gems.device)
    target = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    ref_go = utils.to_reference(grad_output, True)
    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target, True)
    ref_grad_input = torch.empty(shape, dtype=dtype, device=ref_inp.device)
    ref_grad_input = ref_grad_input.resize_(0)
    torch.ops.aten.smooth_l1_loss_backward.grad_input(
        ref_go, ref_inp, ref_target, reduction, beta, grad_input=ref_grad_input
    )
    grad_input = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        torch.ops.aten.smooth_l1_loss_backward.grad_input(
            grad_output, inp, target, reduction, beta, grad_input=grad_input
        )
    utils.gems_assert_close(grad_input, ref_grad_input, dtype, equal_nan=True)

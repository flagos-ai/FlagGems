"""Accuracy tests for ``aten::huber_loss_backward``.

Verified against PyTorch native by routing the gradient through
``torch.ops.aten.huber_loss_backward`` directly so we exercise the exact
schema FlagGems registers.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg


@pytest.mark.huber_loss_backward
@pytest.mark.parametrize(
    "shape,target_shape",
    [
        ((0,), (0,)),
        ((1,), (1,)),
        ((2, 3), (2, 3)),
        ((32, 17), (32, 17)),
        ((4, 8, 16), (4, 8, 16)),
        ((2, 3, 16, 16), (2, 3, 16, 16)),
        ((2, 3, 4), (4,)),
        ((2, 1, 4), (3, 4)),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("delta", [0.25, 0.5, 1.0, 2.0])
def test_huber_loss_backward(shape, target_shape, dtype, reduction, delta):
    """Main matrix: shape x dtype x reduction x delta with both reduced and
    non-reduced grad_output shapes."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn(target_shape, dtype=dtype, device=flag_gems.device)
    out_shape = torch.broadcast_shapes(shape, target_shape)

    # Exercise the non-contiguous code path occasionally to ensure
    # `.contiguous()` is applied as expected.
    if len(shape) >= 2 and shape[0] > 0:
        inp = inp.transpose(0, 1).contiguous().transpose(0, 1)

    if reduction == 0:
        grad_output = torch.randn(out_shape, dtype=dtype, device=flag_gems.device)
        if len(out_shape) >= 2 and out_shape[0] > 0:
            grad_output = grad_output.transpose(0, 1).contiguous().transpose(0, 1)
    else:
        grad_output = torch.randn((), dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp).to(torch.float32)
    ref_target = utils.to_reference(target).to(torch.float32)
    ref_grad_output = utils.to_reference(grad_output).to(torch.float32)
    ref_out = torch.ops.aten.huber_loss_backward(
        ref_grad_output, ref_inp, ref_target, reduction, delta
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.huber_loss_backward(
            grad_output, inp, target, reduction, delta
        )

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True, atol=2e-2)


@pytest.mark.huber_loss_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_huber_loss_backward_quadratic_region(dtype):
    """When |diff| <= delta, the gradient equals grad_output * (input - target),
    i.e. matches the mse_loss gradient.  Exercise that branch explicitly."""
    inp = torch.tensor([0.0, 0.1, -0.2, 0.3], dtype=dtype, device=flag_gems.device)
    target = torch.zeros_like(inp)
    grad_output = torch.tensor(2.0, dtype=dtype, device=flag_gems.device)
    delta = 1.0

    ref_inp = utils.to_reference(inp).to(torch.float32)
    ref_target = utils.to_reference(target).to(torch.float32)
    ref_grad_output = utils.to_reference(grad_output).to(torch.float32)
    ref_out = torch.ops.aten.huber_loss_backward(
        ref_grad_output, ref_inp, ref_target, 2, delta
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.huber_loss_backward(grad_output, inp, target, 2, delta)
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True, atol=2e-2)


@pytest.mark.huber_loss_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_huber_loss_backward_linear_region(dtype):
    """When |diff| > delta, the gradient saturates to +/- delta scaled by
    grad_output.  Exercise that branch explicitly."""
    inp = torch.tensor([5.0, -3.0, 7.0, -10.0], dtype=dtype, device=flag_gems.device)
    target = torch.zeros_like(inp)
    grad_output = torch.tensor(1.0, dtype=dtype, device=flag_gems.device)
    delta = 1.0

    ref_inp = utils.to_reference(inp).to(torch.float32)
    ref_target = utils.to_reference(target).to(torch.float32)
    ref_grad_output = utils.to_reference(grad_output).to(torch.float32)
    ref_out = torch.ops.aten.huber_loss_backward(
        ref_grad_output, ref_inp, ref_target, 2, delta
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.huber_loss_backward(grad_output, inp, target, 2, delta)
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True, atol=2e-2)


@pytest.mark.huber_loss_backward
def test_huber_loss_backward_scalar_grad_output():
    """A 0-D grad_output (the natural output of autograd on a reduced loss)
    should broadcast across the input."""
    inp = torch.tensor([-1.0, -0.5, 1.0], device=flag_gems.device)
    target = torch.zeros_like(inp)
    grad_output = torch.tensor(2.0, device=flag_gems.device)
    delta = 0.75

    ref_inp = utils.to_reference(inp).to(torch.float32)
    ref_target = utils.to_reference(target).to(torch.float32)
    ref_grad_output = utils.to_reference(grad_output).to(torch.float32)

    ref_out = torch.ops.aten.huber_loss_backward(
        ref_grad_output, ref_inp, ref_target, 0, delta
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten.huber_loss_backward(grad_output, inp, target, 0, delta)

    utils.gems_assert_close(res_out, ref_out, torch.float32)


@pytest.mark.huber_loss_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_huber_loss_backward_broadcast(dtype):
    """Mismatched but broadcastable input / target shapes."""
    inp = torch.randn((2, 3, 4), dtype=dtype, device=flag_gems.device)
    target = torch.randn((4,), dtype=dtype, device=flag_gems.device)
    grad_output = torch.randn((), dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp).to(torch.float32)
    ref_target = utils.to_reference(target).to(torch.float32)
    ref_grad_output = utils.to_reference(grad_output).to(torch.float32)
    ref_out = torch.ops.aten.huber_loss_backward(
        ref_grad_output, ref_inp, ref_target, 1, 1.0
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.huber_loss_backward(grad_output, inp, target, 1, 1.0)
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True, atol=2e-2)


@pytest.mark.huber_loss_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_huber_loss_backward_special_values(dtype):
    """Verify behaviour on +/- 0, +/- inf, and NaN inputs."""
    inp = torch.tensor(
        [0.0, -0.0, 1.0, -2.0, float("inf"), float("-inf"), float("nan")],
        dtype=dtype,
        device=flag_gems.device,
    )
    target = torch.tensor(
        [0.0, 1.0, -1.0, -2.0, 1.0, float("-inf"), 0.0],
        dtype=dtype,
        device=flag_gems.device,
    )
    grad_output = torch.tensor(1.0, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp).to(torch.float32)
    ref_target = utils.to_reference(target).to(torch.float32)
    ref_grad_output = utils.to_reference(grad_output).to(torch.float32)
    ref_out = torch.ops.aten.huber_loss_backward(
        ref_grad_output, ref_inp, ref_target, 0, 1.0
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.huber_loss_backward(grad_output, inp, target, 0, 1.0)
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True, atol=2e-2)


@pytest.mark.huber_loss_backward
def test_huber_loss_backward_out():
    """``huber_loss_backward.out`` writes into the supplied tensor."""
    inp = torch.randn((8, 16), dtype=torch.float32, device=flag_gems.device)
    target = torch.randn((8, 16), dtype=torch.float32, device=flag_gems.device)
    grad_output = torch.randn((), dtype=torch.float32, device=flag_gems.device)
    out = torch.empty_like(inp)

    ref_inp = utils.to_reference(inp).to(torch.float32)
    ref_target = utils.to_reference(target).to(torch.float32)
    ref_grad_output = utils.to_reference(grad_output).to(torch.float32)
    ref_out = torch.empty_like(ref_inp)

    torch.ops.aten.huber_loss_backward.out(
        ref_grad_output, ref_inp, ref_target, 1, 0.5, grad_input=ref_out
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten.huber_loss_backward.out(
            grad_output, inp, target, 1, 0.5, grad_input=out
        )

    assert res_out is out
    utils.gems_assert_close(out, ref_out, torch.float32)


@pytest.mark.huber_loss_backward
def test_huber_loss_backward_negative_delta():
    """Negative `delta` should raise -- it's mathematically nonsensical."""
    grad_output = torch.randn((), dtype=torch.float32, device=flag_gems.device)
    inp = torch.randn((8,), dtype=torch.float32, device=flag_gems.device)
    target = torch.randn((8,), dtype=torch.float32, device=flag_gems.device)

    with flag_gems.use_gems(), pytest.raises(RuntimeError, match="negative"):
        torch.ops.aten.huber_loss_backward(grad_output, inp, target, 1, -1.0)


@pytest.mark.huber_loss_backward
def test_huber_loss_backward_empty():
    """Empty tensors are a no-op but must not crash."""
    inp = torch.empty((0,), dtype=torch.float32, device=flag_gems.device)
    target = torch.empty((0,), dtype=torch.float32, device=flag_gems.device)
    grad_output = torch.randn((), dtype=torch.float32, device=flag_gems.device)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.huber_loss_backward(grad_output, inp, target, 1, 1.0)
    assert res_out.shape == inp.shape


@pytest.mark.huber_loss_backward
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_huber_loss_backward_autograd_round_trip(reduction):
    """A round-trip via autograd of huber_loss forward should hit our backward
    kernel and agree numerically with PyTorch's native gradient."""
    if cfg.TO_CPU:
        pytest.skip("Round-trip requires CUDA dispatch.")

    inp = torch.randn(
        (4, 7), dtype=torch.float32, device=flag_gems.device, requires_grad=True
    )
    target = torch.randn((4, 7), dtype=torch.float32, device=flag_gems.device)
    delta = 0.5

    # PyTorch reference gradient.
    ref_inp = inp.detach().clone().requires_grad_(True)
    ref_loss = torch.nn.functional.huber_loss(
        ref_inp,
        target,
        reduction={0: "none", 1: "mean", 2: "sum"}[reduction],
        delta=delta,
    )
    if ref_loss.dim() == 0:
        ref_loss.backward()
    else:
        ref_loss.sum().backward()
    ref_grad = ref_inp.grad

    # Same path through flag_gems' dispatch.
    with flag_gems.use_gems():
        gems_loss = torch.nn.functional.huber_loss(
            inp,
            target,
            reduction={0: "none", 1: "mean", 2: "sum"}[reduction],
            delta=delta,
        )
        if gems_loss.dim() == 0:
            gems_loss.backward()
        else:
            gems_loss.sum().backward()
    gems_grad = inp.grad

    utils.gems_assert_close(gems_grad, ref_grad, torch.float32, atol=2e-2)

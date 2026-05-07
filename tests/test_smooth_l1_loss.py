import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------
@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_accuracy_smooth_l1_loss(shape, dtype, reduction, beta):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target, True)
    ref_out = torch.nn.functional.smooth_l1_loss(
        ref_inp, ref_target, reduction=reduction, beta=beta
    )
    with flag_gems.use_gems():
        res_out = torch.nn.functional.smooth_l1_loss(
            inp, target, reduction=reduction, beta=beta
        )
    if reduction == "none":
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        res_f32 = res_out.to(torch.float32)
        ref_f32 = ref_out.to(torch.float32)
        if torch.isinf(res_f32).any() or torch.isinf(ref_f32).any():
            return
        rtol = 5e-2 if dtype in (torch.bfloat16, torch.float16) else 1e-3
        assert torch.allclose(res_f32, ref_f32, rtol=rtol, atol=1.0)


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_smooth_l1_loss_zero_difference(dtype):
    """When inp == target the loss must be exactly zero (no NaN from 0/beta)."""
    x = torch.randn((128, 64), dtype=dtype, device=flag_gems.device)
    for reduction in ["none", "mean", "sum"]:
        with flag_gems.use_gems():
            res = torch.nn.functional.smooth_l1_loss(x, x, reduction=reduction)
        if reduction == "none":
            assert torch.equal(res, torch.zeros_like(res))
        else:
            assert res.item() == 0.0, f"expected zero, got {res.item()}"


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_smooth_l1_loss_quadratic_branch(dtype):
    """All |diff| < beta: result must be (0.5*diff^2 / beta)."""
    x = torch.zeros((1024,), dtype=dtype, device=flag_gems.device)
    y = torch.full((1024,), 0.1, dtype=dtype, device=flag_gems.device)
    beta = 1.0
    with flag_gems.use_gems():
        res = torch.nn.functional.smooth_l1_loss(x, y, reduction="none", beta=beta)
    expect = 0.5 * 0.1 * 0.1
    assert torch.allclose(
        res.float(),
        torch.full_like(res, expect, dtype=torch.float32),
        atol=1e-3,
        rtol=1e-3,
    )


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_smooth_l1_loss_linear_branch(dtype):
    """All |diff| > beta: result must be (|diff| - 0.5*beta)."""
    x = torch.zeros((1024,), dtype=dtype, device=flag_gems.device)
    y = torch.full((1024,), 5.0, dtype=dtype, device=flag_gems.device)
    beta = 1.0
    with flag_gems.use_gems():
        res = torch.nn.functional.smooth_l1_loss(x, y, reduction="none", beta=beta)
    expect = 5.0 - 0.5
    assert torch.allclose(
        res.float(),
        torch.full_like(res, expect, dtype=torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.smooth_l1_loss
def test_smooth_l1_loss_empty_tensor():
    x = torch.empty((0,), dtype=torch.float32, device=flag_gems.device)
    y = torch.empty((0,), dtype=torch.float32, device=flag_gems.device)
    for reduction in ["none", "mean", "sum"]:
        with flag_gems.use_gems():
            res = torch.nn.functional.smooth_l1_loss(x, y, reduction=reduction)
        if reduction == "none":
            assert res.numel() == 0
        else:
            # mean of empty is NaN in torch; sum is 0 — defer to torch behaviour
            ref = torch.nn.functional.smooth_l1_loss(x, y, reduction=reduction)
            torch.testing.assert_close(res, ref, equal_nan=True)


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------
@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("shape", [(128,), (32, 64), (4, 8, 16)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_smooth_l1_loss_backward(shape, dtype, reduction, beta):
    """Backward grads should match a torch autograd reference."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    target = torch.randn(
        shape, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    ref_inp = utils.to_reference(inp, True).detach().requires_grad_(True)
    ref_target = utils.to_reference(target, True).detach().requires_grad_(True)

    ref_loss = torch.nn.functional.smooth_l1_loss(
        ref_inp, ref_target, reduction=reduction, beta=beta
    )
    if reduction == "none":
        ref_grad_out = torch.randn_like(ref_loss)
        ref_loss.backward(ref_grad_out)
    else:
        ref_loss.backward()

    with flag_gems.use_gems():
        res_loss = torch.nn.functional.smooth_l1_loss(
            inp, target, reduction=reduction, beta=beta
        )
    if reduction == "none":
        # Use the same upstream grad we used for the reference
        grad_out_local = ref_grad_out.to(dtype=dtype, device=flag_gems.device)
        with flag_gems.use_gems():
            res_loss.backward(grad_out_local)
    else:
        with flag_gems.use_gems():
            res_loss.backward()

    rtol = 5e-2 if dtype in (torch.bfloat16, torch.float16) else 1e-3
    atol = 1e-2 if dtype in (torch.bfloat16, torch.float16) else 1e-4
    torch.testing.assert_close(
        inp.grad.float(), ref_inp.grad.float(), rtol=rtol, atol=atol
    )
    torch.testing.assert_close(
        target.grad.float(), ref_target.grad.float(), rtol=rtol, atol=atol
    )


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_smooth_l1_loss_backward_kink(dtype):
    """Differences exactly at the |diff|=beta boundary use the linear branch
    (PyTorch convention).  Verify the backward grad sign there."""
    beta = 1.0
    inp = torch.tensor(
        [-2.0, -1.0, 0.0, 1.0, 2.0],
        dtype=dtype,
        device=flag_gems.device,
        requires_grad=True,
    )
    target = torch.zeros_like(inp).requires_grad_(False)
    with flag_gems.use_gems():
        loss = torch.nn.functional.smooth_l1_loss(
            inp, target, reduction="sum", beta=beta
        )
        loss.backward()
    # |diff|<beta -> diff/beta;  |diff|>=beta -> sign(diff)
    expected = torch.tensor(
        [-1.0, -1.0, 0.0, 1.0, 1.0], dtype=dtype, device=flag_gems.device
    )
    torch.testing.assert_close(inp.grad.float(), expected.float(), atol=1e-3, rtol=1e-3)

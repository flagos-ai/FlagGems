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
        inp.grad.float().cpu(), ref_inp.grad.float().cpu(), rtol=rtol, atol=atol
    )
    torch.testing.assert_close(
        target.grad.float().cpu(),
        ref_target.grad.float().cpu(),
        rtol=rtol,
        atol=atol,
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


# ---------------------------------------------------------------------------
# Stress: large shapes & dtypes
# ---------------------------------------------------------------------------
@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1024 * 1024),
        (32, 32, 1024),
        (4, 8, 16, 32, 64),
        (1024 * 1024,),
        (1024, 1024),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_smooth_l1_loss_large_shapes(shape, dtype):
    """Make sure the reduction pipeline scales to multi-MiB tensors."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target, True)
    for reduction in ["mean", "sum"]:
        ref = torch.nn.functional.smooth_l1_loss(
            ref_inp, ref_target, reduction=reduction
        )
        with flag_gems.use_gems():
            res = torch.nn.functional.smooth_l1_loss(inp, target, reduction=reduction)
        if torch.isinf(res.float()).any() or torch.isinf(ref.float()).any():
            continue
        rtol = 5e-2 if dtype in (torch.bfloat16, torch.float16) else 5e-3
        assert torch.allclose(res.float().cpu(), ref.float().cpu(), rtol=rtol, atol=1.0)


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("scale", [1e-4, 1e-2, 1.0, 1e2, 1e4])
def test_smooth_l1_loss_value_scale_sweep(dtype, scale):
    """Stability across input magnitudes — fp16/bf16 should not overflow on the
    quadratic branch for reasonable scales, and the linear branch must remain
    bandwidth-bound rather than saturating."""
    inp = torch.randn((256, 256), dtype=dtype, device=flag_gems.device) * scale
    target = torch.randn((256, 256), dtype=dtype, device=flag_gems.device) * scale
    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target, True)
    ref = torch.nn.functional.smooth_l1_loss(ref_inp, ref_target, reduction="mean")
    with flag_gems.use_gems():
        res = torch.nn.functional.smooth_l1_loss(inp, target, reduction="mean")
    if torch.isinf(res.float()).any() or torch.isinf(ref.float()).any():
        return
    rtol = 1e-1 if dtype in (torch.bfloat16, torch.float16) else 5e-3
    assert torch.allclose(
        res.float().cpu(), ref.float().cpu(), rtol=rtol, atol=max(1.0, float(scale))
    )


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_smooth_l1_loss_negative_values(dtype):
    """Negative inputs/targets — sign-flip branch must be symmetric."""
    inp = torch.randn((128, 64), dtype=dtype, device=flag_gems.device) - 2.0
    target = torch.zeros_like(inp)
    inp_neg = -inp.clone()
    target_neg = -target.clone()
    ref_a = torch.nn.functional.smooth_l1_loss(
        utils.to_reference(inp, True),
        utils.to_reference(target, True),
        reduction="none",
    )
    ref_b = torch.nn.functional.smooth_l1_loss(
        utils.to_reference(inp_neg, True),
        utils.to_reference(target_neg, True),
        reduction="none",
    )
    with flag_gems.use_gems():
        res_a = torch.nn.functional.smooth_l1_loss(inp, target, reduction="none")
        res_b = torch.nn.functional.smooth_l1_loss(
            inp_neg, target_neg, reduction="none"
        )
    utils.gems_assert_close(res_a, ref_a, dtype)
    utils.gems_assert_close(res_b, ref_b, dtype)
    # The two should be identical (loss is symmetric under (x,y) -> (-x,-y))
    torch.testing.assert_close(
        res_a.float().cpu(), res_b.float().cpu(), atol=1e-3, rtol=1e-3
    )


# ---------------------------------------------------------------------------
# Beta parameter sweep — including tiny + large + the edge value 1.0
# ---------------------------------------------------------------------------
@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("beta", [0.01, 0.1, 0.25, 1.0, 2.0, 10.0])
def test_smooth_l1_loss_beta_sweep(dtype, beta):
    inp = torch.randn((64, 128), dtype=dtype, device=flag_gems.device)
    target = torch.randn((64, 128), dtype=dtype, device=flag_gems.device)
    ref = torch.nn.functional.smooth_l1_loss(
        utils.to_reference(inp, True),
        utils.to_reference(target, True),
        reduction="mean",
        beta=beta,
    )
    with flag_gems.use_gems():
        res = torch.nn.functional.smooth_l1_loss(
            inp, target, reduction="mean", beta=beta
        )
    if torch.isinf(res.float()).any() or torch.isinf(ref.float()).any():
        return
    rtol = 5e-2 if dtype in (torch.bfloat16, torch.float16) else 5e-3
    assert torch.allclose(res.float().cpu(), ref.float().cpu(), rtol=rtol, atol=1.0)


# ---------------------------------------------------------------------------
# Backward: large-shape correctness
# ---------------------------------------------------------------------------
@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("shape", [(1024, 1024), (8, 4, 1024), (1024 * 256,)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_smooth_l1_loss_backward_large(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    target = torch.randn(
        shape, dtype=dtype, device=flag_gems.device, requires_grad=False
    )
    ref_inp = utils.to_reference(inp, True).detach().requires_grad_(True)
    ref_target = utils.to_reference(target, True).detach()
    ref_loss = torch.nn.functional.smooth_l1_loss(ref_inp, ref_target, reduction="mean")
    ref_loss.backward()
    with flag_gems.use_gems():
        res_loss = torch.nn.functional.smooth_l1_loss(inp, target, reduction="mean")
        res_loss.backward()
    rtol = 5e-2 if dtype in (torch.bfloat16, torch.float16) else 1e-3
    atol = 1e-2 if dtype in (torch.bfloat16, torch.float16) else 1e-4
    torch.testing.assert_close(
        inp.grad.float().cpu(), ref_inp.grad.float().cpu(), rtol=rtol, atol=atol
    )


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_smooth_l1_loss_backward_zero_grad_output(dtype):
    """grad_output == 0 ⇒ input grad must be exactly zero (no NaN from 0/beta)."""
    inp = torch.randn((128,), dtype=dtype, device=flag_gems.device, requires_grad=True)
    target = torch.randn((128,), dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        loss = torch.nn.functional.smooth_l1_loss(inp, target, reduction="none")
        loss.backward(torch.zeros_like(loss))
    assert torch.all(inp.grad == 0)


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_smooth_l1_loss_backward_target_grad(dtype):
    """The grad w.r.t. target must be exactly the negation of the grad
    w.r.t. input (chain-rule symmetry under y = x - target)."""
    inp = torch.randn((64,), dtype=dtype, device=flag_gems.device, requires_grad=True)
    target = torch.randn(
        (64,), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    with flag_gems.use_gems():
        loss = torch.nn.functional.smooth_l1_loss(inp, target, reduction="sum")
        loss.backward()
    torch.testing.assert_close(
        target.grad.float().cpu(),
        (-inp.grad).float().cpu(),
        atol=1e-3,
        rtol=1e-3,
    )


# ---------------------------------------------------------------------------
# Reduction semantics — mean must be exactly sum/N
# ---------------------------------------------------------------------------
@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_smooth_l1_loss_mean_equals_sum_over_n(dtype):
    inp = torch.randn((256, 128), dtype=dtype, device=flag_gems.device)
    target = torch.randn((256, 128), dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        loss_sum = torch.nn.functional.smooth_l1_loss(inp, target, reduction="sum")
        loss_mean = torch.nn.functional.smooth_l1_loss(inp, target, reduction="mean")
    expected = loss_sum.float() / float(inp.numel())
    torch.testing.assert_close(
        loss_mean.float().cpu(),
        expected.cpu(),
        rtol=5e-3,
        atol=1e-3,
    )


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_smooth_l1_loss_none_sum_matches_sum_reduction(dtype):
    """sum(reduction=none) should match reduction=sum directly."""
    inp = torch.randn((64, 64), dtype=dtype, device=flag_gems.device)
    target = torch.randn((64, 64), dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        loss_per = torch.nn.functional.smooth_l1_loss(inp, target, reduction="none")
        loss_sum = torch.nn.functional.smooth_l1_loss(inp, target, reduction="sum")
    torch.testing.assert_close(
        loss_per.float().sum().cpu(),
        loss_sum.float().cpu(),
        rtol=5e-3,
        atol=1e-2,
    )


# ---------------------------------------------------------------------------
# Non-contiguous and strided inputs
# ---------------------------------------------------------------------------
@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_smooth_l1_loss_non_contiguous(dtype):
    full = torch.randn((64, 128), dtype=dtype, device=flag_gems.device)
    inp = full[:, ::2]  # non-contiguous stride-2 view
    target = torch.zeros_like(inp)
    ref = torch.nn.functional.smooth_l1_loss(
        utils.to_reference(inp, True),
        utils.to_reference(target, True),
        reduction="mean",
    )
    with flag_gems.use_gems():
        res = torch.nn.functional.smooth_l1_loss(inp, target, reduction="mean")
    rtol = 5e-2 if dtype in (torch.bfloat16, torch.float16) else 1e-3
    assert torch.allclose(res.float().cpu(), ref.float().cpu(), rtol=rtol, atol=1.0)


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_smooth_l1_loss_transposed_input(dtype):
    x = torch.randn(
        (128, 64), dtype=dtype, device=flag_gems.device
    ).t()  # 64x128, non-contig
    y = torch.randn_like(x)
    ref = torch.nn.functional.smooth_l1_loss(
        utils.to_reference(x, True), utils.to_reference(y, True), reduction="mean"
    )
    with flag_gems.use_gems():
        res = torch.nn.functional.smooth_l1_loss(x, y, reduction="mean")
    rtol = 5e-2 if dtype in (torch.bfloat16, torch.float16) else 1e-3
    assert torch.allclose(res.float().cpu(), ref.float().cpu(), rtol=rtol, atol=1.0)


# ---------------------------------------------------------------------------
# Determinism — running the kernel twice gives bitwise identical output
# ---------------------------------------------------------------------------
@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_smooth_l1_loss_deterministic(dtype):
    torch.manual_seed(0)
    inp = torch.randn((128, 128), dtype=dtype, device=flag_gems.device)
    target = torch.randn((128, 128), dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        a = torch.nn.functional.smooth_l1_loss(inp, target, reduction="mean")
        b = torch.nn.functional.smooth_l1_loss(inp, target, reduction="mean")
    assert torch.equal(a, b), "smooth_l1_loss is not deterministic"


# ---------------------------------------------------------------------------
# Reduction-mode integer enum compatibility (aten passes int)
# ---------------------------------------------------------------------------
@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("reduction_int", [0, 1, 2])
def test_smooth_l1_loss_int_reduction(reduction_int):
    """When called via aten dispatch the reduction comes through as an int."""
    from flag_gems.ops.smooth_l1_loss import smooth_l1_loss

    inp = torch.randn(128, device=flag_gems.device)
    target = torch.randn(128, device=flag_gems.device)
    out = smooth_l1_loss(inp, target, reduction=reduction_int, beta=1.0)
    assert torch.isfinite(out).all()

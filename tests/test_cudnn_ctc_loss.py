import pytest
import torch
import torch.nn.functional as F

import flag_gems

from . import accuracy_utils as utils

# `cudnn_ctc_loss` only accepts float32 `log_probs` (cuDNN restriction).
CTC_DTYPES = [torch.float32]

# (T, N, C, S) tuples: time steps, batch size, number of classes, max target
# length. `cudnn_ctc_loss` only accepts float32 `log_probs` and int32 targets.
CTC_SHAPES = [
    (1, 1, 2, 1),
    (4, 1, 8, 2),
    (16, 4, 16, 5),
    (32, 8, 32, 8),
    (64, 16, 64, 16),
    (128, 32, 128, 32),
]


def _make_inputs(shape, device, input_lengths=None, target_lengths=None):
    """Build log_probs/targets for a CTC problem.

    The returned targets are a single 1D int32 tensor, which is the layout
    `cudnn_ctc_loss` expects (targets are kept on the CPU since cuDNN requires
    the CPU backend for them).
    """
    t_steps, batch, classes, max_target = shape
    raw = torch.randn(t_steps, batch, classes, dtype=torch.float32, device=device)
    log_probs = torch.log_softmax(raw, dim=-1)
    if input_lengths is None:
        input_lengths = [t_steps] * batch
    if target_lengths is None:
        target_lengths = [max_target] * batch
    pieces = [
        torch.randint(1, classes, (length,), device=device, dtype=torch.int32)
        for length in target_lengths
    ]
    targets = (
        torch.cat(pieces).cpu()
        if pieces
        else torch.empty(0, dtype=torch.int32, device="cpu")
    )
    return log_probs, targets, input_lengths, target_lengths


def _reference(log_probs, targets, input_lengths, target_lengths):
    """Reference result using `F.ctc_loss`, comparing only the valid region.

    cuDNN leaves the gradient of the padded time steps uninitialized, so the
    rows `t >= input_lengths[b]` are not part of the reference.
    """
    losses = F.ctc_loss(
        log_probs.detach(),
        targets.to(log_probs.device),
        input_lengths,
        target_lengths,
        reduction="none",
        zero_infinity=False,
    )
    lp = log_probs.detach().clone().requires_grad_(True)
    ref = F.ctc_loss(
        lp,
        targets.to(log_probs.device),
        input_lengths,
        target_lengths,
        reduction="none",
        zero_infinity=False,
    )
    (grads,) = torch.autograd.grad(ref.sum(), lp, allow_unused=True)
    return losses, grads, lp


def _assert_valid_region_close(res_grad, ref_grad, input_lengths):
    """Compare only the rows the cuDNN alignment actually visits.

    The alignment accumulates `log_alpha` over `input_lengths[b]` steps, so the
    absolute error grows with the sequence length. Scale the tolerance the same
    way the `ctc_loss` tests do.
    """
    batch = ref_grad.shape[1]
    reduce_dim = max(1, ref_grad.shape[0] * (max(input_lengths) + 1))
    for b in range(batch):
        valid = input_lengths[b]
        utils.gems_assert_close(
            res_grad[:valid, b],
            ref_grad[:valid, b],
            torch.float32,
            reduce_dim=reduce_dim,
        )


@pytest.mark.cudnn_ctc_loss
@pytest.mark.parametrize("shape", CTC_SHAPES)
def test_cudnn_ctc_loss_forward(shape):
    device = flag_gems.device
    torch.manual_seed(0)
    log_probs, targets, input_lengths, target_lengths = _make_inputs(shape, device)
    ref_log_probs = utils.to_reference(log_probs, True)
    ref_targets = utils.to_reference(targets)

    ref_loss, _, _ = _reference(
        ref_log_probs, ref_targets, input_lengths, target_lengths
    )

    res_loss, res_grad = flag_gems.cudnn_ctc_loss(
        log_probs, targets, input_lengths, target_lengths, 0, False, False
    )

    assert res_loss.shape == (shape[1],)
    assert res_loss.dtype == torch.float32
    utils.gems_assert_close(res_loss, ref_loss, torch.float32)


@pytest.mark.cudnn_ctc_loss
@pytest.mark.parametrize("shape", CTC_SHAPES)
def test_cudnn_ctc_loss_backward(shape):
    device = flag_gems.device
    torch.manual_seed(0)
    log_probs, targets, input_lengths, target_lengths = _make_inputs(shape, device)
    ref_log_probs = utils.to_reference(log_probs, True)
    ref_targets = utils.to_reference(targets)

    _, ref_grad, _ = _reference(
        ref_log_probs, ref_targets, input_lengths, target_lengths
    )

    _, res_grad = flag_gems.cudnn_ctc_loss(
        log_probs, targets, input_lengths, target_lengths, 0, False, False
    )

    assert res_grad.shape == log_probs.shape
    assert res_grad.dtype == torch.float32
    _assert_valid_region_close(res_grad, ref_grad, input_lengths)


@pytest.mark.cudnn_ctc_loss
@pytest.mark.parametrize(
    "input_lengths,target_lengths",
    [
        ([16, 16, 16, 16], [5, 5, 5, 5]),
        ([16, 12, 8, 15], [5, 3, 1, 4]),
        ([16, 16, 16, 16], [1, 2, 3, 4]),
        ([1, 2, 3, 4], [1, 1, 1, 1]),
    ],
)
def test_cudnn_ctc_loss_varying_lengths(input_lengths, target_lengths):
    device = flag_gems.device
    torch.manual_seed(0)
    # Fixed (T, N, C, S) host shape; only the per-batch lengths vary.
    shape = (16, 4, 16, 5)
    log_probs, targets, _, _ = _make_inputs(
        shape, device, input_lengths, target_lengths
    )
    ref_log_probs = utils.to_reference(log_probs, True)
    ref_targets = utils.to_reference(targets)

    ref_loss, ref_grad, _ = _reference(
        ref_log_probs, ref_targets, input_lengths, target_lengths
    )

    res_loss, res_grad = flag_gems.cudnn_ctc_loss(
        log_probs, targets, input_lengths, target_lengths, 0, False, False
    )

    utils.gems_assert_close(res_loss, ref_loss, torch.float32)
    _assert_valid_region_close(res_grad, ref_grad, input_lengths)


@pytest.mark.cudnn_ctc_loss
def test_cudnn_ctc_loss_empty_target():
    """An empty label sequence yields a zero loss and a `log_probs` gradient."""
    device = flag_gems.device
    torch.manual_seed(0)
    t_steps, batch, classes = 8, 2, 6
    raw = torch.randn(t_steps, batch, classes, dtype=torch.float32, device=device)
    log_probs = torch.log_softmax(raw, dim=-1)
    targets = torch.tensor([3], dtype=torch.int32)
    input_lengths = [5, 8]
    target_lengths = [0, 1]

    res_loss, res_grad = flag_gems.cudnn_ctc_loss(
        log_probs, targets, input_lengths, target_lengths, 0, False, False
    )

    # Zero loss for the empty target, and the gradient of the valid region is
    # `exp(log_probs)` since no label state can consume probability mass.
    assert res_loss[0].item() == 0.0
    ref_grad = utils.to_reference(log_probs[: input_lengths[0], 0], True).exp()
    utils.gems_assert_close(
        res_grad[: input_lengths[0], 0],
        ref_grad,
        torch.float32,
    )


@pytest.mark.cudnn_ctc_loss
def test_cudnn_ctc_loss_autograd():
    """The loss must be differentiable end to end through `log_probs`."""
    device = flag_gems.device
    torch.manual_seed(0)
    t_steps, batch, classes = 16, 3, 8
    raw = torch.randn(t_steps, batch, classes, dtype=torch.float32, device=device)
    log_probs = torch.log_softmax(raw, dim=-1)
    targets = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    target_lengths = [2, 1, 1]

    ref_lp = utils.to_reference(log_probs, True).clone().requires_grad_(True)
    ref = F.ctc_loss(
        ref_lp,
        targets.to(ref_lp.device),
        [t_steps] * batch,
        target_lengths,
        reduction="none",
        zero_infinity=False,
    )
    (ref_grad,) = torch.autograd.grad(ref.sum(), ref_lp)

    gems_lp = log_probs.clone().requires_grad_(True)
    out = flag_gems.cudnn_ctc_loss(
        gems_lp, targets, [t_steps] * batch, target_lengths, 0, False, False
    )
    gems_loss = out[0]
    (gems_grad,) = torch.autograd.grad(gems_loss.sum(), gems_lp)

    utils.gems_assert_close(gems_loss, ref.detach(), torch.float32)
    utils.gems_assert_close(gems_grad, ref_grad, torch.float32)


@pytest.mark.cudnn_ctc_loss
def test_cudnn_ctc_loss_weighted_grad():
    """Each batch entry must be scaled by its own upstream gradient.

    A plain `.sum()` upstream folds into a uniform gradient, which hides a
    per-batch indexing bug in the backward kernel: every sample would be
    scaled by `grad_output[0]` and still match. Weighting the upstream by a
    per-batch vector makes the scaling observable.
    """
    device = flag_gems.device
    torch.manual_seed(0)
    t_steps, batch, classes = 16, 3, 8
    raw = torch.randn(t_steps, batch, classes, dtype=torch.float32, device=device)
    log_probs = torch.log_softmax(raw, dim=-1)
    targets = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    target_lengths = [2, 1, 1]
    # One weight vector per side: under --ref=cpu the reference works on the
    # host while the operator stays on the device, and the upstream gradient has
    # to sit next to the loss it scales.
    upstream = [1.0, 0.0, -2.0]

    ref_lp = utils.to_reference(log_probs, True).clone().requires_grad_(True)
    ref = F.ctc_loss(
        ref_lp,
        targets.to(ref_lp.device),
        [t_steps] * batch,
        target_lengths,
        reduction="none",
        zero_infinity=False,
    )
    ref_weights = torch.tensor(upstream, dtype=ref_lp.dtype, device=ref_lp.device)
    (ref_grad,) = torch.autograd.grad((ref * ref_weights).sum(), ref_lp)

    gems_lp = log_probs.clone().requires_grad_(True)
    out = flag_gems.cudnn_ctc_loss(
        gems_lp, targets, [t_steps] * batch, target_lengths, 0, False, False
    )
    gems_weights = torch.tensor(upstream, dtype=log_probs.dtype, device=device)
    (gems_grad,) = torch.autograd.grad((out[0] * gems_weights).sum(), gems_lp)

    utils.gems_assert_close(gems_grad, ref_grad, torch.float32)


@pytest.mark.cudnn_ctc_loss
def test_cudnn_ctc_loss_tensor_lengths():
    """`input_lengths`/`target_lengths` may be passed as tensors."""
    device = flag_gems.device
    torch.manual_seed(0)
    # Fixed (T, N, C, S) shape; only the lengths-as-tensors path is exercised.
    shape = (16, 4, 16, 5)
    log_probs, targets, input_lengths, target_lengths = _make_inputs(shape, device)
    ref_log_probs = utils.to_reference(log_probs, True)
    ref_targets = utils.to_reference(targets)
    length_tensors = (
        torch.tensor(input_lengths, dtype=torch.int32),
        torch.tensor(target_lengths, dtype=torch.int32),
    )

    ref_loss, ref_grad, _ = _reference(
        ref_log_probs, ref_targets, input_lengths, target_lengths
    )

    res_loss, res_grad = flag_gems.cudnn_ctc_loss(
        log_probs, targets, length_tensors[0], length_tensors[1], 0, False, False
    )

    utils.gems_assert_close(res_loss, ref_loss, torch.float32)
    _assert_valid_region_close(res_grad, ref_grad, input_lengths)


@pytest.mark.cudnn_ctc_loss
def test_cudnn_ctc_loss_invalid_blank():
    """`blank` must be 0, matching the cuDNN restriction."""
    device = flag_gems.device
    log_probs, targets, input_lengths, target_lengths = _make_inputs(
        (8, 2, 6, 2), device
    )
    with pytest.raises(RuntimeError, match="blank must be label 0"):
        flag_gems.cudnn_ctc_loss(
            log_probs, targets, input_lengths, target_lengths, 1, False, False
        )

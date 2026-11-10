import pytest
import torch
import torch.nn.functional as F

import flag_gems

from . import accuracy_utils as utils
from .conftest import TO_CPU

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
    """An empty label sequence matches cuDNN on both loss and gradient.

    Compared against cuDNN itself rather than a re-derived formula: cuDNN
    reports a zero loss for the empty label sequence and seeds the gradient
    with `exp(log_probs)` (nothing accumulates posterior mass into state 0, so
    the seed survives the backward pass). Only the valid time steps are
    compared, since cuDNN leaves the padded rows uninitialized.
    """
    if TO_CPU:
        pytest.skip("_cudnn_ctc_loss is CUDA-only, cannot run in quick-cpu mode")

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

    # Captured at import time is not necessary: this is the native path unless a
    # previous test leaked a global registration (see the dispatch test, which is
    # scoped so it cannot).
    ref_loss, ref_grad = torch.ops.aten._cudnn_ctc_loss(
        log_probs,
        targets.to(device),
        torch.tensor(input_lengths, dtype=torch.int32, device=device),
        torch.tensor(target_lengths, dtype=torch.int32, device=device),
        0,
        False,
        False,
    )

    utils.gems_assert_close(res_loss, ref_loss, torch.float32)
    valid = input_lengths[0]
    utils.gems_assert_close(res_grad[:valid, 0], ref_grad[:valid, 0], torch.float32)
    # The empty-label gradient really is the `exp(log_probs)` seed, not zeros.
    utils.gems_assert_close(
        res_grad[:valid, 0], log_probs[:valid, 0].exp(), torch.float32
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


@pytest.mark.cudnn_ctc_loss
def test_cudnn_ctc_loss_tensor_lengths_dispatch():
    """The `.Tensor` overload must be reachable through dispatch, not only via
    the Python wrapper.

    `_cudnn_ctc_loss.Tensor` has to be in `_FULL_CONFIG` with an Autograd
    fallthrough; a registration regression is otherwise silent because
    `register_error` only logs. Calling the real schema under a scoped
    `use_gems` exercises the same path a global registration would, and the
    context tears the registration down on exit, so later tests still see the
    native cuDNN kernel in their references.
    """

    if TO_CPU:
        pytest.skip("_cudnn_ctc_loss is CUDA-only, cannot run in quick-cpu mode")

    device = flag_gems.device
    torch.manual_seed(0)
    shape = (16, 4, 16, 5)
    log_probs, targets, input_lengths, target_lengths = _make_inputs(shape, device)
    inp_len = torch.tensor(input_lengths, dtype=torch.int32, device=device)
    tgt_len = torch.tensor(target_lengths, dtype=torch.int32, device=device)

    direct_loss, direct_grad = flag_gems.cudnn_ctc_loss(
        log_probs, targets, inp_len, tgt_len, 0, False, False
    )

    with flag_gems.use_gems(include=["cudnn_ctc_loss"]):
        dispatch_loss, dispatch_grad = torch.ops.aten._cudnn_ctc_loss.Tensor(
            log_probs, targets, inp_len, tgt_len, 0, False, False
        )

    utils.gems_assert_close(dispatch_loss, direct_loss, torch.float32)
    utils.gems_assert_close(dispatch_grad, direct_grad, torch.float32)


@pytest.mark.cudnn_ctc_loss
def test_cudnn_ctc_loss_zero_infinity():
    """`zero_infinity` must not change the reported loss.

    cuDNN ignores the flag: an infeasible alignment already comes back as a
    zero loss with a zero gradient whether it is set or not, unlike
    `F.ctc_loss`, which reports `inf` and needs the flag to zero it. This pins
    the cuDNN contract so a future "fix" toward the `F.ctc_loss` convention
    cannot land silently.
    """

    if TO_CPU:
        pytest.skip("_cudnn_ctc_loss is CUDA-only, cannot run in quick-cpu mode")

    device = flag_gems.device
    torch.manual_seed(0)
    t_steps, batch, classes = 8, 2, 6
    raw = torch.randn(t_steps, batch, classes, dtype=torch.float32, device=device)
    log_probs = torch.log_softmax(raw, dim=-1)
    # Three labels into two time steps can never be aligned: each label needs a
    # step, and adjacent repeats need a separating blank on top.
    targets = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
    input_lengths = [2, 8]
    target_lengths = [3, 2]

    ref = torch.ops.aten._cudnn_ctc_loss(
        log_probs,
        targets.to(device),
        torch.tensor(input_lengths, dtype=torch.int32, device=device),
        torch.tensor(target_lengths, dtype=torch.int32, device=device),
        0,
        True,
        False,
    )

    for zero_infinity in (False, True):
        res_loss, res_grad = flag_gems.cudnn_ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            0,
            True,
            zero_infinity,
        )
        # Infeasible alignment: zero loss, zero gradient, no NaN.
        assert res_loss[0].item() == 0.0
        assert res_grad[:, 0].count_nonzero().item() == 0
        assert not torch.isnan(res_grad).any()
        utils.gems_assert_close(res_loss, ref[0], torch.float32)


@pytest.mark.cudnn_ctc_loss
def test_cudnn_ctc_loss_infeasible_alignment():
    """An infeasible alignment yields zero loss and zero gradient.

    Both shapes below reach the terminal-state reduction with every state still
    `-inf`: a label sequence longer than the input, and adjacent repeated
    labels with no time step left for the separating blank. The loss must come
    back as zero (not `inf`), and the gradient must be all zeros (not `NaN`
    from `inf - inf`, and not the `exp(log_probs)` seed the empty-target path
    leaves in place).

    The gradient of the *infeasible* batch entry is asserted directly rather
    than compared against cuDNN: cuDNN does not write that entry's gradient
    buffer at all, so its value is uninitialized memory (measured to be
    whatever the allocator last handed out). The feasible trailing entry is
    still compared against cuDNN.
    """

    if TO_CPU:
        pytest.skip("_cudnn_ctc_loss is CUDA-only, cannot run in quick-cpu mode")

    device = flag_gems.device
    torch.manual_seed(0)
    t_steps, batch, classes = 8, 2, 6
    raw = torch.randn(t_steps, batch, classes, dtype=torch.float32, device=device)
    log_probs = torch.log_softmax(raw, dim=-1)

    cases = [
        # (targets, input_lengths, target_lengths)
        ([1, 2, 3, 4, 5], [2, 8], [3, 2]),
        ([2, 2, 4], [2, 8], [2, 1]),
    ]
    for targets, input_lengths, target_lengths in cases:
        tgt = torch.tensor(targets, dtype=torch.int32)
        inp_len = torch.tensor(input_lengths, dtype=torch.int32, device=device)
        tgt_len = torch.tensor(target_lengths, dtype=torch.int32, device=device)
        ref_loss, ref_grad = torch.ops.aten._cudnn_ctc_loss(
            log_probs, tgt.to(device), inp_len, tgt_len, 0, True, False
        )

        res_loss, res_grad = flag_gems.cudnn_ctc_loss(
            log_probs, tgt, input_lengths, target_lengths, 0, True, False
        )

        assert res_loss[0].item() == 0.0, (targets, res_loss.tolist())
        assert res_grad[:, 0].count_nonzero().item() == 0
        assert not torch.isnan(res_grad).any()
        utils.gems_assert_close(res_loss, ref_loss, torch.float32)
        # Batch 1 is alignable, so cuDNN does define its gradient.
        utils.gems_assert_close(res_grad[:, 1], ref_grad[:, 1], torch.float32)


@pytest.mark.cudnn_ctc_loss
def test_cudnn_ctc_loss_empty_input():
    """An empty input with an empty target is the one infinite-loss case."""

    if TO_CPU:
        pytest.skip("_cudnn_ctc_loss is CUDA-only, cannot run in quick-cpu mode")

    device = flag_gems.device
    torch.manual_seed(0)
    t_steps, batch, classes = 8, 2, 6
    raw = torch.randn(t_steps, batch, classes, dtype=torch.float32, device=device)
    log_probs = torch.log_softmax(raw, dim=-1)
    targets = torch.tensor([1], dtype=torch.int32)
    input_lengths = [0, 8]
    target_lengths = [0, 1]

    res_loss, res_grad = flag_gems.cudnn_ctc_loss(
        log_probs, targets, input_lengths, target_lengths, 0, True, False
    )

    assert torch.isinf(res_loss[0])
    assert res_grad[:, 0].count_nonzero().item() == 0

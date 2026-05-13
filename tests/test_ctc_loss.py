"""Accuracy tests for ctc_loss.

Reference is torch.nn.functional.ctc_loss on CPU upcast to fp32.

Coverage:
    * dtype:         fp32 / fp16 / bf16
    * target layout: padded (N, S) / concatenated (sum_L,)
    * reduction:     none / mean / sum
    * blank index:   0 / nonzero
    * zero_infinity: True / False
    * shapes:        small / medium / long-target
    * edge cases:    empty targets, repeated labels, target_len > input_len,
                     unbatched (2D log_probs), int-list lengths, noncontiguous
                     log_probs, gradient correctness via torch.autograd.grad,
                     dispatcher path via flag_gems.use_gems()
"""
import pytest
import torch
import torch.nn.functional as F

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

# In quick mode we still cover the main dtype matrix; bf16 only if hardware says yes.
_BF16_OK = flag_gems.runtime.device.support_bf16
_DTYPES = [torch.float32, torch.float16]
if _BF16_OK:
    _DTYPES.append(torch.bfloat16)

CTC_DTYPES = [torch.float32] if cfg.QUICK_MODE else _DTYPES
REDUCTIONS = ["mean"] if cfg.QUICK_MODE else ["none", "mean", "sum"]
LAYOUTS = ["padded"] if cfg.QUICK_MODE else ["padded", "concatenated"]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _log_probs(shape, dtype, *, noncontiguous=False, requires_grad=True):
    """Build a contiguous (or transposed-then-non-contiguous) log_probs tensor."""
    if len(shape) == 2:
        raw = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
        out = raw.log_softmax(-1)
    elif noncontiguous:
        T, N, C = shape
        # (N, T, C) -> .transpose(0,1) -> (T, N, C), strided non-contiguous
        raw = torch.randn(N, T, C, dtype=torch.float32, device=flag_gems.device)
        out = raw.log_softmax(-1).transpose(0, 1)
    else:
        raw = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
        out = raw.log_softmax(-1)
    out = out.to(dtype).detach()
    if requires_grad:
        out.requires_grad_()
    return out


def _make_targets_from_rows(rows, layout, *, max_target=None, device=None):
    """Pack a list of int-lists into (targets, target_lengths) under a layout."""
    if device is None:
        device = flag_gems.device
    if max_target is None:
        max_target = max((len(r) for r in rows), default=0)
    target_lengths = torch.tensor(
        [len(r) for r in rows], device=device, dtype=torch.int64
    )
    padded = torch.zeros(len(rows), max_target, device=device, dtype=torch.int64)
    pieces = []
    for i, row in enumerate(rows):
        if len(row) > 0:
            t = torch.tensor(row, device=device, dtype=torch.int64)
            padded[i, : len(row)] = t
            pieces.append(t)
        else:
            pieces.append(torch.empty(0, device=device, dtype=torch.int64))
    if layout == "padded":
        return padded, target_lengths
    return (
        torch.cat(pieces)
        if pieces
        else torch.empty(0, device=device, dtype=torch.int64),
        target_lengths,
    )


def _synth_rows(batch, max_target, classes, blank, *, vary=True, repeated=False):
    """Generate synthetic label rows that avoid `blank` and exercise repeats."""
    values = [v for v in range(classes) if v != blank]
    if not values:
        values = [0]
    rows = []
    for i in range(batch):
        L = max(1, max_target - (i if vary else 0))
        if repeated:
            row = [values[i % len(values)]] * min(2, L)
            row += [values[(i + j + 1) % len(values)] for j in range(L - len(row))]
        else:
            row = [values[(i + j) % len(values)] for j in range(L)]
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Reference + assertions
# ---------------------------------------------------------------------------
def _reference(log_probs, targets, input_lengths, target_lengths, **kwargs):
    ref_lp = utils.to_reference(log_probs.detach(), False).to(torch.float32)
    ref_lp.requires_grad_(log_probs.requires_grad)
    ref_tgt = utils.to_reference(targets)
    ref_in = (
        utils.to_reference(input_lengths)
        if torch.is_tensor(input_lengths)
        else input_lengths
    )
    ref_tl = (
        utils.to_reference(target_lengths)
        if torch.is_tensor(target_lengths)
        else target_lengths
    )
    ref_out = F.ctc_loss(ref_lp, ref_tgt, ref_in, ref_tl, **kwargs)
    return ref_lp, ref_out.to(log_probs.dtype)


def _assert_forward_backward(
    log_probs, targets, input_lengths, target_lengths, dtype, **kwargs
):
    ref_lp, ref_out = _reference(
        log_probs, targets, input_lengths, target_lengths, **kwargs
    )
    res_out = flag_gems.ctc_loss(
        log_probs, targets, input_lengths, target_lengths, **kwargs
    )

    # reduce_dim heuristic: the lattice involves O(T * S') summations, so use
    # T * (max_target+1) as the effective fan-in for floating tolerance.
    T = log_probs.shape[0]
    if torch.is_tensor(target_lengths):
        max_tl = int(target_lengths.max().item()) if target_lengths.numel() > 0 else 0
    else:
        max_tl = max(target_lengths) if len(target_lengths) > 0 else 0
    reduce_dim = max(1, T * (max_tl + 1))

    utils.gems_assert_close(
        res_out, ref_out, dtype, equal_nan=True, reduce_dim=reduce_dim
    )

    # gradient
    grad = torch.randn_like(res_out)
    ref_grad_out = utils.to_reference(grad, False).to(ref_out.dtype)
    (ref_grad,) = torch.autograd.grad(ref_out, ref_lp, ref_grad_out)
    (res_grad,) = torch.autograd.grad(res_out, log_probs, grad)
    utils.gems_assert_close(
        res_grad, ref_grad, dtype, equal_nan=True, reduce_dim=reduce_dim
    )


# ---------------------------------------------------------------------------
# Core matrix: dtype x layout x reduction
# ---------------------------------------------------------------------------
@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", CTC_DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_ctc_loss_basic(dtype, layout, reduction):
    utils.init_seed(20260512)
    T, N, C, max_target = 16, 4, 8, 5
    log_probs = _log_probs((T, N, C), dtype)
    rows = _synth_rows(N, max_target, C, blank=0)
    targets, target_lengths = _make_targets_from_rows(
        rows, layout, max_target=max_target
    )
    input_lengths = torch.tensor(
        [T, T - 1, T - 2, T - 3], device=flag_gems.device, dtype=torch.int64
    )
    _assert_forward_backward(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        dtype,
        blank=0,
        reduction=reduction,
        zero_infinity=False,
    )


# ---------------------------------------------------------------------------
# Nonzero blank + repeated labels + zero_infinity True
# ---------------------------------------------------------------------------
@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", CTC_DTYPES)
def test_ctc_loss_nonzero_blank_repeated(dtype):
    utils.init_seed(20260513)
    log_probs = _log_probs((12, 3, 7), dtype, noncontiguous=True)
    rows = _synth_rows(3, 4, 7, blank=2, repeated=True)
    targets, target_lengths = _make_targets_from_rows(rows, "padded", max_target=4)
    input_lengths = torch.tensor(
        [12, 10, 8], device=flag_gems.device, dtype=torch.int64
    )
    _assert_forward_backward(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        dtype,
        blank=2,
        reduction="mean",
        zero_infinity=False,
    )


# ---------------------------------------------------------------------------
# Empty targets (single batch element)
# ---------------------------------------------------------------------------
@pytest.mark.ctc_loss
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_ctc_loss_some_empty_targets(layout, reduction):
    utils.init_seed(20260514)
    log_probs = _log_probs((10, 3, 6), torch.float32)
    targets, target_lengths = _make_targets_from_rows(
        [[], [1, 4], [3]],
        layout,
        max_target=3,
    )
    input_lengths = torch.tensor([10, 8, 5], device=flag_gems.device, dtype=torch.int64)
    _assert_forward_backward(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        torch.float32,
        blank=2,
        reduction=reduction,
        zero_infinity=False,
    )


@pytest.mark.ctc_loss
@pytest.mark.parametrize("layout", LAYOUTS)
def test_ctc_loss_all_empty_targets(layout):
    utils.init_seed(20260515)
    log_probs = _log_probs((6, 2, 5), torch.float32)
    targets, target_lengths = _make_targets_from_rows(
        [[], []],
        layout,
        max_target=0,
    )
    input_lengths = torch.tensor([6, 4], device=flag_gems.device, dtype=torch.int64)
    _assert_forward_backward(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        torch.float32,
        blank=1,
        reduction="sum",
        zero_infinity=False,
    )


# ---------------------------------------------------------------------------
# target_len > input_len with zero_infinity True (loss == 0 enforced)
# ---------------------------------------------------------------------------
@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", CTC_DTYPES)
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_ctc_loss_zero_infinity_impossible(dtype, reduction):
    utils.init_seed(20260516)
    log_probs = _log_probs((8, 2, 6), dtype)
    # target length > input length -> CTC = inf without zero_infinity
    targets, target_lengths = _make_targets_from_rows(
        [[1, 1, 2, 2, 3], [4, 4, 1]],
        "padded",
        max_target=5,
    )
    input_lengths = torch.tensor([3, 2], device=flag_gems.device, dtype=torch.int64)
    _assert_forward_backward(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        dtype,
        blank=5,
        reduction=reduction,
        zero_infinity=True,
    )


# ---------------------------------------------------------------------------
# Padded vs concatenated equivalence
# ---------------------------------------------------------------------------
@pytest.mark.ctc_loss
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_ctc_loss_padded_concat_equivalent(reduction):
    utils.init_seed(20260517)
    rows = [[1, 2, 3], [4, 5], [1]]
    padded, target_lengths = _make_targets_from_rows(rows, "padded", max_target=3)
    concat, _ = _make_targets_from_rows(rows, "concatenated", max_target=3)
    input_lengths = torch.tensor([7, 6, 5], device=flag_gems.device, dtype=torch.int64)

    lp_p = _log_probs((7, 3, 6), torch.float32)
    lp_c = lp_p.detach().clone().requires_grad_()

    out_p = flag_gems.ctc_loss(
        lp_p,
        padded,
        input_lengths,
        target_lengths,
        blank=0,
        reduction=reduction,
        zero_infinity=False,
    )
    out_c = flag_gems.ctc_loss(
        lp_c,
        concat,
        input_lengths,
        target_lengths,
        blank=0,
        reduction=reduction,
        zero_infinity=False,
    )
    utils.gems_assert_close(out_p, out_c, torch.float32, reduce_dim=7 * 4)

    grad = torch.randn_like(out_p)
    (gp,) = torch.autograd.grad(out_p, lp_p, grad)
    (gc,) = torch.autograd.grad(out_c, lp_c, grad)
    utils.gems_assert_close(gp, gc, torch.float32, reduce_dim=7 * 4)


# ---------------------------------------------------------------------------
# Unbatched (2D log_probs) + int-list lengths
# ---------------------------------------------------------------------------
@pytest.mark.ctc_loss
def test_ctc_loss_unbatched():
    utils.init_seed(20260518)
    log_probs = _log_probs((10, 6), torch.float32)
    targets = torch.tensor([1, 3, 4], device=flag_gems.device, dtype=torch.int64)
    out = flag_gems.ctc_loss(log_probs, targets, [10], [3], blank=0, reduction="sum")
    ref_lp, ref = _reference(log_probs, targets, [10], [3], blank=0, reduction="sum")
    utils.gems_assert_close(out, ref, torch.float32, reduce_dim=10 * 4)

    g = torch.ones_like(out)
    (res_grad,) = torch.autograd.grad(out, log_probs, g)
    (ref_grad,) = torch.autograd.grad(ref, ref_lp, g)
    utils.gems_assert_close(res_grad, ref_grad, torch.float32, reduce_dim=10 * 4)


# ---------------------------------------------------------------------------
# Long target  (stresses BLOCK_S = next_power_of_2(2L+1))
# Exercises the 8-warp + tl.debug_barrier() path: at BLOCK_S >= 256 the kernel
# switches to 8 warps, which made the previously-implicit cross-warp ordering
# of alpha[t] store -> alpha[t-1] load on the next iter unsafe.  Parametrize
# over all dtypes to verify the barrier closes the race on every precision.
# ---------------------------------------------------------------------------
@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", CTC_DTYPES)
def test_ctc_loss_long_target(dtype):
    if cfg.QUICK_MODE:
        pytest.skip("long-target test skipped in quick mode")
    utils.init_seed(20260519)
    # L = 100 -> S' = 201 -> BLOCK_S = 256.
    T, N, C, L = 256, 2, 50, 100
    log_probs = _log_probs((T, N, C), dtype)
    rows = _synth_rows(N, L, C, blank=0)
    targets, target_lengths = _make_targets_from_rows(rows, "padded", max_target=L)
    input_lengths = torch.tensor(
        [T, T - 20], device=flag_gems.device, dtype=torch.int64
    )
    _assert_forward_backward(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        dtype,
        blank=0,
        reduction="mean",
        zero_infinity=False,
    )


# ---------------------------------------------------------------------------
# target_length > input_length: the CTC alignment is impossible (a label
# sequence of length L needs at least L emissions to be decodable), so loss
# should be +inf without zero_infinity and 0 with it.  This corner is easy to
# get wrong in DP implementations because the "skip blank" rule + the
# (S'_n - 1, S'_n - 2) terminal logsumexp can both end up reading -inf.
# ---------------------------------------------------------------------------
@pytest.mark.ctc_loss
def test_ctc_loss_target_longer_than_input():
    utils.init_seed(20260601)
    T, N, C = 4, 2, 5
    L = 8  # > T -> alignment impossible
    log_probs = _log_probs((T, N, C), torch.float32, requires_grad=False)
    rows = _synth_rows(N, L, C, blank=0)
    targets, target_lengths = _make_targets_from_rows(rows, "padded", max_target=L)
    input_lengths = torch.full((N,), T, device=flag_gems.device, dtype=torch.int64)

    # Without zero_infinity: expect +inf loss.
    raw = flag_gems.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=0,
        reduction="none",
        zero_infinity=False,
    )
    assert torch.all(torch.isinf(raw)), f"expected all-inf loss, got {raw}"

    # With zero_infinity: expect finite zero loss and finite gradient.
    log_probs_g = log_probs.detach().clone().requires_grad_()
    finite = flag_gems.ctc_loss(
        log_probs_g,
        targets,
        input_lengths,
        target_lengths,
        blank=0,
        reduction="mean",
        zero_infinity=True,
    )
    assert torch.isfinite(
        finite
    ), f"zero_infinity should produce finite loss, got {finite}"
    assert (
        finite.item() == 0.0
    ), f"zero_infinity should zero the loss, got {finite.item()}"
    finite.backward()
    assert torch.all(
        torch.isfinite(log_probs_g.grad)
    ), "grad must be finite under zero_infinity"


# ---------------------------------------------------------------------------
# Realistic ASR-ish workload  (small classes, moderate batch)
# ---------------------------------------------------------------------------
@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", CTC_DTYPES)
def test_ctc_loss_asr_workload(dtype):
    utils.init_seed(20260520)
    T, N, C, L = 50, 8, 30, 15
    log_probs = _log_probs((T, N, C), dtype)
    rows = _synth_rows(N, L, C, blank=29)
    targets, target_lengths = _make_targets_from_rows(
        rows, "concatenated", max_target=L
    )
    input_lengths = torch.full((N,), T, device=flag_gems.device, dtype=torch.int64)
    _assert_forward_backward(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        dtype,
        blank=29,
        reduction="mean",
        zero_infinity=False,
    )


# ---------------------------------------------------------------------------
# Variable input lengths (heterogeneous T_n)
# ---------------------------------------------------------------------------
@pytest.mark.ctc_loss
def test_ctc_loss_variable_input_lengths():
    utils.init_seed(20260521)
    T, N, C, L = 20, 5, 10, 6
    log_probs = _log_probs((T, N, C), torch.float32)
    rows = _synth_rows(N, L, C, blank=0)
    targets, target_lengths = _make_targets_from_rows(rows, "padded", max_target=L)
    input_lengths = torch.tensor(
        [T, T - 2, T - 5, T - 8, T - 12], device=flag_gems.device, dtype=torch.int64
    )
    _assert_forward_backward(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        torch.float32,
        blank=0,
        reduction="mean",
        zero_infinity=False,
    )


# ---------------------------------------------------------------------------
# Dispatcher path (F.ctc_loss via use_gems)
# ---------------------------------------------------------------------------
@pytest.mark.ctc_loss
def test_ctc_loss_dispatcher():
    utils.init_seed(20260522)
    log_probs = _log_probs((9, 2, 6), torch.float32)
    rows = _synth_rows(2, 3, 6, blank=0)
    targets, target_lengths = _make_targets_from_rows(
        rows, "concatenated", max_target=3
    )
    input_lengths = torch.tensor([9, 8], device=flag_gems.device, dtype=torch.int64)

    ref_lp, ref = _reference(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=0,
        reduction="mean",
        zero_infinity=False,
    )
    with flag_gems.use_gems(include=["ctc_loss"]):
        out = F.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            reduction="mean",
            zero_infinity=False,
        )
    utils.gems_assert_close(out, ref, torch.float32, reduce_dim=9 * 4)


# ---------------------------------------------------------------------------
# Invalid input rejection
# ---------------------------------------------------------------------------
@pytest.mark.ctc_loss
def test_ctc_loss_rejects_float_lengths():
    log_probs = _log_probs((6, 2, 5), torch.float32, requires_grad=False)
    targets = torch.tensor([[1, 2], [2, 3]], device=flag_gems.device, dtype=torch.int64)
    target_lengths = torch.tensor([2, 2], device=flag_gems.device, dtype=torch.int64)
    bad_input_lengths = torch.tensor([6.0, 6.0], device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.ctc_loss(log_probs, targets, bad_input_lengths, target_lengths)


@pytest.mark.ctc_loss
def test_ctc_loss_rejects_input_len_over_T():
    log_probs = _log_probs((5, 2, 5), torch.float32, requires_grad=False)
    targets = torch.tensor([[1, 2], [2, 3]], device=flag_gems.device, dtype=torch.int64)
    target_lengths = torch.tensor([2, 2], device=flag_gems.device, dtype=torch.int64)
    bad_input_lengths = torch.tensor([5, 7], device=flag_gems.device, dtype=torch.int64)
    with pytest.raises(RuntimeError):
        flag_gems.ctc_loss(log_probs, targets, bad_input_lengths, target_lengths)


@pytest.mark.ctc_loss
def test_ctc_loss_rejects_invalid_blank():
    log_probs = _log_probs((4, 1, 3), torch.float32, requires_grad=False)
    targets = torch.tensor([[1, 2]], device=flag_gems.device, dtype=torch.int64)
    target_lengths = torch.tensor([2], device=flag_gems.device, dtype=torch.int64)
    input_lengths = torch.tensor([4], device=flag_gems.device, dtype=torch.int64)
    with pytest.raises(RuntimeError):
        flag_gems.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=3)

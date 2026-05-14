import pytest
import torch
import torch.nn.functional as F

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg


def _get_bf16_is_supported():
    value = getattr(utils, "bf16_is_supported", None)
    if value is None:
        value = getattr(cfg, "bf16_is_supported", None)
    if callable(value):
        return bool(value())
    if value is not None:
        return bool(value)

    device = torch.device(flag_gems.device)
    if device.type == "cuda" and torch.cuda.is_available():
        return bool(torch.cuda.is_bf16_supported())
    return False


bf16_is_supported = _get_bf16_is_supported()
PRIMARY_FLOAT_DTYPES = [torch.float16, torch.float32]
FLOAT_DTYPES = (
    PRIMARY_FLOAT_DTYPES + [torch.bfloat16]
    if bf16_is_supported
    else PRIMARY_FLOAT_DTYPES
)

if cfg.QUICK_MODE:
    CTC_SHAPES = [(8, 2, 10, 5)]
    CTC_REDUCTIONS = ["mean"]
    CTC_BLANKS = [0]
else:
    CTC_SHAPES = [
        (5, 1, 10, 3),
        (8, 2, 10, 5),
        (20, 4, 15, 7),
        (50, 4, 28, 10),
        (30, 3, 50, 15),
        (100, 2, 10, 3),
    ]
    CTC_REDUCTIONS = ["none", "mean", "sum"]


def _int_reduction(reduction):
    mapping = {"none": 0, "mean": 1, "sum": 2}
    return mapping[reduction]


def _ctc_loss_reference(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    blank=0,
    reduction="mean",
    zero_infinity=False,
):
    """Reference path: compute fp16/bf16 CTC in fp32, then cast output back."""
    original_dtype = log_probs.dtype
    work_log_probs = log_probs
    if work_log_probs.dtype in (torch.float16, torch.bfloat16):
        work_log_probs = work_log_probs.float()

    out = F.ctc_loss(
        work_log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction=reduction,
        zero_infinity=zero_infinity,
    )
    return out.to(original_dtype) if out.dtype != original_dtype else out


def _make_targets(batch, max_target, classes, device, target_layout, blank=0):
    """Generate targets in padded or concatenated layout."""
    target_lengths = torch.empty(batch, device=device, dtype=torch.long)
    padded = torch.zeros(batch, max_target, device=device, dtype=torch.long)
    pieces = []
    for row in range(batch):
        length = max(1, max_target - (row % 5))
        target_lengths[row] = length
        values = (torch.arange(length, device=device, dtype=torch.long) + row) % (
            classes - 1
        )
        values = values + 1
        padded[row, :length] = values
        pieces.append(values)
    if target_layout == "padded":
        targets = padded
    else:
        targets = torch.cat(pieces)
    return targets, target_lengths


@pytest.mark.ctc_loss
@pytest.mark.parametrize("T, N, C, S", CTC_SHAPES)
@pytest.mark.parametrize("reduction", CTC_REDUCTIONS)
@pytest.mark.parametrize("target_layout", ["padded", "concatenated"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_ctc_loss_core(T, N, C, S, dtype, target_layout, reduction):
    """Core forward+backward accuracy test for padded and concatenated targets."""
    blank = 0
    raw = torch.randn(
        T, N, C, dtype=torch.float32, device=flag_gems.device, requires_grad=True
    )
    log_probs = raw.log_softmax(-1).to(dtype)
    targets, target_lengths = _make_targets(
        N, S, C, flag_gems.device, target_layout, blank
    )
    input_lengths = torch.full((N,), T, dtype=torch.long, device=flag_gems.device)

    ref_lp = utils.to_reference(log_probs, True)
    ref_targets = utils.to_reference(targets)
    ref_il = utils.to_reference(input_lengths)
    ref_tl = utils.to_reference(target_lengths)

    with torch.backends.cudnn.flags(enabled=False):
        ref_out = _ctc_loss_reference(
            ref_lp,
            ref_targets,
            ref_il,
            ref_tl,
            blank=blank,
            reduction=reduction,
            zero_infinity=False,
        )

    res_out = flag_gems.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction=_int_reduction(reduction),
        zero_infinity=False,
    )

    atol = 0.01 if dtype == torch.float16 else 0.001
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=max(C, 1), atol=atol)

    if reduction != "none":
        out_grad = torch.randn_like(res_out)
        ref_grad = utils.to_reference(out_grad, True)
        (ref_in_grad,) = torch.autograd.grad(ref_out, ref_lp, ref_grad)
        (res_in_grad,) = torch.autograd.grad(res_out, log_probs, out_grad)
        utils.gems_assert_close(
            res_in_grad,
            ref_in_grad,
            dtype,
            reduce_dim=max(C, 1),
            atol=0.02,
            equal_nan=True,
        )


@pytest.mark.ctc_loss
@pytest.mark.parametrize("T, N, C, S", CTC_SHAPES[:2])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("target_layout", ["padded", "concatenated"])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_ctc_loss_zero_infinity(T, N, C, S, dtype, target_layout, reduction):
    """Test zero_infinity=True behavior."""
    blank = 0
    log_probs = (
        torch.randn(
            T, N, C, dtype=torch.float32, device=flag_gems.device, requires_grad=True
        )
        .log_softmax(-1)
        .to(dtype)
    )
    targets, target_lengths = _make_targets(
        N, S, C, flag_gems.device, target_layout, blank
    )
    input_lengths = torch.full((N,), T, dtype=torch.long, device=flag_gems.device)

    ref_lp = utils.to_reference(log_probs, True)
    ref_targets = utils.to_reference(targets)
    ref_il = utils.to_reference(input_lengths)
    ref_tl = utils.to_reference(target_lengths)

    with torch.backends.cudnn.flags(enabled=False):
        ref_out = _ctc_loss_reference(
            ref_lp,
            ref_targets,
            ref_il,
            ref_tl,
            blank=blank,
            reduction=reduction,
            zero_infinity=True,
        )
    res_out = flag_gems.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction=_int_reduction(reduction),
        zero_infinity=True,
    )
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=max(C, 1), atol=0.001)


@pytest.mark.ctc_loss
@pytest.mark.parametrize("T, N, C, S", CTC_SHAPES[:2])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_ctc_loss_no_grad(T, N, C, S, dtype):
    """Test forward execution without gradient computation."""
    blank = 0
    log_probs = torch.randn(
        T, N, C, dtype=dtype, device=flag_gems.device, requires_grad=False
    ).log_softmax(dim=2)
    targets, target_lengths = _make_targets(N, S, C, flag_gems.device, "padded", blank)
    input_lengths = torch.full((N,), T, dtype=torch.long, device=flag_gems.device)

    res_out = flag_gems.ctc_loss(
        log_probs, targets, input_lengths, target_lengths, blank=blank, reduction=1
    )

    ref_lp = utils.to_reference(log_probs, True)
    ref_targets = utils.to_reference(targets)
    ref_il = utils.to_reference(input_lengths)
    ref_tl = utils.to_reference(target_lengths)

    with torch.backends.cudnn.flags(enabled=False):
        ref_out = _ctc_loss_reference(
            ref_lp, ref_targets, ref_il, ref_tl, blank=blank, reduction="mean"
        )
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=max(C, 1), atol=1e-4)


@pytest.mark.ctc_loss
@pytest.mark.parametrize("T, C, S", [(8, 10, 5)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_ctc_loss_unbatched(T, C, S, dtype):
    """Test unbatched 2D input (T, C)."""
    blank = 0
    log_probs = (
        torch.randn(
            T, C, dtype=torch.float32, device=flag_gems.device, requires_grad=True
        )
        .log_softmax(dim=1)
        .to(dtype)
    )
    targets, target_lengths = _make_targets(1, S, C, flag_gems.device, "padded", blank)
    targets = targets.squeeze(0)
    target_lengths = target_lengths.squeeze(0)
    input_lengths = target_lengths.new_tensor([T])

    ref_lp = utils.to_reference(log_probs, True)
    ref_targets = utils.to_reference(targets)
    ref_il = utils.to_reference(input_lengths)
    ref_tl = utils.to_reference(target_lengths)

    with torch.backends.cudnn.flags(enabled=False):
        ref_out = _ctc_loss_reference(
            ref_lp, ref_targets, ref_il, ref_tl, blank=blank, reduction="mean"
        )
    res_out = flag_gems.ctc_loss(
        log_probs, targets, input_lengths, target_lengths, blank=blank, reduction=1
    )
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=C, atol=1e-4)

    out_grad = torch.randn_like(res_out)
    ref_grad = utils.to_reference(out_grad, True)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_lp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, log_probs, out_grad)
    utils.gems_assert_close(
        res_in_grad, ref_in_grad, dtype, reduce_dim=C, atol=0.02, equal_nan=True
    )


@pytest.mark.ctc_loss
@pytest.mark.parametrize("T, N, C, S", [(10, 2, 6, 5)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_ctc_loss_variable_lengths(T, N, C, S, dtype):
    """Test input_lengths != T, with variable target lengths."""
    blank = 0
    log_probs = (
        torch.randn(
            T, N, C, dtype=torch.float32, device=flag_gems.device, requires_grad=True
        )
        .log_softmax(-1)
        .to(dtype)
    )
    targets, target_lengths = _make_targets(N, S, C, flag_gems.device, "padded", blank)
    input_lengths = torch.tensor([7, 10], dtype=torch.long, device=flag_gems.device)

    ref_lp = utils.to_reference(log_probs, True)
    ref_targets = utils.to_reference(targets)
    ref_il = utils.to_reference(input_lengths)
    ref_tl = utils.to_reference(target_lengths)

    with torch.backends.cudnn.flags(enabled=False):
        ref_out = _ctc_loss_reference(
            ref_lp, ref_targets, ref_il, ref_tl, blank=blank, reduction="mean"
        )
    res_out = flag_gems.ctc_loss(
        log_probs, targets, input_lengths, target_lengths, blank=blank, reduction=1
    )
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=C, atol=1e-4)


@pytest.mark.ctc_loss
@pytest.mark.parametrize("T, N, C, S", [(10, 2, 8, 3)])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_ctc_loss_concat_variable_lengths(T, N, C, S, dtype, reduction):
    """Test concatenated targets with variable lengths regression."""
    blank = 0
    raw = torch.randn(
        T, N, C, dtype=torch.float32, device=flag_gems.device, requires_grad=True
    )
    log_probs = raw.log_softmax(-1).to(dtype)
    targets, target_lengths = _make_targets(
        N, S, C, flag_gems.device, "concatenated", blank
    )
    input_lengths = torch.full((N,), T, dtype=torch.long, device=flag_gems.device)

    ref_lp = utils.to_reference(log_probs, True)
    ref_targets = utils.to_reference(targets)
    ref_il = utils.to_reference(input_lengths)
    ref_tl = utils.to_reference(target_lengths)

    with torch.backends.cudnn.flags(enabled=False):
        ref_out = _ctc_loss_reference(
            ref_lp,
            ref_targets,
            ref_il,
            ref_tl,
            blank=blank,
            reduction=reduction,
            zero_infinity=False,
        )
    res_out = flag_gems.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction=_int_reduction(reduction),
        zero_infinity=False,
    )
    atol = 0.01 if dtype == torch.float16 else 0.001
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=max(C, 1), atol=atol)


@pytest.mark.ctc_loss
@pytest.mark.parametrize("T, N, C, S", CTC_SHAPES[:2])
@pytest.mark.parametrize("reduction", CTC_REDUCTIONS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_ctc_loss_backward(T, N, C, S, dtype, reduction):
    """Dedicated backward gradient accuracy test."""
    blank = 0
    raw = torch.randn(
        T, N, C, dtype=torch.float32, device=flag_gems.device, requires_grad=True
    )
    log_probs = raw.log_softmax(-1).to(dtype)
    targets, target_lengths = _make_targets(N, S, C, flag_gems.device, "padded", blank)
    input_lengths = torch.full((N,), T, dtype=torch.long, device=flag_gems.device)

    ref_lp = utils.to_reference(log_probs, True)
    ref_targets = utils.to_reference(targets)
    ref_il = utils.to_reference(input_lengths)
    ref_tl = utils.to_reference(target_lengths)

    with torch.backends.cudnn.flags(enabled=False):
        ref_out = _ctc_loss_reference(
            ref_lp, ref_targets, ref_il, ref_tl, blank=blank, reduction=reduction
        )
    res_out = flag_gems.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction=_int_reduction(reduction),
    )

    out_grad = torch.randn_like(res_out)
    ref_grad = utils.to_reference(out_grad, True)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_lp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, log_probs, out_grad)

    tol = 0.05 if dtype == torch.float16 else 0.02
    utils.gems_assert_close(
        res_in_grad, ref_in_grad, dtype, reduce_dim=max(C, 1), atol=tol, equal_nan=True
    )


@pytest.mark.ctc_loss
@pytest.mark.parametrize("T, N, C, S", [(8, 2, 8, 3)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_ctc_loss_non_contiguous(T, N, C, S, dtype):
    """Test non-contiguous inputs."""
    blank = 0
    # Create non-contiguous log_probs
    big = torch.randn(T * 2, N, C, dtype=torch.float32, device=flag_gems.device)
    log_probs = big[::2].log_softmax(-1).to(dtype).requires_grad_()
    targets, target_lengths = _make_targets(N, S, C, flag_gems.device, "padded", blank)

    # Non-contiguous lengths
    big_len = torch.full((N * 2,), T, dtype=torch.long, device=flag_gems.device)
    input_lengths = big_len[::2]

    ref_lp = utils.to_reference(log_probs, True)
    ref_targets = utils.to_reference(targets)
    ref_il = utils.to_reference(input_lengths)
    ref_tl = utils.to_reference(target_lengths)

    with torch.backends.cudnn.flags(enabled=False):
        ref_out = _ctc_loss_reference(
            ref_lp, ref_targets, ref_il, ref_tl, blank=blank, reduction="mean"
        )
    res_out = flag_gems.ctc_loss(
        log_probs, targets, input_lengths, target_lengths, blank=blank, reduction=1
    )
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=C, atol=1e-4)


# --- input validation tests ---


@pytest.mark.ctc_loss
@pytest.mark.parametrize("bad_shape", [(8,), (2, 3, 4, 5)])
def test_ctc_loss_rank_validation(bad_shape):
    """Input validation: log_probs rank must be 2 or 3."""
    log_probs = torch.randn(bad_shape, dtype=torch.float32, device=flag_gems.device)
    targets = torch.randint(1, 3, (1, 2), dtype=torch.long, device=flag_gems.device)
    input_lengths = torch.tensor([8], dtype=torch.long, device=flag_gems.device)
    target_lengths = torch.tensor([2], dtype=torch.long, device=flag_gems.device)
    with pytest.raises(Exception):
        flag_gems.ctc_loss(
            log_probs, targets, input_lengths, target_lengths, blank=0, reduction=1
        )


@pytest.mark.ctc_loss
def test_ctc_loss_invalid_blank_validation():
    """Input validation: blank must be in [0, C)."""
    T, N, C, S = 8, 2, 5, 3
    log_probs = torch.randn(
        T, N, C, dtype=torch.float32, device=flag_gems.device
    ).log_softmax(-1)
    targets, target_lengths = _make_targets(N, S, C, flag_gems.device, "padded")
    input_lengths = torch.full((N,), T, dtype=torch.long, device=flag_gems.device)

    with pytest.raises(Exception):
        flag_gems.ctc_loss(
            log_probs, targets, input_lengths, target_lengths, blank=-1, reduction=1
        )
    with pytest.raises(Exception):
        flag_gems.ctc_loss(
            log_probs, targets, input_lengths, target_lengths, blank=C, reduction=1
        )


@pytest.mark.ctc_loss
def test_ctc_loss_invalid_length_size_validation():
    """Input validation: length tensors/lists must have one entry per batch element."""
    T, N, C, S = 8, 2, 6, 3
    log_probs = torch.randn(
        T, N, C, dtype=torch.float32, device=flag_gems.device
    ).log_softmax(-1)
    targets, target_lengths = _make_targets(N, S, C, flag_gems.device, "padded")
    input_lengths = torch.full((N,), T, dtype=torch.long, device=flag_gems.device)

    with pytest.raises(Exception):
        flag_gems.ctc_loss(
            log_probs, targets, input_lengths[:1], target_lengths, blank=0, reduction=1
        )
    with pytest.raises(Exception):
        flag_gems.ctc_loss(
            log_probs, targets, input_lengths, target_lengths[:1], blank=0, reduction=1
        )


@pytest.mark.ctc_loss
def test_ctc_loss_invalid_target_layout_validation():
    """Input validation: padded width and concatenated length must match target_lengths."""
    T, N, C = 8, 2, 6
    log_probs = torch.randn(
        T, N, C, dtype=torch.float32, device=flag_gems.device
    ).log_softmax(-1)
    input_lengths = torch.full((N,), T, dtype=torch.long, device=flag_gems.device)

    padded = torch.ones(N, 2, dtype=torch.long, device=flag_gems.device)
    too_long_tl = torch.tensor([3, 2], dtype=torch.long, device=flag_gems.device)
    with pytest.raises(Exception):
        flag_gems.ctc_loss(
            log_probs, padded, input_lengths, too_long_tl, blank=0, reduction=1
        )

    concat = torch.ones(4, dtype=torch.long, device=flag_gems.device)
    bad_tl = torch.tensor([3, 3], dtype=torch.long, device=flag_gems.device)
    with pytest.raises(Exception):
        flag_gems.ctc_loss(
            log_probs, concat, input_lengths, bad_tl, blank=0, reduction=1
        )


@pytest.mark.ctc_loss
@pytest.mark.parametrize("T, N, C, S", [(8, 2, 8, 3)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_ctc_loss_blank_arg(T, N, C, S, dtype):
    """Test that non-default blank label works."""
    blank = C - 1  # use last class as blank
    log_probs = (
        torch.randn(T, N, C, dtype=dtype, device=flag_gems.device, requires_grad=True)
        .log_softmax(dim=2)
        .to(dtype)
    )
    targets, target_lengths = _make_targets(
        N, S, C, flag_gems.device, "padded", blank=blank
    )
    input_lengths = torch.full((N,), T, dtype=torch.long, device=flag_gems.device)

    ref_lp = utils.to_reference(log_probs, True)
    ref_targets = utils.to_reference(targets)
    ref_il = utils.to_reference(input_lengths)
    ref_tl = utils.to_reference(target_lengths)

    with torch.backends.cudnn.flags(enabled=False):
        ref_out = _ctc_loss_reference(
            ref_lp, ref_targets, ref_il, ref_tl, blank=blank, reduction="mean"
        )
    res_out = flag_gems.ctc_loss(
        log_probs, targets, input_lengths, target_lengths, blank=blank, reduction=1
    )
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=C, atol=1e-4)


@pytest.mark.ctc_loss
@pytest.mark.parametrize("T, N, C, S", [(5, 2, 8, 3)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_ctc_loss_empty_target(T, N, C, S, dtype):
    """Edge case: empty target for one batch element."""
    blank = 0
    log_probs = (
        torch.randn(T, N, C, dtype=dtype, device=flag_gems.device, requires_grad=True)
        .log_softmax(dim=2)
        .to(dtype)
    )
    targets = torch.zeros(N, S, dtype=torch.long, device=flag_gems.device) + blank
    target_lengths = torch.tensor([0, S], dtype=torch.long, device=flag_gems.device)
    input_lengths = torch.full((N,), T, dtype=torch.long, device=flag_gems.device)

    res_out = flag_gems.ctc_loss(
        log_probs, targets, input_lengths, target_lengths, blank=blank, reduction=1
    )

    ref_lp = utils.to_reference(log_probs, True)
    ref_targets = utils.to_reference(targets)
    ref_il = utils.to_reference(input_lengths)
    ref_tl = utils.to_reference(target_lengths)

    with torch.backends.cudnn.flags(enabled=False):
        ref_out = _ctc_loss_reference(
            ref_lp, ref_targets, ref_il, ref_tl, blank=blank, reduction="mean"
        )
    # Check output is finite
    assert torch.isfinite(res_out).all()
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=C, atol=1e-4)


@pytest.mark.ctc_loss
@pytest.mark.parametrize("target_layout", ["padded", "concatenated"])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_ctc_loss_lengths_as_python_list(target_layout, reduction):
    """Cover list/tuple input_lengths and target_lengths, not only tensor lengths."""
    T, N, C, S = 10, 2, 8, 4
    blank = 0
    log_probs = torch.randn(
        T, N, C, dtype=torch.float32, device=flag_gems.device, requires_grad=True
    ).log_softmax(-1)
    targets, target_lengths_t = _make_targets(
        N, S, C, flag_gems.device, target_layout, blank
    )
    input_lengths_t = torch.tensor(
        [T, T - 2], dtype=torch.long, device=flag_gems.device
    )

    input_lengths = [int(v) for v in input_lengths_t.cpu().tolist()]
    target_lengths = [int(v) for v in target_lengths_t.cpu().tolist()]

    ref_lp = utils.to_reference(log_probs, True)
    ref_targets = utils.to_reference(targets)

    with torch.backends.cudnn.flags(enabled=False):
        ref_out = _ctc_loss_reference(
            ref_lp,
            ref_targets,
            input_lengths,
            target_lengths,
            blank=blank,
            reduction=reduction,
            zero_infinity=False,
        )
    res_out = flag_gems.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction=_int_reduction(reduction),
        zero_infinity=False,
    )
    utils.gems_assert_close(res_out, ref_out, torch.float32, reduce_dim=C, atol=1e-4)


@pytest.mark.ctc_loss
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_ctc_loss_zero_infinity_impossible_alignment_forward_backward(reduction):
    """Cover true infinite-loss path: target is impossible to align with input."""
    T, N, C = 2, 2, 6
    blank = 0
    raw = torch.randn(
        T, N, C, dtype=torch.float32, device=flag_gems.device, requires_grad=True
    )
    log_probs = raw.log_softmax(-1)
    targets = torch.tensor(
        [[1, 2, 3], [1, 1, 2]], dtype=torch.long, device=flag_gems.device
    )
    target_lengths = torch.tensor([3, 3], dtype=torch.long, device=flag_gems.device)
    input_lengths = torch.tensor([2, 2], dtype=torch.long, device=flag_gems.device)

    ref_lp = utils.to_reference(log_probs, True)
    ref_targets = utils.to_reference(targets)
    ref_il = utils.to_reference(input_lengths)
    ref_tl = utils.to_reference(target_lengths)

    with torch.backends.cudnn.flags(enabled=False):
        ref_out = _ctc_loss_reference(
            ref_lp,
            ref_targets,
            ref_il,
            ref_tl,
            blank=blank,
            reduction=reduction,
            zero_infinity=True,
        )
    res_out = flag_gems.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction=_int_reduction(reduction),
        zero_infinity=True,
    )
    utils.gems_assert_close(res_out, ref_out, torch.float32, reduce_dim=C, atol=1e-4)

    out_grad = torch.randn_like(res_out)
    ref_grad = utils.to_reference(out_grad, True)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_lp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, log_probs, out_grad)
    utils.gems_assert_close(
        res_in_grad, ref_in_grad, torch.float32, reduce_dim=C, atol=1e-4, equal_nan=True
    )
    assert torch.isfinite(res_out).all()


@pytest.mark.ctc_loss
@pytest.mark.parametrize("target_layout", ["padded", "concatenated"])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_ctc_loss_repeated_labels(target_layout, reduction):
    """Cover CTC skip-transition rule when adjacent target labels are repeated."""
    T, N, C = 8, 2, 6
    blank = 0
    raw = torch.randn(
        T, N, C, dtype=torch.float32, device=flag_gems.device, requires_grad=True
    )
    log_probs = raw.log_softmax(-1)

    padded = torch.tensor(
        [[1, 1, 2, 0], [2, 3, 3, 1]], dtype=torch.long, device=flag_gems.device
    )
    target_lengths = torch.tensor([3, 4], dtype=torch.long, device=flag_gems.device)
    if target_layout == "padded":
        targets = padded
    else:
        targets = torch.cat([padded[0, :3], padded[1, :4]])
    input_lengths = torch.tensor([T, T - 1], dtype=torch.long, device=flag_gems.device)

    ref_lp = utils.to_reference(log_probs, True)
    ref_targets = utils.to_reference(targets)
    ref_il = utils.to_reference(input_lengths)
    ref_tl = utils.to_reference(target_lengths)

    with torch.backends.cudnn.flags(enabled=False):
        ref_out = _ctc_loss_reference(
            ref_lp,
            ref_targets,
            ref_il,
            ref_tl,
            blank=blank,
            reduction=reduction,
            zero_infinity=False,
        )
    res_out = flag_gems.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction=_int_reduction(reduction),
        zero_infinity=False,
    )
    utils.gems_assert_close(res_out, ref_out, torch.float32, reduce_dim=C, atol=1e-4)

    out_grad = torch.randn_like(res_out)
    ref_grad = utils.to_reference(out_grad, True)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_lp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, log_probs, out_grad)
    utils.gems_assert_close(
        res_in_grad, ref_in_grad, torch.float32, reduce_dim=C, atol=0.02, equal_nan=True
    )


@pytest.mark.ctc_loss
@pytest.mark.parametrize("target_layout", ["padded", "concatenated"])
def test_ctc_loss_all_empty_targets(target_layout):
    """Cover state_count_max == 1 when every target sequence is empty."""
    T, N, C = 6, 3, 5
    blank = 0
    raw = torch.randn(
        T, N, C, dtype=torch.float32, device=flag_gems.device, requires_grad=True
    )
    log_probs = raw.log_softmax(-1)
    target_lengths = torch.zeros(N, dtype=torch.long, device=flag_gems.device)
    if target_layout == "padded":
        targets = torch.zeros(N, 0, dtype=torch.long, device=flag_gems.device)
    else:
        targets = torch.empty(0, dtype=torch.long, device=flag_gems.device)
    input_lengths = torch.full((N,), T, dtype=torch.long, device=flag_gems.device)

    ref_lp = utils.to_reference(log_probs, True)
    ref_targets = utils.to_reference(targets)
    ref_il = utils.to_reference(input_lengths)
    ref_tl = utils.to_reference(target_lengths)

    with torch.backends.cudnn.flags(enabled=False):
        ref_out = _ctc_loss_reference(
            ref_lp,
            ref_targets,
            ref_il,
            ref_tl,
            blank=blank,
            reduction="mean",
            zero_infinity=False,
        )
    res_out = flag_gems.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction=1,
        zero_infinity=False,
    )
    utils.gems_assert_close(res_out, ref_out, torch.float32, reduce_dim=C, atol=1e-4)

    out_grad = torch.randn_like(res_out)
    ref_grad = utils.to_reference(out_grad, True)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_lp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, log_probs, out_grad)
    utils.gems_assert_close(
        res_in_grad, ref_in_grad, torch.float32, reduce_dim=C, atol=0.02, equal_nan=True
    )


@pytest.mark.ctc_loss
@pytest.mark.parametrize("target_len", [0, 2])
@pytest.mark.parametrize("zero_infinity", [False, True])
def test_ctc_loss_zero_input_length(target_len, zero_infinity):
    """Cover input_len == 0 with both empty and non-empty target."""
    T, N, C, S = 3, 1, 5, 2
    blank = 0
    log_probs = torch.randn(
        T, N, C, dtype=torch.float32, device=flag_gems.device, requires_grad=True
    ).log_softmax(-1)
    targets = torch.tensor([[1, 2]], dtype=torch.long, device=flag_gems.device)
    target_lengths = torch.tensor(
        [target_len], dtype=torch.long, device=flag_gems.device
    )
    input_lengths = torch.tensor([0], dtype=torch.long, device=flag_gems.device)

    ref_lp = utils.to_reference(log_probs, True)
    ref_targets = utils.to_reference(targets)
    ref_il = utils.to_reference(input_lengths)
    ref_tl = utils.to_reference(target_lengths)

    with torch.backends.cudnn.flags(enabled=False):
        ref_out = _ctc_loss_reference(
            ref_lp,
            ref_targets,
            ref_il,
            ref_tl,
            blank=blank,
            reduction="mean",
            zero_infinity=zero_infinity,
        )
    res_out = flag_gems.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction=1,
        zero_infinity=zero_infinity,
    )
    utils.gems_assert_close(
        res_out, ref_out, torch.float32, reduce_dim=C, atol=1e-4, equal_nan=True
    )


@pytest.mark.ctc_loss
def test_ctc_loss_unbatched_scalar_lengths():
    """Cover 0-d scalar input_lengths/target_lengths for unbatched input."""
    T, C, S = 8, 7, 3
    blank = 0
    raw = torch.randn(
        T, C, dtype=torch.float32, device=flag_gems.device, requires_grad=True
    )
    log_probs = raw.log_softmax(-1)
    targets = torch.tensor([1, 2, 3], dtype=torch.long, device=flag_gems.device)
    input_lengths = torch.tensor(T, dtype=torch.long, device=flag_gems.device)
    target_lengths = torch.tensor(S, dtype=torch.long, device=flag_gems.device)

    ref_lp = utils.to_reference(log_probs, True)
    ref_targets = utils.to_reference(targets)
    ref_il = utils.to_reference(input_lengths)
    ref_tl = utils.to_reference(target_lengths)

    with torch.backends.cudnn.flags(enabled=False):
        ref_out = _ctc_loss_reference(
            ref_lp,
            ref_targets,
            ref_il,
            ref_tl,
            blank=blank,
            reduction="mean",
            zero_infinity=False,
        )
    res_out = flag_gems.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction=1,
        zero_infinity=False,
    )
    utils.gems_assert_close(res_out, ref_out, torch.float32, reduce_dim=C, atol=1e-4)


@pytest.mark.ctc_loss
@pytest.mark.parametrize(
    "dtype", [dt for dt in FLOAT_DTYPES if dt in (torch.float16, torch.bfloat16)]
)
def test_ctc_loss_low_precision_forward(dtype):
    """Optional low-precision coverage. Reference is computed in fp32 for stability."""
    if dtype == torch.float16 and flag_gems.device == "cpu":
        pytest.skip("float16 CTC is not a useful CPU coverage path")
    if (
        dtype == torch.bfloat16
        and not torch.cuda.is_available()
        and str(flag_gems.device).startswith("cuda")
    ):
        pytest.skip("bfloat16 CUDA device is unavailable")

    T, N, C, S = 8, 2, 8, 3
    blank = 0
    raw = torch.randn(T, N, C, dtype=torch.float32, device=flag_gems.device)
    log_probs_fp32 = raw.log_softmax(-1)
    log_probs = log_probs_fp32.to(dtype)
    targets, target_lengths = _make_targets(N, S, C, flag_gems.device, "padded", blank)
    input_lengths = torch.full((N,), T, dtype=torch.long, device=flag_gems.device)

    ref_lp = utils.to_reference(log_probs_fp32, True)
    ref_targets = utils.to_reference(targets)
    ref_il = utils.to_reference(input_lengths)
    ref_tl = utils.to_reference(target_lengths)

    with torch.backends.cudnn.flags(enabled=False):
        ref_out = _ctc_loss_reference(
            ref_lp,
            ref_targets,
            ref_il,
            ref_tl,
            blank=blank,
            reduction="mean",
            zero_infinity=False,
        )
    res_out = flag_gems.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction=1,
        zero_infinity=False,
    )
    tol = 0.03 if dtype == torch.float16 else 0.05
    utils.gems_assert_close(
        res_out.float(), ref_out.float(), torch.float32, reduce_dim=C, atol=tol
    )

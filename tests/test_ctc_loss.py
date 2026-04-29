import pytest
import torch

import flag_gems
from flag_gems.ops.ctc_loss import _can_use_triton_ctc_loss

from .accuracy_utils import gems_assert_close, to_reference
from .conftest import QUICK_MODE

CTC_LOSS_DTYPES = [torch.float16, torch.float32]
if flag_gems.runtime.device.support_bf16:
    CTC_LOSS_DTYPES.append(torch.bfloat16)
CTC_LOSS_ATOL = {torch.float16: 1e-3, torch.float32: 1e-5, torch.bfloat16: 0.02}


def _make_ctc_targets(
    batch_size, max_target_length, num_classes, device, blank, concatenated=False
):
    target_lengths = torch.randint(
        0, max_target_length + 1, (batch_size,), device=device, dtype=torch.long
    )
    padded = torch.empty(
        (batch_size, max_target_length), device=device, dtype=torch.long
    )
    if max_target_length > 0:
        labels = torch.randint(
            0, num_classes - 1, (batch_size, max_target_length), device=device
        )
        padded.copy_(labels + (labels >= blank).to(labels.dtype))

    if concatenated:
        pieces = [padded[i, : int(target_lengths[i].item())] for i in range(batch_size)]
        if pieces:
            concatenated_targets = torch.cat(pieces, dim=0)
        else:
            concatenated_targets = torch.empty((0,), device=device, dtype=torch.long)
        return concatenated_targets, target_lengths
    return padded, target_lengths


def _assert_ctc_close(res, ref, dtype, equal_nan=False, reduce_dim=1):
    atol = CTC_LOSS_ATOL.get(dtype, 1e-4)
    gems_assert_close(
        res, ref, dtype, equal_nan=equal_nan, reduce_dim=reduce_dim, atol=atol
    )


def _ctc_result_dtype(input_dtype):
    return torch.float32


def _to_reference_ctc_log_probs(log_probs, requires_grad=False):
    ref_log_probs = log_probs if log_probs.dtype == torch.float32 else log_probs.float()
    return to_reference(ref_log_probs, requires_grad)


def _to_reference_ctc_arg(arg, requires_grad=False):
    return to_reference(arg, requires_grad) if torch.is_tensor(arg) else arg


def _assert_ctc_forward_matches(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    *,
    blank,
    reduction,
    zero_infinity,
    dtype,
    reduce_dim,
    equal_nan=True,
):
    ref_out = torch.nn.functional.ctc_loss(
        _to_reference_ctc_log_probs(log_probs),
        _to_reference_ctc_arg(targets),
        _to_reference_ctc_arg(input_lengths),
        _to_reference_ctc_arg(target_lengths),
        blank=blank,
        reduction=reduction,
        zero_infinity=zero_infinity,
    )
    with torch.no_grad():
        with flag_gems.use_gems():
            res_out = torch.nn.functional.ctc_loss(
                log_probs,
                targets,
                input_lengths,
                target_lengths,
                blank=blank,
                reduction=reduction,
                zero_infinity=zero_infinity,
            )
    _assert_ctc_close(
        res_out,
        ref_out,
        _ctc_result_dtype(dtype),
        equal_nan=equal_nan,
        reduce_dim=reduce_dim,
    )


def _assert_ctc_matches(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    *,
    blank,
    reduction,
    zero_infinity,
    dtype,
    reduce_dim,
    check_backward,
    equal_nan=True,
):
    ref_log_probs = _to_reference_ctc_log_probs(log_probs, check_backward)
    ref_targets = _to_reference_ctc_arg(targets)
    ref_input_lengths = _to_reference_ctc_arg(input_lengths)
    ref_target_lengths = _to_reference_ctc_arg(target_lengths)

    ref_out = torch.nn.functional.ctc_loss(
        ref_log_probs,
        ref_targets,
        ref_input_lengths,
        ref_target_lengths,
        blank=blank,
        reduction=reduction,
        zero_infinity=zero_infinity,
    )
    with flag_gems.use_gems():
        res_out = torch.nn.functional.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=blank,
            reduction=reduction,
            zero_infinity=zero_infinity,
        )
    _assert_ctc_close(
        res_out,
        ref_out,
        _ctc_result_dtype(dtype),
        equal_nan=equal_nan,
        reduce_dim=reduce_dim,
    )

    if not check_backward:
        return

    out_grad = torch.randn_like(res_out)
    ref_grad = to_reference(out_grad, True)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_log_probs, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, log_probs, out_grad)
    _assert_ctc_close(
        res_in_grad,
        ref_in_grad,
        dtype,
        equal_nan=equal_nan,
        reduce_dim=log_probs.shape[-1],
    )
    gems_assert_close(
        res_in_grad.sum(dim=-1),
        ref_in_grad.sum(dim=-1),
        dtype,
        equal_nan=equal_nan,
        reduce_dim=log_probs.shape[-1],
        atol=CTC_LOSS_ATOL.get(dtype, 1e-4),
    )


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", CTC_LOSS_DTYPES)
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("concatenated_targets", [False, True])
def test_accuracy_ctc_loss(dtype, reduction, concatenated_targets):
    T, N, C, S = (8, 2, 6, 4) if QUICK_MODE else (16, 3, 12, 6)
    logits = torch.randn((T, N, C), dtype=torch.float32, device=flag_gems.device)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1).to(dtype)
    log_probs.requires_grad_()
    targets, target_lengths = _make_ctc_targets(
        N,
        S,
        C,
        flag_gems.device,
        blank=0,
        concatenated=concatenated_targets,
    )
    input_lengths = torch.full((N,), T, dtype=torch.long, device=flag_gems.device)
    _assert_ctc_matches(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=0,
        reduction=reduction,
        zero_infinity=False,
        dtype=dtype,
        reduce_dim=max(1, int(target_lengths.clamp_min(1).max().item())),
        check_backward=True,
    )


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", CTC_LOSS_DTYPES)
@pytest.mark.parametrize("concatenated_targets", [False, True])
def test_accuracy_ctc_loss_nonzero_blank(dtype, concatenated_targets):
    T, N, C, S = (8, 2, 6, 4) if QUICK_MODE else (16, 3, 12, 6)
    blank = 1
    logits = torch.randn((T, N, C), dtype=torch.float32, device=flag_gems.device)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1).to(dtype)
    targets, target_lengths = _make_ctc_targets(
        N,
        S,
        C,
        flag_gems.device,
        blank=blank,
        concatenated=concatenated_targets,
    )
    input_lengths = torch.full((N,), T, dtype=torch.long, device=flag_gems.device)

    log_probs.requires_grad_()
    _assert_ctc_matches(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction="none",
        zero_infinity=False,
        dtype=dtype,
        reduce_dim=max(1, int(target_lengths.clamp_min(1).max().item())),
        check_backward=True,
        equal_nan=True,
    )


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", CTC_LOSS_DTYPES)
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_accuracy_ctc_loss_intlist(dtype, reduction):
    T, N, C, S = (8, 2, 6, 4) if QUICK_MODE else (16, 3, 12, 6)
    logits = torch.randn((T, N, C), dtype=torch.float32, device=flag_gems.device)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1).to(dtype)
    log_probs.requires_grad_()
    targets, target_lengths = _make_ctc_targets(
        N,
        S,
        C,
        flag_gems.device,
        blank=0,
        concatenated=False,
    )
    input_lengths = [T] * N
    target_lengths_list = target_lengths.tolist()
    _assert_ctc_matches(
        log_probs,
        targets,
        input_lengths,
        target_lengths_list,
        blank=0,
        reduction=reduction,
        zero_infinity=False,
        dtype=dtype,
        reduce_dim=max(1, int(target_lengths.clamp_min(1).max().item())),
        check_backward=True,
    )


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", CTC_LOSS_DTYPES)
def test_accuracy_ctc_loss_zero_infinity(dtype):
    T, N, C = 2, 1, 5
    logits = torch.randn((T, N, C), dtype=torch.float32, device=flag_gems.device)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1).to(dtype)
    log_probs.requires_grad_()
    targets = torch.tensor([[1, 1, 1]], dtype=torch.long, device=flag_gems.device)
    input_lengths = torch.tensor([2], dtype=torch.long, device=flag_gems.device)
    target_lengths = torch.tensor([3], dtype=torch.long, device=flag_gems.device)
    _assert_ctc_matches(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=0,
        reduction="mean",
        zero_infinity=True,
        dtype=dtype,
        reduce_dim=1,
        check_backward=True,
        equal_nan=True,
    )


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", CTC_LOSS_DTYPES)
def test_accuracy_ctc_loss_unbatched(dtype):
    T, C, S = 9, 7, 4
    blank = 1
    logits = torch.randn((T, C), dtype=torch.float32, device=flag_gems.device)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1).to(dtype)
    log_probs.requires_grad_()
    labels = torch.randint(0, C - 1, (S,), device=flag_gems.device)
    targets = labels + (labels >= blank).to(labels.dtype)
    input_lengths = torch.tensor(T, dtype=torch.long, device=flag_gems.device)
    target_lengths = torch.tensor(S, dtype=torch.long, device=flag_gems.device)
    _assert_ctc_matches(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction="none",
        zero_infinity=False,
        dtype=dtype,
        reduce_dim=S,
        check_backward=True,
    )


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", CTC_LOSS_DTYPES)
@pytest.mark.parametrize("concatenated_targets", [False, True])
def test_accuracy_ctc_loss_repeated_labels_backward(dtype, concatenated_targets):
    T, N, C = 5, 1, 4
    blank = 0
    logits = torch.randn((T, N, C), dtype=torch.float32, device=flag_gems.device)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1).to(dtype)
    log_probs.requires_grad_()
    padded_targets = torch.tensor([[1, 1]], dtype=torch.long, device=flag_gems.device)
    targets = padded_targets[0, :2].clone() if concatenated_targets else padded_targets
    input_lengths = torch.tensor([T], dtype=torch.long, device=flag_gems.device)
    target_lengths = torch.tensor([2], dtype=torch.long, device=flag_gems.device)

    _assert_ctc_matches(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction="none",
        zero_infinity=False,
        dtype=dtype,
        reduce_dim=2,
        check_backward=True,
        equal_nan=True,
    )


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", CTC_LOSS_DTYPES)
@pytest.mark.parametrize("concatenated_targets", [False, True])
def test_accuracy_ctc_loss_impossible_alignment_backward(dtype, concatenated_targets):
    T, N, C, S = 2, 1, 5, 3
    logits = torch.randn((T, N, C), dtype=torch.float32, device=flag_gems.device)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1).to(dtype)
    log_probs.requires_grad_()
    padded_targets = torch.tensor(
        [[1, 2, 3]], dtype=torch.long, device=flag_gems.device
    )
    targets = padded_targets[0, :S].clone() if concatenated_targets else padded_targets
    input_lengths = torch.tensor([T], dtype=torch.long, device=flag_gems.device)
    target_lengths = torch.tensor([S], dtype=torch.long, device=flag_gems.device)

    _assert_ctc_matches(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=0,
        reduction="none",
        zero_infinity=False,
        dtype=dtype,
        reduce_dim=S,
        check_backward=True,
        equal_nan=True,
    )


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", CTC_LOSS_DTYPES)
def test_accuracy_ctc_loss_triton_forward_path(dtype):
    T, N, C, S = 16, 3, 12, 6
    logits = torch.randn((T, N, C), dtype=torch.float32, device=flag_gems.device)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1).to(dtype)
    targets, target_lengths = _make_ctc_targets(
        N,
        S,
        C,
        flag_gems.device,
        blank=0,
        concatenated=False,
    )
    input_lengths = torch.full((N,), T, dtype=torch.long, device=flag_gems.device)
    assert _can_use_triton_ctc_loss(
        log_probs, targets, input_lengths, target_lengths, blank=0
    )

    _assert_ctc_forward_matches(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=0,
        reduction="none",
        zero_infinity=False,
        dtype=dtype,
        equal_nan=True,
        reduce_dim=max(1, int(target_lengths.clamp_min(1).max().item())),
    )


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", CTC_LOSS_DTYPES)
def test_accuracy_ctc_loss_triton_forward_path_concatenated(dtype):
    T, N, C, S = 16, 3, 12, 6
    logits = torch.randn((T, N, C), dtype=torch.float32, device=flag_gems.device)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1).to(dtype)
    targets, target_lengths = _make_ctc_targets(
        N,
        S,
        C,
        flag_gems.device,
        blank=0,
        concatenated=True,
    )
    input_lengths = torch.full((N,), T, dtype=torch.long, device=flag_gems.device)
    assert _can_use_triton_ctc_loss(
        log_probs, targets, input_lengths, target_lengths, blank=0
    )

    _assert_ctc_forward_matches(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=0,
        reduction="none",
        zero_infinity=False,
        dtype=dtype,
        equal_nan=True,
        reduce_dim=max(1, int(target_lengths.clamp_min(1).max().item())),
    )


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", CTC_LOSS_DTYPES)
def test_accuracy_ctc_loss_triton_forward_path_unbatched(dtype):
    T, C, S = 16, 12, 6
    logits = torch.randn((T, C), dtype=torch.float32, device=flag_gems.device)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1).to(dtype)
    labels = torch.randint(0, C - 1, (S,), device=flag_gems.device)
    blank = 1
    targets = labels + (labels >= blank).to(labels.dtype)
    input_lengths = torch.tensor(T, dtype=torch.long, device=flag_gems.device)
    target_lengths = torch.tensor(S, dtype=torch.long, device=flag_gems.device)
    assert _can_use_triton_ctc_loss(
        log_probs, targets, input_lengths, target_lengths, blank=blank
    )

    _assert_ctc_forward_matches(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction="none",
        zero_infinity=False,
        dtype=dtype,
        equal_nan=True,
        reduce_dim=max(1, int(target_lengths.clamp_min(1).item())),
    )


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", CTC_LOSS_DTYPES)
def test_accuracy_ctc_loss_triton_forward_path_empty_targets(dtype):
    T, N, C = 12, 2, 7
    logits = torch.randn((T, N, C), dtype=torch.float32, device=flag_gems.device)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1).to(dtype)
    targets = torch.empty((N, 0), dtype=torch.long, device=flag_gems.device)
    input_lengths = torch.full((N,), T, dtype=torch.long, device=flag_gems.device)
    target_lengths = torch.zeros((N,), dtype=torch.long, device=flag_gems.device)
    assert _can_use_triton_ctc_loss(
        log_probs, targets, input_lengths, target_lengths, blank=0
    )

    _assert_ctc_forward_matches(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=0,
        reduction="none",
        zero_infinity=False,
        dtype=dtype,
        equal_nan=True,
        reduce_dim=1,
    )


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", CTC_LOSS_DTYPES)
def test_accuracy_ctc_loss_triton_forward_path_noncontiguous(dtype):
    T, N, C, S = 16, 3, 12, 6
    logits = torch.randn((N, T, C), dtype=torch.float32, device=flag_gems.device)
    log_probs = (
        torch.nn.functional.log_softmax(logits, dim=-1).to(dtype).permute(1, 0, 2)
    )
    assert not log_probs.is_contiguous()
    targets, target_lengths = _make_ctc_targets(
        N,
        S,
        C,
        flag_gems.device,
        blank=0,
        concatenated=False,
    )
    input_lengths = torch.full((N,), T, dtype=torch.long, device=flag_gems.device)
    assert _can_use_triton_ctc_loss(
        log_probs, targets, input_lengths, target_lengths, blank=0
    )

    _assert_ctc_forward_matches(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=0,
        reduction="none",
        zero_infinity=False,
        dtype=dtype,
        equal_nan=True,
        reduce_dim=max(1, int(target_lengths.clamp_min(1).max().item())),
    )


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", CTC_LOSS_DTYPES)
def test_accuracy_ctc_loss_triton_forward_path_negative_infinity(dtype):
    T, N, C = 10, 2, 6
    logits = torch.randn((T, N, C), dtype=torch.float32, device=flag_gems.device)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1).to(dtype)
    log_probs[3, 0, 2] = float("-inf")
    log_probs[5, 1, 4] = float("-inf")
    targets = torch.tensor(
        [[1, 2, 3], [2, 3, 1]], dtype=torch.long, device=flag_gems.device
    )
    input_lengths = torch.full((N,), T, dtype=torch.long, device=flag_gems.device)
    target_lengths = torch.tensor([3, 3], dtype=torch.long, device=flag_gems.device)
    assert _can_use_triton_ctc_loss(
        log_probs, targets, input_lengths, target_lengths, blank=0
    )

    _assert_ctc_forward_matches(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=0,
        reduction="none",
        zero_infinity=False,
        dtype=dtype,
        equal_nan=True,
        reduce_dim=3,
    )


@pytest.mark.ctc_loss
def test_accuracy_ctc_loss_invalid_concatenated_targets():
    log_probs = torch.randn((6, 2, 5), dtype=torch.float32, device=flag_gems.device)
    log_probs = torch.nn.functional.log_softmax(log_probs, dim=-1)
    targets = torch.tensor([1, 2, 3], dtype=torch.long, device=flag_gems.device)
    input_lengths = torch.tensor([6, 6], dtype=torch.long, device=flag_gems.device)
    target_lengths = torch.tensor([2, 2], dtype=torch.long, device=flag_gems.device)

    ref_log_probs = to_reference(log_probs, True)
    ref_targets = to_reference(targets)
    ref_input_lengths = to_reference(input_lengths)
    ref_target_lengths = to_reference(target_lengths)

    with pytest.raises(RuntimeError):
        torch.nn.functional.ctc_loss(
            ref_log_probs, ref_targets, ref_input_lengths, ref_target_lengths
        )
    with flag_gems.use_gems():
        with pytest.raises(RuntimeError, match="ctc_loss_allocate_outputs"):
            torch.nn.functional.ctc_loss(
                log_probs, targets, input_lengths, target_lengths
            )

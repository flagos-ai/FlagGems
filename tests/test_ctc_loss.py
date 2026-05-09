import pytest
import torch
import torch.nn.functional as F

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

CTC_DTYPES = [torch.float32] if cfg.QUICK_MODE else [torch.float32, torch.float16]
REDUCTIONS = ["none", "mean"] if cfg.QUICK_MODE else ["none", "mean", "sum"]
TARGET_LAYOUTS = ["padded", "concatenated"]


def _make_log_probs(shape, dtype, *, noncontiguous=False):
    if len(shape) == 2:
        data = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
        result = data.log_softmax(-1)
    elif noncontiguous:
        t_steps, batch, classes = shape
        data = torch.randn(
            (batch, t_steps, classes),
            dtype=torch.float32,
            device=flag_gems.device,
        )
        result = data.log_softmax(-1).transpose(0, 1)
    else:
        data = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
        result = data.log_softmax(-1)
    return result.to(dtype).detach().requires_grad_()


def _make_targets(batch, max_target, classes, blank, layout, *, repeated=False):
    target_lengths = torch.empty(batch, device=flag_gems.device, dtype=torch.long)
    padded = torch.zeros(batch, max_target, device=flag_gems.device, dtype=torch.long)
    pieces = []
    values = [x for x in range(classes) if x != blank]
    for row in range(batch):
        length = max(1, max_target - row)
        target_lengths[row] = length
        if repeated:
            row_values = [values[row % len(values)]] * min(2, length)
            row_values += [
                values[(row + col + 1) % len(values)]
                for col in range(length - len(row_values))
            ]
        else:
            row_values = [values[(row + col) % len(values)] for col in range(length)]
        current = torch.tensor(row_values, device=flag_gems.device, dtype=torch.long)
        padded[row, :length] = current
        pieces.append(current)
    if layout == "padded":
        return padded, target_lengths
    return torch.cat(pieces), target_lengths


def _targets_from_rows(rows, layout, *, max_target=None):
    if max_target is None:
        max_target = max((len(row) for row in rows), default=0)
    target_lengths = torch.tensor(
        [len(row) for row in rows],
        device=flag_gems.device,
        dtype=torch.long,
    )
    padded = torch.zeros(
        len(rows), max_target, device=flag_gems.device, dtype=torch.long
    )
    pieces = []
    for row_index, row_values in enumerate(rows):
        current = torch.tensor(row_values, device=flag_gems.device, dtype=torch.long)
        if len(row_values) > 0:
            padded[row_index, : len(row_values)] = current
        pieces.append(current)
    if layout == "padded":
        return padded, target_lengths
    return torch.cat(pieces), target_lengths


def _reference(log_probs, targets, input_lengths, target_lengths, **kwargs):
    ref_log_probs = utils.to_reference(log_probs.detach(), False).to(torch.float32)
    ref_log_probs.requires_grad_(log_probs.requires_grad)
    ref_out = F.ctc_loss(
        ref_log_probs,
        utils.to_reference(targets),
        utils.to_reference(input_lengths)
        if torch.is_tensor(input_lengths)
        else input_lengths,
        utils.to_reference(target_lengths)
        if torch.is_tensor(target_lengths)
        else target_lengths,
        **kwargs,
    )
    return ref_log_probs, ref_out.to(log_probs.dtype)


def _assert_forward_backward(
    log_probs, targets, input_lengths, target_lengths, dtype, **kwargs
):
    ref_log_probs, ref_out = _reference(
        log_probs, targets, input_lengths, target_lengths, **kwargs
    )
    res_out = flag_gems.ctc_loss(
        log_probs, targets, input_lengths, target_lengths, **kwargs
    )
    reduce_dim = max(
        1,
        log_probs.shape[0] * int(torch.as_tensor(target_lengths).max().item() + 1),
    )
    utils.gems_assert_close(
        res_out, ref_out, dtype, equal_nan=True, reduce_dim=reduce_dim
    )

    grad = torch.randn_like(res_out)
    ref_grad_out = utils.to_reference(grad, False).to(ref_out.dtype)
    (ref_grad,) = torch.autograd.grad(ref_out, ref_log_probs, ref_grad_out)
    (res_grad,) = torch.autograd.grad(res_out, log_probs, grad)
    utils.gems_assert_close(
        res_grad, ref_grad, dtype, equal_nan=True, reduce_dim=reduce_dim
    )


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", CTC_DTYPES)
@pytest.mark.parametrize("layout", TARGET_LAYOUTS)
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_ctc_loss_matches_pytorch(dtype, layout, reduction):
    utils.init_seed(20260505)
    t_steps, batch, classes, max_target = (12, 3, 7, 4)
    log_probs = _make_log_probs((t_steps, batch, classes), dtype)
    targets, target_lengths = _make_targets(batch, max_target, classes, 0, layout)
    input_lengths = torch.tensor([12, 10, 9], device=flag_gems.device, dtype=torch.long)
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


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", CTC_DTYPES)
def test_ctc_loss_nonzero_blank_repeated_and_zero_infinity(dtype):
    utils.init_seed(17)
    log_probs = _make_log_probs((9, 2, 6), dtype, noncontiguous=True)
    targets, target_lengths = _make_targets(2, 4, 6, 2, "padded", repeated=True)
    input_lengths = torch.tensor([2, 2], device=flag_gems.device, dtype=torch.long)
    _assert_forward_backward(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        dtype,
        blank=2,
        reduction="none",
        zero_infinity=True,
    )


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", CTC_DTYPES)
@pytest.mark.parametrize("layout", TARGET_LAYOUTS)
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_ctc_loss_zero_length_targets_match_pytorch(dtype, layout, reduction):
    utils.init_seed(2026050601)
    log_probs = _make_log_probs((8, 3, 6), dtype, noncontiguous=True)
    targets, target_lengths = _targets_from_rows([[], [1, 4], []], layout, max_target=3)
    input_lengths = torch.tensor([8, 7, 5], device=flag_gems.device, dtype=torch.long)
    _assert_forward_backward(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        dtype,
        blank=3,
        reduction=reduction,
        zero_infinity=False,
    )


@pytest.mark.ctc_loss
@pytest.mark.parametrize("layout", TARGET_LAYOUTS)
def test_ctc_loss_all_empty_targets_match_pytorch(layout):
    utils.init_seed(2026050602)
    log_probs = _make_log_probs((5, 2, 4), torch.float32)
    targets, target_lengths = _targets_from_rows([[], []], layout, max_target=0)
    input_lengths = torch.tensor([5, 3], device=flag_gems.device, dtype=torch.long)
    _assert_forward_backward(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        torch.float32,
        blank=2,
        reduction="mean",
        zero_infinity=False,
    )


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", CTC_DTYPES)
@pytest.mark.parametrize("layout", TARGET_LAYOUTS)
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_ctc_loss_repeated_labels_nonzero_blank_reductions(dtype, layout, reduction):
    utils.init_seed(2026050603)
    log_probs = _make_log_probs((11, 2, 7), dtype)
    targets, target_lengths = _targets_from_rows([[1, 1, 2, 2], [4, 4, 5]], layout)
    input_lengths = torch.tensor([11, 9], device=flag_gems.device, dtype=torch.long)
    _assert_forward_backward(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        dtype,
        blank=6,
        reduction=reduction,
        zero_infinity=False,
    )


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", CTC_DTYPES)
@pytest.mark.parametrize("layout", TARGET_LAYOUTS)
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_ctc_loss_zero_infinity_reductions_impossible_targets(dtype, layout, reduction):
    utils.init_seed(2026050604)
    log_probs = _make_log_probs((7, 2, 6), dtype, noncontiguous=True)
    targets, target_lengths = _targets_from_rows([[1, 1, 2, 2], [3, 3, 1]], layout)
    input_lengths = torch.tensor([3, 2], device=flag_gems.device, dtype=torch.long)
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


@pytest.mark.ctc_loss
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_ctc_loss_padded_and_concatenated_targets_equivalent(reduction):
    utils.init_seed(2026050605)
    rows = [[], [1, 1, 4], [2, 5]]
    input_lengths = torch.tensor([9, 8, 7], device=flag_gems.device, dtype=torch.long)
    padded, target_lengths = _targets_from_rows(rows, "padded", max_target=4)
    concatenated, _ = _targets_from_rows(rows, "concatenated", max_target=4)
    padded_log_probs = _make_log_probs((9, 3, 7), torch.float32)
    concatenated_log_probs = padded_log_probs.detach().clone().requires_grad_()

    padded_out = flag_gems.ctc_loss(
        padded_log_probs,
        padded,
        input_lengths,
        target_lengths,
        blank=3,
        reduction=reduction,
        zero_infinity=False,
    )
    concatenated_out = flag_gems.ctc_loss(
        concatenated_log_probs,
        concatenated,
        input_lengths,
        target_lengths,
        blank=3,
        reduction=reduction,
        zero_infinity=False,
    )
    flag_gems.testing.assert_close(
        padded_out, concatenated_out, torch.float32, reduce_dim=9
    )

    grad = torch.randn_like(padded_out)
    (padded_grad,) = torch.autograd.grad(padded_out, padded_log_probs, grad)
    (concatenated_grad,) = torch.autograd.grad(
        concatenated_out, concatenated_log_probs, grad
    )
    flag_gems.testing.assert_close(
        padded_grad, concatenated_grad, torch.float32, reduce_dim=36
    )


@pytest.mark.ctc_loss
def test_ctc_loss_unbatched_and_intlist_registered_paths():
    utils.init_seed(23)
    log_probs = _make_log_probs((10, 6), torch.float32)
    targets = torch.tensor([1, 3, 4], device=flag_gems.device, dtype=torch.long)
    out = flag_gems.ctc_loss(log_probs, targets, [10], [3], reduction="sum")
    ref_log_probs, ref = _reference(
        log_probs,
        targets,
        [10],
        [3],
        blank=0,
        reduction="sum",
        zero_infinity=False,
    )
    utils.gems_assert_close(out, ref, torch.float32, reduce_dim=10)
    torch.autograd.grad(out, log_probs, torch.ones_like(out))
    torch.autograd.grad(ref, ref_log_probs, torch.ones_like(ref))

    batched = _make_log_probs((8, 2, 6), torch.float32)
    padded, target_lengths = _make_targets(2, 3, 6, 0, "padded")
    with flag_gems.use_gems(include=["ctc_loss"]):
        registered = F.ctc_loss(
            batched,
            padded,
            [8, 7],
            target_lengths.tolist(),
            reduction="mean",
        )
    _, expected = _reference(
        batched,
        padded,
        [8, 7],
        target_lengths.tolist(),
        blank=0,
        reduction="mean",
        zero_infinity=False,
    )
    utils.gems_assert_close(registered, expected, torch.float32, reduce_dim=8)


@pytest.mark.ctc_loss
def test_ctc_loss_registered_tensor_backward_path():
    utils.init_seed(29)
    log_probs = _make_log_probs((9, 2, 6), torch.float32)
    targets, target_lengths = _make_targets(2, 3, 6, 0, "concatenated")
    input_lengths = torch.tensor([9, 8], device=flag_gems.device, dtype=torch.long)

    ref_log_probs, ref_out = _reference(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=0,
        reduction="mean",
        zero_infinity=False,
    )

    with flag_gems.use_gems(include=["ctc_loss"]):
        assert "ctc_loss.Tensor" in flag_gems.all_registered_keys()
        res_out = F.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            reduction="mean",
        )

    assert res_out.grad_fn.__class__.__name__ == "_CtcLossBackward"
    grad = torch.ones_like(res_out)
    (ref_grad,) = torch.autograd.grad(ref_out, ref_log_probs, grad.to(ref_out.dtype))
    (res_grad,) = torch.autograd.grad(res_out, log_probs, grad)

    utils.gems_assert_close(res_out, ref_out, torch.float32, reduce_dim=9)
    utils.gems_assert_close(res_grad, ref_grad, torch.float32, reduce_dim=9)


@pytest.mark.ctc_loss
def test_ctc_loss_invalid_lengths_raise():
    log_probs = _make_log_probs((6, 2, 5), torch.float32)
    targets = torch.tensor([[1, 2], [2, 3]], device=flag_gems.device, dtype=torch.long)
    target_lengths = torch.tensor([2, 2], device=flag_gems.device, dtype=torch.long)
    bad_input_lengths = torch.tensor([6.0, 6.0], device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.ctc_loss(log_probs, targets, bad_input_lengths, target_lengths)

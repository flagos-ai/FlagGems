import pytest
import torch
import torch.nn.functional as F

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    REDUCTIONS = ["mean", "none"]
    ZERO_INFINITY = [False, True]
else:
    FLOAT_DTYPES = [torch.float32, torch.float16]
    REDUCTIONS = ["none", "mean", "sum"]
    ZERO_INFINITY = [False, True]

TARGET_LAYOUTS = ["padded", "concatenated"]


def _nonblank_values(classes, blank):
    return [value for value in range(classes) if value != blank]


def _make_log_probs(shape, dtype, noncontiguous=False, has_neginf=False):
    if len(shape) == 2:
        raw = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
        log_probs = raw.log_softmax(-1)
    elif noncontiguous:
        t_steps, batch, classes = shape
        raw = torch.randn(
            (batch, t_steps, classes), dtype=torch.float32, device=flag_gems.device
        )
        log_probs = raw.log_softmax(-1).transpose(0, 1)
        assert not log_probs.is_contiguous()
    else:
        raw = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
        log_probs = raw.log_softmax(-1)

    if has_neginf:
        log_probs = log_probs.clone()
        log_probs[0, ..., -1] = -float("inf")

    return log_probs.to(dtype).detach().requires_grad_()


def _make_targets(batch, max_target, classes, blank, pattern, target_layout):
    values = _nonblank_values(classes, blank)
    rows = []
    if pattern == "normal":
        lengths = torch.tensor([max_target, max_target - 1, 2][:batch])
        for row in range(batch):
            rows.append(
                torch.tensor(
                    [values[(row + col) % len(values)] for col in range(max_target)],
                    device=flag_gems.device,
                    dtype=torch.long,
                )
            )
    elif pattern == "repeated":
        lengths = torch.tensor([max_target, max_target - 1][:batch])
        for row in range(batch):
            label_a = values[row % len(values)]
            label_b = values[(row + 2) % len(values)]
            rows.append(
                torch.tensor(
                    [label_a, label_a, label_b, label_b][:max_target],
                    device=flag_gems.device,
                    dtype=torch.long,
                )
            )
    elif pattern == "empty":
        lengths = torch.tensor([0, max_target - 1, 1][:batch])
        for row in range(batch):
            rows.append(
                torch.tensor(
                    [values[(row + col) % len(values)] for col in range(max_target)],
                    device=flag_gems.device,
                    dtype=torch.long,
                )
            )
    elif pattern == "impossible":
        lengths = torch.tensor([max_target, max_target - 1][:batch])
        for row in range(batch):
            rows.append(
                torch.tensor(
                    [values[(row + col) % len(values)] for col in range(max_target)],
                    device=flag_gems.device,
                    dtype=torch.long,
                )
            )
    elif pattern == "repeated_impossible":
        lengths = torch.tensor([2, 3][:batch])
        for row in range(batch):
            label = values[row % len(values)]
            other = values[(row + 1) % len(values)]
            rows.append(
                torch.tensor(
                    [label, label, other, other][:max_target],
                    device=flag_gems.device,
                    dtype=torch.long,
                )
            )
    else:
        raise ValueError(f"unknown CTC target pattern: {pattern}")

    target_lengths = lengths.to(device=flag_gems.device, dtype=torch.long)
    padded = torch.zeros(batch, max_target, device=flag_gems.device, dtype=torch.long)
    pieces = []
    for row, length in enumerate(target_lengths.tolist()):
        padded[row, :length] = rows[row][:length]
        pieces.append(rows[row][:length])

    if target_layout == "padded":
        targets = padded
    else:
        targets = torch.cat(pieces) if pieces else padded.new_empty((0,))
    return targets, target_lengths


def _reference_ctc_loss(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    blank,
    reduction,
    zero_infinity,
):
    ref_log_probs = utils.to_reference(log_probs.detach(), False).to(torch.float32)
    ref_log_probs.requires_grad_(True)
    ref_targets = utils.to_reference(targets)
    ref_input_lengths = utils.to_reference(input_lengths)
    ref_target_lengths = utils.to_reference(target_lengths)
    ref_out = F.ctc_loss(
        ref_log_probs,
        ref_targets,
        ref_input_lengths,
        ref_target_lengths,
        blank=blank,
        reduction=reduction,
        zero_infinity=zero_infinity,
    )
    return ref_log_probs, ref_out.to(log_probs.dtype)


def _assert_forward_backward(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    dtype,
    blank=0,
    reduction="mean",
    zero_infinity=False,
    equal_nan=False,
):
    ref_log_probs, ref_out = _reference_ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank,
        reduction,
        zero_infinity,
    )
    res_out = flag_gems.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction=reduction,
        zero_infinity=zero_infinity,
    )

    reduce_dim = max(1, log_probs.shape[0] * int(target_lengths.max().item() + 1))
    utils.gems_assert_close(
        res_out, ref_out, dtype, equal_nan=equal_nan, reduce_dim=reduce_dim
    )

    out_grad = torch.randn_like(res_out)
    ref_grad_out = utils.to_reference(out_grad, False).to(ref_out.dtype)
    (ref_grad,) = torch.autograd.grad(ref_out, ref_log_probs, ref_grad_out)
    (res_grad,) = torch.autograd.grad(res_out, log_probs, out_grad)

    utils.gems_assert_close(
        res_grad,
        ref_grad,
        dtype,
        equal_nan=equal_nan,
        reduce_dim=reduce_dim,
    )


def _call_ctc_loss(
    path,
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    **kwargs,
):
    if path == "direct":
        return flag_gems.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            **kwargs,
        )
    if path == "registered":
        with flag_gems.use_gems(include=["ctc_loss"]):
            return F.ctc_loss(
                log_probs,
                targets,
                input_lengths,
                target_lengths,
                **kwargs,
            )
    raise ValueError(f"unknown CTC call path: {path}")


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("target_layout", TARGET_LAYOUTS)
@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize("zero_infinity", ZERO_INFINITY)
def test_ctc_loss_core_matrix(dtype, target_layout, reduction, zero_infinity):
    utils.init_seed(2026)
    shape = (12, 3, 7, 4)
    t_steps, batch, classes, max_target = shape
    blank = 0 if target_layout == "padded" else 2
    log_probs = _make_log_probs((t_steps, batch, classes), dtype)
    targets, target_lengths = _make_targets(
        batch, max_target, classes, blank, "normal", target_layout
    )
    input_lengths = torch.tensor([12, 10, 9], device=flag_gems.device, dtype=torch.long)

    _assert_forward_backward(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        dtype,
        blank=blank,
        reduction=reduction,
        zero_infinity=zero_infinity,
    )


EDGE_CASES = [
    ("repeated", "concatenated", "none", False, 0, False, False),
    ("empty", "padded", "sum", True, 0, False, False),
    ("impossible", "concatenated", "mean", True, 0, False, False),
    ("repeated_impossible", "padded", "none", False, 0, True, True),
    ("normal", "padded", "mean", False, 1, True, True),
]

if cfg.QUICK_MODE:
    EDGE_CASES = EDGE_CASES[:2]


@pytest.mark.ctc_loss
@pytest.mark.parametrize(
    (
        "pattern",
        "target_layout",
        "reduction",
        "zero_infinity",
        "blank",
        "noncontiguous",
        "equal_nan",
    ),
    EDGE_CASES,
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_ctc_loss_edge_cases(
    dtype,
    pattern,
    target_layout,
    reduction,
    zero_infinity,
    blank,
    noncontiguous,
    equal_nan,
):
    utils.init_seed(123)
    t_steps, batch, classes, max_target = (10, 2, 6, 4)
    if pattern in ("impossible", "repeated_impossible"):
        input_lengths = torch.tensor([3, 2], device=flag_gems.device, dtype=torch.long)
    else:
        input_lengths = torch.tensor([10, 8], device=flag_gems.device, dtype=torch.long)

    log_probs = _make_log_probs(
        (t_steps, batch, classes),
        dtype,
        noncontiguous=noncontiguous,
        has_neginf=pattern == "normal",
    )
    targets, target_lengths = _make_targets(
        batch, max_target, classes, blank, pattern, target_layout
    )

    _assert_forward_backward(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        dtype,
        blank=blank,
        reduction=reduction,
        zero_infinity=zero_infinity,
        equal_nan=equal_nan,
    )


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_ctc_loss_unbatched(dtype, reduction):
    utils.init_seed(77)
    t_steps, classes, target_length = (9, 6, 4)
    log_probs = _make_log_probs((t_steps, classes), dtype)
    targets = torch.tensor([1, 3, 4, 5], device=flag_gems.device, dtype=torch.long)
    input_lengths = torch.tensor(t_steps, device=flag_gems.device, dtype=torch.long)
    target_lengths = torch.tensor(
        target_length, device=flag_gems.device, dtype=torch.long
    )

    _assert_forward_backward(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        dtype,
        reduction=reduction,
    )


@pytest.mark.ctc_loss
def test_ctc_loss_registered_intlist_forward():
    utils.init_seed(99)
    t_steps, batch, classes, max_target = (8, 2, 6, 3)
    log_probs = _make_log_probs((t_steps, batch, classes), torch.float32)
    targets, target_lengths = _make_targets(
        batch, max_target, classes, 0, "normal", "padded"
    )
    input_lengths = [8, 7]
    target_lengths_list = target_lengths.tolist()

    ref_out = F.ctc_loss(
        utils.to_reference(log_probs.detach(), False).to(torch.float32),
        utils.to_reference(targets),
        input_lengths,
        target_lengths_list,
        reduction="sum",
    )
    with flag_gems.use_gems(include=["ctc_loss"]):
        res_out = F.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths_list,
            reduction="sum",
        )

    utils.gems_assert_close(res_out, ref_out, torch.float32, reduce_dim=t_steps)


@pytest.mark.ctc_loss
def test_ctc_loss_registered_tensor_backward():
    utils.init_seed(101)
    t_steps, batch, classes, max_target = (9, 2, 6, 3)
    log_probs = _make_log_probs((t_steps, batch, classes), torch.float32)
    targets, target_lengths = _make_targets(
        batch, max_target, classes, 0, "repeated", "concatenated"
    )
    input_lengths = torch.tensor([9, 8], device=flag_gems.device, dtype=torch.long)

    ref_log_probs = utils.to_reference(log_probs.detach(), False).to(torch.float32)
    ref_log_probs.requires_grad_(True)
    ref_out = F.ctc_loss(
        ref_log_probs,
        utils.to_reference(targets),
        utils.to_reference(input_lengths),
        utils.to_reference(target_lengths),
        reduction="mean",
    )
    with flag_gems.use_gems(include=["ctc_loss"]):
        res_out = F.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            reduction="mean",
        )

    out_grad = torch.ones_like(res_out)
    (ref_grad,) = torch.autograd.grad(
        ref_out, ref_log_probs, utils.to_reference(out_grad)
    )
    (res_grad,) = torch.autograd.grad(res_out, log_probs, out_grad)

    utils.gems_assert_close(res_out, ref_out, torch.float32, reduce_dim=t_steps)
    utils.gems_assert_close(res_grad, ref_grad, torch.float32, reduce_dim=t_steps)


@pytest.mark.ctc_loss
@pytest.mark.parametrize("target_layout", TARGET_LAYOUTS)
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_ctc_loss_forward_no_grad_path(target_layout, reduction):
    utils.init_seed(303)
    t_steps, batch, classes, max_target = (11, 3, 7, 5)
    log_probs = _make_log_probs((t_steps, batch, classes), torch.float32).detach()
    targets, target_lengths = _make_targets(
        batch, max_target, classes, 0, "empty", target_layout
    )
    input_lengths = torch.tensor([11, 8, 6], device=flag_gems.device, dtype=torch.long)

    _, ref_out = _reference_ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        0,
        reduction,
        False,
    )
    res_out = flag_gems.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        reduction=reduction,
    )

    utils.gems_assert_close(
        res_out,
        ref_out,
        torch.float32,
        reduce_dim=t_steps * int(target_lengths.max().item() + 1),
    )


@pytest.mark.ctc_loss
def test_ctc_loss_no_grad_large_concatenated_variable_lengths_regression():
    utils.init_seed(2026)
    t_steps, batch, classes, max_target = (256, 16, 64, 48)
    log_probs = _make_log_probs((t_steps, batch, classes), torch.float32).detach()
    input_lengths = torch.tensor(
        [t_steps - row for row in range(batch)],
        device=flag_gems.device,
        dtype=torch.long,
    )

    target_lengths = torch.empty(batch, device=flag_gems.device, dtype=torch.long)
    pieces = []
    values = _nonblank_values(classes, 0)
    for row in range(batch):
        length = max(1, max_target - (row % 5))
        target_lengths[row] = length
        pieces.append(
            torch.tensor(
                [values[(row + col) % len(values)] for col in range(length)],
                device=flag_gems.device,
                dtype=torch.long,
            )
        )
    targets = torch.cat(pieces)

    _, ref_out = _reference_ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        0,
        "none",
        False,
    )
    res_out = flag_gems.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        reduction="none",
    )

    utils.gems_assert_close(
        res_out,
        ref_out,
        torch.float32,
        reduce_dim=t_steps * int(target_lengths.max().item() + 1),
    )


@pytest.mark.ctc_loss
def test_ctc_loss_no_grad_padded_noncontiguous_blank_zero_infinity_none():
    utils.init_seed(404)
    t_steps, batch, classes, max_target = (13, 3, 8, 5)
    blank = 2
    log_probs = _make_log_probs(
        (t_steps, batch, classes), torch.float32, noncontiguous=True
    ).detach()
    targets, target_lengths = _make_targets(
        batch, max_target, classes, blank, "normal", "padded"
    )
    input_lengths = torch.tensor([3, 2, 1], device=flag_gems.device, dtype=torch.long)

    _, ref_out = _reference_ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank,
        "none",
        True,
    )
    res_out = flag_gems.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction="none",
        zero_infinity=True,
    )

    utils.gems_assert_close(
        res_out,
        ref_out,
        torch.float32,
        reduce_dim=t_steps * int(target_lengths.max().item() + 1),
    )


@pytest.mark.ctc_loss
@pytest.mark.parametrize(
    ("input_lengths", "target_count"),
    [
        ([65, 64, 64, 64], 58),
        ([64, 64, 64, 64], 57),
    ],
)
def test_ctc_loss_invalid_small_shape_inputs_raise(input_lengths, target_count):
    log_probs = _make_log_probs((64, 4, 32), torch.float32).detach()
    target_lengths = torch.tensor(
        [16, 15, 14, 13], device=flag_gems.device, dtype=torch.long
    )
    targets = (
        torch.arange(target_count, device=flag_gems.device, dtype=torch.long) % 31
    ) + 1
    input_lengths = torch.tensor(
        input_lengths, device=flag_gems.device, dtype=torch.long
    )

    with pytest.raises(RuntimeError):
        flag_gems.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            reduction="mean",
        )


@pytest.mark.ctc_loss
def test_ctc_loss_length_stats_cache_observes_inplace_mutation():
    log_probs = _make_log_probs((64, 4, 32), torch.float32).detach()
    target_lengths = torch.tensor(
        [16, 15, 14, 13], device=flag_gems.device, dtype=torch.long
    )
    targets = (torch.arange(58, device=flag_gems.device, dtype=torch.long) % 31) + 1
    input_lengths = torch.full((4,), 64, device=flag_gems.device, dtype=torch.long)

    flag_gems.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        reduction="sum",
    )

    input_lengths[0] = 65
    with pytest.raises(RuntimeError):
        flag_gems.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            reduction="sum",
        )

    input_lengths[0] = 64
    target_lengths[0] = 15
    with pytest.raises(RuntimeError):
        flag_gems.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            reduction="sum",
        )


@pytest.mark.ctc_loss
def test_ctc_loss_invalid_concatenated_target_raises():
    log_probs = _make_log_probs((6, 2, 5), torch.float32)
    targets = torch.tensor([1, 2], device=flag_gems.device, dtype=torch.long)
    input_lengths = torch.tensor([6, 6], device=flag_gems.device, dtype=torch.long)
    target_lengths = torch.tensor([2, 2], device=flag_gems.device, dtype=torch.long)

    with pytest.raises(RuntimeError):
        flag_gems.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            reduction="mean",
        )


@pytest.mark.ctc_loss
def test_ctc_loss_invalid_reduction_raises():
    log_probs = _make_log_probs((6, 1, 5), torch.float32)
    targets = torch.tensor([[1, 2]], device=flag_gems.device, dtype=torch.long)
    lengths = torch.tensor([6], device=flag_gems.device, dtype=torch.long)
    target_lengths = torch.tensor([2], device=flag_gems.device, dtype=torch.long)

    with pytest.raises(ValueError):
        flag_gems.ctc_loss(
            log_probs,
            targets,
            lengths,
            target_lengths,
            reduction="batchmean",
        )


@pytest.mark.ctc_loss
@pytest.mark.parametrize("path", ["direct", "registered"])
@pytest.mark.parametrize("blank", [-1, 5])
def test_ctc_loss_invalid_blank_raises(path, blank):
    log_probs = _make_log_probs((6, 2, 5), torch.float32).detach()
    targets = torch.tensor([[1, 2], [2, 3]], device=flag_gems.device, dtype=torch.long)
    input_lengths = torch.tensor([6, 6], device=flag_gems.device, dtype=torch.long)
    target_lengths = torch.tensor([2, 2], device=flag_gems.device, dtype=torch.long)

    with pytest.raises(RuntimeError):
        F.ctc_loss(
            utils.to_reference(log_probs),
            utils.to_reference(targets),
            utils.to_reference(input_lengths),
            utils.to_reference(target_lengths),
            blank=blank,
        )
    with pytest.raises(RuntimeError):
        _call_ctc_loss(
            path,
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=blank,
        )


@pytest.mark.ctc_loss
@pytest.mark.parametrize("path", ["direct", "registered"])
def test_ctc_loss_int_log_probs_raises(path):
    log_probs = (_make_log_probs((6, 2, 5), torch.float32).detach() * 100).to(
        torch.int32
    )
    targets = torch.tensor([[1, 2], [2, 3]], device=flag_gems.device, dtype=torch.long)
    input_lengths = torch.tensor([6, 6], device=flag_gems.device, dtype=torch.long)
    target_lengths = torch.tensor([2, 2], device=flag_gems.device, dtype=torch.long)

    with pytest.raises(RuntimeError):
        F.ctc_loss(
            utils.to_reference(log_probs),
            utils.to_reference(targets),
            utils.to_reference(input_lengths),
            utils.to_reference(target_lengths),
        )
    with pytest.raises(RuntimeError):
        _call_ctc_loss(
            path,
            log_probs,
            targets,
            input_lengths,
            target_lengths,
        )


@pytest.mark.ctc_loss
@pytest.mark.parametrize("path", ["direct", "registered"])
@pytest.mark.parametrize("length_name", ["input_lengths", "target_lengths"])
def test_ctc_loss_float_lengths_raise(path, length_name):
    log_probs = _make_log_probs((6, 2, 5), torch.float32).detach()
    targets = torch.tensor([[1, 2], [2, 3]], device=flag_gems.device, dtype=torch.long)
    input_lengths = torch.tensor([6, 6], device=flag_gems.device, dtype=torch.long)
    target_lengths = torch.tensor([2, 2], device=flag_gems.device, dtype=torch.long)
    if length_name == "input_lengths":
        input_lengths = input_lengths.float()
    else:
        target_lengths = target_lengths.float()

    with pytest.raises(RuntimeError):
        F.ctc_loss(
            utils.to_reference(log_probs),
            utils.to_reference(targets),
            utils.to_reference(input_lengths),
            utils.to_reference(target_lengths),
        )
    with pytest.raises(RuntimeError):
        _call_ctc_loss(
            path,
            log_probs,
            targets,
            input_lengths,
            target_lengths,
        )


@pytest.mark.ctc_loss
@pytest.mark.parametrize("path", ["direct", "registered"])
@pytest.mark.parametrize("target_layout", TARGET_LAYOUTS)
def test_ctc_loss_float_targets_match_pytorch(path, target_layout):
    utils.init_seed(505)
    t_steps, batch, classes, max_target = (8, 2, 6, 3)
    log_probs = _make_log_probs((t_steps, batch, classes), torch.float32).detach()
    targets, target_lengths = _make_targets(
        batch, max_target, classes, 0, "normal", target_layout
    )
    targets = targets.float()
    input_lengths = torch.tensor([8, 7], device=flag_gems.device, dtype=torch.long)

    _, ref_out = _reference_ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        0,
        "none",
        False,
    )
    res_out = _call_ctc_loss(
        path,
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        reduction="none",
    )

    utils.gems_assert_close(res_out, ref_out, torch.float32, reduce_dim=t_steps)

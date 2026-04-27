import pytest
import torch

import flag_gems

from .accuracy_utils import gems_assert_close, to_reference
from .conftest import QUICK_MODE

# (T, N, C, S) configurations: time, batch, classes, max target length
CTC_SHAPES = [
    (10, 1, 8, 3),  # small, single batch
    (10, 4, 8, 5),  # small, multi-batch
    (50, 8, 32, 15),  # medium
    (100, 4, 64, 25),  # medium-large
    (30, 16, 20, 10),  # wider batch
]
CTC_SHAPES_QUICK = [
    (10, 2, 8, 3),
    (50, 4, 32, 10),
]

CTC_TEST_SHAPES = CTC_SHAPES_QUICK if QUICK_MODE else CTC_SHAPES


@pytest.mark.ctc_loss
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("T_N_C_S", CTC_TEST_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_ctc_loss(T_N_C_S, dtype, reduction):
    # Fix seed to avoid rare numerical overflow in DP
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    T, N, C, S = T_N_C_S
    log_probs = torch.randn(T, N, C, dtype=dtype, device=flag_gems.device).log_softmax(
        2
    )
    targets = torch.randint(1, C, (N, S), dtype=torch.long, device=flag_gems.device)
    input_lengths = torch.full((N,), T, dtype=torch.long, device=flag_gems.device)
    target_lengths = torch.full((N,), S, dtype=torch.long, device=flag_gems.device)

    ref_inp = to_reference(log_probs)
    ref_targets = to_reference(targets)
    ref_il = to_reference(input_lengths)
    ref_tl = to_reference(target_lengths)

    ref_out = torch.nn.functional.ctc_loss(
        ref_inp, ref_targets, ref_il, ref_tl, blank=0, reduction=reduction
    )
    with flag_gems.use_gems():
        res_out = torch.nn.functional.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=0,
            reduction=reduction,
        )
    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.ctc_loss
@pytest.mark.parametrize("T_N_C_S", CTC_TEST_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_ctc_loss_backward(T_N_C_S, dtype):
    # Fix seed to avoid rare numerical overflow in DP
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    T, N, C, S = T_N_C_S
    inp = torch.randn(T, N, C, dtype=dtype, device=flag_gems.device).log_softmax(2)
    inp.requires_grad_(True)
    targets = torch.randint(1, C, (N, S), dtype=torch.long, device=flag_gems.device)
    input_lengths = torch.full((N,), T, dtype=torch.long, device=flag_gems.device)
    target_lengths = torch.full((N,), S, dtype=torch.long, device=flag_gems.device)

    # No upcast: CTC backward is sensitive to float32 vs float64 precision
    ref_inp = to_reference(inp)
    ref_targets = to_reference(targets)
    ref_il = to_reference(input_lengths)
    ref_tl = to_reference(target_lengths)

    ref_out = torch.nn.functional.ctc_loss(
        ref_inp,
        ref_targets,
        ref_il,
        ref_tl,
        blank=0,
        reduction="sum",
    )
    with flag_gems.use_gems():
        res_out = torch.nn.functional.ctc_loss(
            inp,
            targets,
            input_lengths,
            target_lengths,
            blank=0,
            reduction="sum",
        )

    out_grad = torch.ones_like(res_out)
    ref_grad = to_reference(out_grad)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    with flag_gems.use_gems():
        (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    # CTC backward involves T-step sequential DP; numerical error scales with T
    gems_assert_close(res_in_grad, ref_in_grad, dtype, equal_nan=True, reduce_dim=T)


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_ctc_loss_varying_lengths(dtype):
    T, N, C = 50, 4, 16
    log_probs = torch.randn(T, N, C, dtype=dtype, device=flag_gems.device).log_softmax(
        2
    )
    targets = torch.randint(1, C, (N, 10), dtype=torch.long, device=flag_gems.device)
    input_lengths = torch.tensor(
        [50, 30, 45, 20], dtype=torch.long, device=flag_gems.device
    )
    target_lengths = torch.tensor(
        [10, 7, 9, 4], dtype=torch.long, device=flag_gems.device
    )

    ref_inp = to_reference(log_probs)
    ref_targets = to_reference(targets)

    ref_out = torch.nn.functional.ctc_loss(
        ref_inp,
        ref_targets,
        input_lengths.cpu(),
        target_lengths.cpu(),
        blank=0,
        reduction="none",
    )
    with flag_gems.use_gems():
        res_out = torch.nn.functional.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=0,
            reduction="none",
        )
    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_ctc_loss_varying_lengths_backward(dtype):
    T, N, C = 50, 4, 16
    inp = torch.randn(T, N, C, dtype=dtype, device=flag_gems.device).log_softmax(2)
    inp.requires_grad_(True)
    targets = torch.randint(1, C, (N, 10), dtype=torch.long, device=flag_gems.device)
    input_lengths = torch.tensor(
        [50, 30, 45, 20], dtype=torch.long, device=flag_gems.device
    )
    target_lengths = torch.tensor(
        [10, 7, 9, 4], dtype=torch.long, device=flag_gems.device
    )

    ref_inp = to_reference(inp)
    ref_targets = to_reference(targets)
    ref_il = to_reference(input_lengths)
    ref_tl = to_reference(target_lengths)

    ref_out = torch.nn.functional.ctc_loss(
        ref_inp,
        ref_targets,
        ref_il,
        ref_tl,
        blank=0,
        reduction="sum",
    )
    with flag_gems.use_gems():
        res_out = torch.nn.functional.ctc_loss(
            inp,
            targets,
            input_lengths,
            target_lengths,
            blank=0,
            reduction="sum",
        )

    out_grad = torch.ones_like(res_out)
    ref_grad = to_reference(out_grad)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    with flag_gems.use_gems():
        (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    # CTC backward involves T-step sequential DP; numerical error scales with T
    gems_assert_close(res_in_grad, ref_in_grad, dtype, equal_nan=True, reduce_dim=T)


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_ctc_loss_empty_target(dtype):
    T, N, C = 20, 2, 8
    log_probs = torch.randn(T, N, C, dtype=dtype, device=flag_gems.device).log_softmax(
        2
    )
    targets = torch.randint(1, C, (N, 5), dtype=torch.long, device=flag_gems.device)
    input_lengths = torch.tensor([T, T], dtype=torch.long, device=flag_gems.device)
    target_lengths = torch.tensor([5, 0], dtype=torch.long, device=flag_gems.device)

    ref_inp = to_reference(log_probs)
    ref_targets = to_reference(targets)

    ref_out = torch.nn.functional.ctc_loss(
        ref_inp,
        ref_targets,
        input_lengths.cpu(),
        target_lengths.cpu(),
        blank=0,
        reduction="none",
    )
    with flag_gems.use_gems():
        res_out = torch.nn.functional.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=0,
            reduction="none",
        )
    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.ctc_loss
@pytest.mark.parametrize("blank", [0, 3])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_ctc_loss_blank(blank, dtype):
    T, N, C = 20, 3, 8
    log_probs = torch.randn(T, N, C, dtype=dtype, device=flag_gems.device).log_softmax(
        2
    )
    # Targets must not contain the blank label
    valid_labels = [i for i in range(C) if i != blank]
    targets = torch.tensor(
        [[valid_labels[j % len(valid_labels)] for j in range(5)] for _ in range(N)],
        dtype=torch.long,
        device=flag_gems.device,
    )
    input_lengths = torch.full((N,), T, dtype=torch.long, device=flag_gems.device)
    target_lengths = torch.full((N,), 5, dtype=torch.long, device=flag_gems.device)

    ref_inp = to_reference(log_probs)
    ref_targets = to_reference(targets)

    ref_out = torch.nn.functional.ctc_loss(
        ref_inp,
        ref_targets,
        input_lengths.cpu(),
        target_lengths.cpu(),
        blank=blank,
        reduction="none",
    )
    with flag_gems.use_gems():
        res_out = torch.nn.functional.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=blank,
            reduction="none",
        )
    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_ctc_loss_1d_targets(dtype):
    T, N, C = 20, 3, 8
    log_probs = torch.randn(T, N, C, dtype=dtype, device=flag_gems.device).log_softmax(
        2
    )
    # 1D concatenated targets
    target_lengths = torch.tensor([5, 3, 4], dtype=torch.long, device=flag_gems.device)
    targets_1d = torch.randint(
        1, C, (target_lengths.sum().item(),), dtype=torch.long, device=flag_gems.device
    )
    input_lengths = torch.full((N,), T, dtype=torch.long, device=flag_gems.device)

    ref_inp = to_reference(log_probs)
    ref_targets = to_reference(targets_1d)

    ref_out = torch.nn.functional.ctc_loss(
        ref_inp,
        ref_targets,
        input_lengths.cpu().tolist(),
        target_lengths.cpu().tolist(),
        blank=0,
        reduction="none",
    )
    with flag_gems.use_gems():
        res_out = torch.nn.functional.ctc_loss(
            log_probs,
            targets_1d,
            input_lengths,
            target_lengths,
            blank=0,
            reduction="none",
        )
    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_ctc_loss_zero_infinity(dtype):
    T, N, C = 20, 2, 8
    log_probs = torch.randn(T, N, C, dtype=dtype, device=flag_gems.device).log_softmax(
        2
    )
    targets = torch.randint(1, C, (N, 5), dtype=torch.long, device=flag_gems.device)
    input_lengths = torch.full((N,), T, dtype=torch.long, device=flag_gems.device)
    target_lengths = torch.full((N,), 5, dtype=torch.long, device=flag_gems.device)

    ref_inp = to_reference(log_probs)
    ref_targets = to_reference(targets)

    ref_out = torch.nn.functional.ctc_loss(
        ref_inp,
        ref_targets,
        input_lengths.cpu(),
        target_lengths.cpu(),
        blank=0,
        reduction="mean",
        zero_infinity=True,
    )
    with flag_gems.use_gems():
        res_out = torch.nn.functional.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=0,
            reduction="mean",
            zero_infinity=True,
        )
    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_accuracy_ctc_loss_half_precision(dtype):
    T, N, C = 20, 2, 8
    log_probs = torch.randn(T, N, C, dtype=dtype, device=flag_gems.device).log_softmax(
        2
    )
    targets = torch.randint(1, C, (N, 5), dtype=torch.long, device=flag_gems.device)
    input_lengths = torch.full((N,), T, dtype=torch.long, device=flag_gems.device)
    target_lengths = torch.full((N,), 5, dtype=torch.long, device=flag_gems.device)

    ref_inp = to_reference(log_probs, upcast=True)
    ref_targets = to_reference(targets)
    ref_il = to_reference(input_lengths)
    ref_tl = to_reference(target_lengths)

    ref_out = torch.nn.functional.ctc_loss(
        ref_inp,
        ref_targets,
        ref_il,
        ref_tl,
        blank=0,
        reduction="none",
    )
    with flag_gems.use_gems():
        res_out = torch.nn.functional.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=0,
            reduction="none",
        )
    # CTC composite decomposition returns float32 regardless of input dtype
    gems_assert_close(res_out, ref_out, res_out.dtype, equal_nan=True)

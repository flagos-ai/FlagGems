# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Correctness tests for torch.ops.aten._cudnn_ctc_loss.

Native contract (isolated probes on the active backend):

* log_probs is a 3-D float32 compute-device tensor; every other rank and scalar
  type is rejected by the operator itself.
* targets is a 1-D CPU int32 label tensor with labels below C and blank = 0, of
  length sum(target_lengths); input_lengths/target_lengths are length-N int
  lists.
* the result is (nll (N,), log_alpha (T, N, C)) float32; only nll carries a
  grad_fn. Both are accumulated log-likelihoods, so the shared close comparison
  applies.

Every positive case compares the whole tuple against torch.ops.aten on an
independent equal-valued input built with tu.to_reference, so a candidate
cannot pass by sharing state with the reference.

Coverage notes:

* The operator is fixed at rank 3, so the spec's 0-5 dim shape grid is mapped
  onto (T, N, C) triples of the same element count; the 0-dim and 1-element
  shapes both map to the minimal legal (1, 1, 2), since C must be >= 2. The
  quick level uses the required (2, 19, 7) triple.
* A length-L transcript fits T frames when T >= L + (adjacent equal label
  pairs): 2 * L + 1 is the expanded state count, not a frame count, so the
  quick row's single label fits T == 2.
* log_probs accepts float32 only, so the other eight spec dtypes are negative
  cases; bf16/fp64/int64/fp8 are gated on the static backend capability flags
  so their construction cannot fail in the input helper first.
* No broadcast case: targets and the length arguments are metadata, not a
  second broadcastable operand.
* The schema has no optional arguments, so blank/deterministic/zero_infinity are
  always passed explicitly; the boolean-parameter test covers both values of
  each.
* Measured native scope: every row delivered here is repeatable for
  deterministic=True on an independent equal-valued buffer. Two native calls on
  one shared buffer can disagree element-wise where cuDNN leaves log_alpha[t, i]
  undefined (t >= input_lengths[i], or a transcript that cannot fit its frames)
  or under the non-deterministic algorithm; descriptors measured that way are
  excluded here and kept with their exact shapes in the gap record.
"""

import pytest
import torch
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

setattr(
    pytest.mark,
    "_cudnn_ctc_loss",
    MarkDecorator(Mark("_cudnn_ctc_loss", (), {}, _ispytest=True), _ispytest=True),
)

TARGETS_DTYPE = torch.int32

# Native targets are CPU metadata: a compute-device or non-contiguous targets
# tensor raises.
SUPPORTED_LOG_PROBS_DTYPES = (torch.float32,)
# Every other scalar type is rejected by the operator itself with
# 'Expected tensor for argument #1 log_probs to have scalar type Float'. The
# optional types are gated on the static backend capability flags so their
# construction cannot fail in the input helper on a backend without support.
ALWAYS_REJECTED_LOG_PROBS_DTYPES = (
    torch.float16,
    torch.int8,
    torch.uint8,
    torch.int32,
    torch.bool,
)
CAPABILITY_GATED_LOG_PROBS_DTYPES = (
    (torch.bfloat16, utils.bf16_is_supported),
    (torch.float64, utils.fp64_is_supported),
    (torch.int64, utils.int64_is_supported),
    (torch.float8_e4m3fn, utils.fp8_is_supported),
    (torch.float8_e5m2, utils.fp8_is_supported),
)
REJECTED_LOG_PROBS_DTYPES = ALWAYS_REJECTED_LOG_PROBS_DTYPES + tuple(
    dtype for dtype, supported in CAPABILITY_GATED_LOG_PROBS_DTYPES if supported
)


def _labels(target_lengths, num_classes):
    """CPU int32 targets of length sum(target_lengths), labels in [1, C - 1]."""
    total = sum(target_lengths)
    if total == 0:
        return torch.zeros(0, dtype=TARGETS_DTYPE)
    return torch.arange(total, dtype=TARGETS_DTYPE) % (num_classes - 1) + 1


# (time_steps, batch, num_classes, input_lengths, target_lengths) rows. Every row
# uses input_lengths[i] == time_steps and a transcript that fits, so
# log_alpha[t, i] is a defined CTC recursion step for every frame. Spec sizes are
# mapped to (T, N, C) triples of the same element count: () and (1,) -> the
# minimal legal (1, 1, 2), (256,) -> (128, 1, 2), (1024, 1024) -> (1024, 32, 32),
# (20, 320, 15) as is, (16, 128, 64, 60) -> (1024, 128, 60), and
# (16, 7, 57, 32, 29) -> (6384, 32, 29).
GRID_ROWS = [
    (1, 1, 2, [1], [1]),
    (2, 19, 7, [2] * 19, [1] * 19),
    (4, 1, 2, [4], [1]),
    (7, 5, 9, [7] * 5, [6, 1, 5, 2, 6]),
    (8, 3, 6, [8] * 3, [3] * 3),
    (9, 7, 5, [9] * 7, [1] * 7),
    (16, 7, 57, [16] * 7, [8] * 7),
    (20, 320, 15, [20] * 320, [8] * 320),
    (24, 6, 12, [24] * 6, [1, 2, 3, 4, 5, 6]),
    (32, 16, 40, [32] * 16, [10] * 16),
    (50, 32, 20, [50] * 32, [20] * 32),
    (64, 4, 12, [64] * 4, [1, 2, 3, 4]),
    (128, 1, 2, [128], [1]),
    (128, 16, 64, [128] * 16, [32] * 16),
    (256, 8, 50, [256] * 8, [64] * 8),
    (1024, 4, 8, [1024] * 4, [64] * 4),
    (1024, 32, 32, [1024] * 32, [32] * 32),
    (1024, 128, 60, [1024] * 128, [32] * 128),
    (24, 128, 64, [24] * 128, [8] * 128),
    (6384, 32, 29, [6384] * 32, [32] * 32),
]
# Quick level keeps the required (2, 19, 7) size as (T, N, C) with the one-label
# transcript that fits two frames.
QUICK_ROWS = [(2, 19, 7, [2] * 19, [1] * 19)]


@pytest.mark._cudnn_ctc_loss
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("row", tu.selected_cases(GRID_ROWS, quick=QUICK_ROWS))
def test__cudnn_ctc_loss_accuracy(row, value_range):
    # deterministic=True keeps the vendor reference repeatable for one input;
    # both bool values are covered by the parameter test below.
    time_steps, batch, num_classes, input_lengths, target_lengths = row
    log_probs = tu.make_input(
        torch.float32, (time_steps, batch, num_classes), value_range
    )
    ref_log_probs = tu.to_reference(log_probs)
    targets = _labels(target_lengths, num_classes)

    ref_nll, ref_log_alpha = torch.ops.aten._cudnn_ctc_loss(
        ref_log_probs, targets, input_lengths, target_lengths, 0, True, True
    )
    res_nll, res_log_alpha = flag_gems._cudnn_ctc_loss(
        log_probs, targets, input_lengths, target_lengths, 0, True, True
    )

    tu.assert_result_close(res_nll, ref_nll)
    tu.assert_result_close(res_log_alpha, ref_log_alpha)


# An all-empty target_lengths batch is accepted natively: the blank-only
# alignment exists, so the whole tuple is compared.
EMPTY_TARGET_ROWS = [
    (2, 19, 7, [2] * 19, [0] * 19),
    (8, 3, 6, [8] * 3, [0] * 3),
    (16, 4, 10, [16] * 4, [0] * 4),
]


@pytest.mark._cudnn_ctc_loss
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("row", tu.selected_cases(EMPTY_TARGET_ROWS, quick=[]))
def test__cudnn_ctc_loss_empty_targets(row, value_range):
    time_steps, batch, num_classes, input_lengths, target_lengths = row
    log_probs = tu.make_input(
        torch.float32, (time_steps, batch, num_classes), value_range
    )
    ref_log_probs = tu.to_reference(log_probs)
    targets = _labels(target_lengths, num_classes)

    ref_nll, ref_log_alpha = torch.ops.aten._cudnn_ctc_loss(
        ref_log_probs, targets, input_lengths, target_lengths, 0, True, True
    )
    res_nll, res_log_alpha = flag_gems._cudnn_ctc_loss(
        log_probs, targets, input_lengths, target_lengths, 0, True, True
    )

    tu.assert_result_close(res_nll, ref_nll)
    tu.assert_result_close(res_log_alpha, ref_log_alpha)


SPECIAL_ROWS = [
    (8, 3, 6, [8] * 3, [3] * 3),
    (16, 4, 10, [16] * 4, [5] * 4),
]

SPECIAL_CASES = tu.selected_cases(
    [
        (dtype, scenario, row)
        for dtype, scenario in tu.special_value_cases(SUPPORTED_LOG_PROBS_DTYPES)
        for row in SPECIAL_ROWS
    ],
    quick=[],
)


def _special_log_probs(dtype, scenario, shape):
    """Spread the shared special-value payload over a (T, N, C) tensor."""
    payload = tu.make_special_input(dtype, scenario)
    offsets = torch.arange(shape[0] * shape[1] * shape[2], device=payload.device)
    return payload[offsets % payload.numel()].reshape(shape)


@pytest.mark._cudnn_ctc_loss
@pytest.mark.parametrize("dtype,scenario,row", SPECIAL_CASES)
def test__cudnn_ctc_loss_special_values(dtype, scenario, row):
    time_steps, batch, num_classes, input_lengths, target_lengths = row
    log_probs = _special_log_probs(dtype, scenario, (time_steps, batch, num_classes))
    ref_log_probs = tu.to_reference(log_probs)
    targets = _labels(target_lengths, num_classes)

    ref_nll, ref_log_alpha = torch.ops.aten._cudnn_ctc_loss(
        ref_log_probs, targets, input_lengths, target_lengths, 0, True, True
    )
    res_nll, res_log_alpha = flag_gems._cudnn_ctc_loss(
        log_probs, targets, input_lengths, target_lengths, 0, True, True
    )

    tu.assert_result_close(res_nll, ref_nll)
    tu.assert_result_close(res_log_alpha, ref_log_alpha)


FLAG_CASES = ((False, False), (False, True), (True, False), (True, True))
# Parameter coverage uses the (20, 320, 15) spec size as (T, N, C) with a
# transcript length that fits its 20 frames.
PARAM_ROW = (20, 320, 15, [20] * 320, [8] * 320)


@pytest.mark._cudnn_ctc_loss
@pytest.mark.parametrize(
    "deterministic,zero_infinity", tu.selected_cases(FLAG_CASES, quick=[])
)
def test__cudnn_ctc_loss_boolean_parameters(deterministic, zero_infinity):
    # blank/deterministic/zero_infinity have no schema default, so blank=0 is
    # always passed explicitly; the four cases cover both values of both bools.
    time_steps, batch, num_classes, input_lengths, target_lengths = PARAM_ROW
    log_probs = tu.make_input(
        torch.float32, (time_steps, batch, num_classes), ["-1", "1"]
    )
    ref_log_probs = tu.to_reference(log_probs)
    targets = _labels(target_lengths, num_classes)

    ref_nll, ref_log_alpha = torch.ops.aten._cudnn_ctc_loss(
        ref_log_probs,
        targets,
        input_lengths,
        target_lengths,
        0,
        deterministic,
        zero_infinity,
    )
    res_nll, res_log_alpha = flag_gems._cudnn_ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        0,
        deterministic,
        zero_infinity,
    )

    tu.assert_result_close(res_nll, ref_nll)
    tu.assert_result_close(res_log_alpha, ref_log_alpha)


OUT_ROWS = [
    (8, 3, 6, [8] * 3, [3] * 3),
    (16, 4, 10, [16] * 4, [5] * 4),
]


@pytest.mark._cudnn_ctc_loss
@pytest.mark.parametrize("row", tu.selected_cases(OUT_ROWS, quick=[]))
def test__cudnn_ctc_loss_out(row):
    time_steps, batch, num_classes, input_lengths, target_lengths = row
    log_probs = tu.make_input(
        torch.float32, (time_steps, batch, num_classes), ["-1", "1"]
    )
    ref_log_probs = tu.to_reference(log_probs)
    targets = _labels(target_lengths, num_classes)
    ref_nll = torch.empty(batch, dtype=ref_log_probs.dtype, device=ref_log_probs.device)
    ref_log_alpha = torch.empty(
        (time_steps, batch, num_classes),
        dtype=ref_log_probs.dtype,
        device=ref_log_probs.device,
    )
    res_nll = torch.empty(batch, dtype=log_probs.dtype, device=log_probs.device)
    res_log_alpha = torch.empty(
        (time_steps, batch, num_classes),
        dtype=log_probs.dtype,
        device=log_probs.device,
    )

    ref = torch.ops.aten._cudnn_ctc_loss.out(
        ref_log_probs,
        targets,
        input_lengths,
        target_lengths,
        0,
        True,
        True,
        out0=ref_nll,
        out1=ref_log_alpha,
    )
    res = flag_gems._cudnn_ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        0,
        True,
        True,
        out0=res_nll,
        out1=res_log_alpha,
    )

    # the out= form writes into and returns the caller-provided buffers
    assert res[0] is res_nll
    assert res[1] is res_log_alpha
    tu.assert_result_close(res[0], ref[0])
    tu.assert_result_close(res[1], ref[1])


@pytest.mark._cudnn_ctc_loss
@pytest.mark.parametrize("row", tu.selected_cases([OUT_ROWS[0]], quick=[]))
def test__cudnn_ctc_loss_out_strided(row):
    """A non-contiguous out1 is accepted and keeps its layout metadata."""
    time_steps, batch, num_classes, input_lengths, target_lengths = row
    log_probs = tu.make_input(
        torch.float32, (time_steps, batch, num_classes), ["-1", "1"]
    )
    ref_log_probs = tu.to_reference(log_probs)
    targets = _labels(target_lengths, num_classes)
    ref_nll = torch.empty(batch, dtype=ref_log_probs.dtype, device=ref_log_probs.device)
    ref_storage = torch.empty(
        (time_steps, batch, 2 * num_classes),
        dtype=ref_log_probs.dtype,
        device=ref_log_probs.device,
    )
    ref_log_alpha = ref_storage[:, :, ::2]

    res_nll = torch.empty(batch, dtype=log_probs.dtype, device=log_probs.device)
    storage = torch.empty(
        (time_steps, batch, 2 * num_classes),
        dtype=log_probs.dtype,
        device=log_probs.device,
    )
    res_log_alpha = storage[:, :, ::2]
    stride = res_log_alpha.stride()
    offset = res_log_alpha.storage_offset()

    ref = torch.ops.aten._cudnn_ctc_loss.out(
        ref_log_probs,
        targets,
        input_lengths,
        target_lengths,
        0,
        True,
        True,
        out0=ref_nll,
        out1=ref_log_alpha,
    )
    res = flag_gems._cudnn_ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        0,
        True,
        True,
        out0=res_nll,
        out1=res_log_alpha,
    )

    assert res[0] is res_nll
    assert res[1] is res_log_alpha
    # the strided buffer is updated in place, not reallocated or compacted
    assert res[1].stride() == stride
    assert res[1].storage_offset() == offset
    tu.assert_result_close(res[0], ref[0])
    tu.assert_result_close(res[1], ref[1])


TENSOR_ROWS = [
    (8, 3, 6, [8] * 3, [3] * 3),
    (16, 4, 10, [16] * 4, [5] * 4),
]


@pytest.mark._cudnn_ctc_loss
@pytest.mark.parametrize("row", tu.selected_cases(TENSOR_ROWS, quick=[]))
def test__cudnn_ctc_loss_length_tensors(row):
    """The .Tensor overload takes int32 length tensors on the compute device."""
    time_steps, batch, num_classes, input_lengths, target_lengths = row
    log_probs = tu.make_input(
        torch.float32, (time_steps, batch, num_classes), ["-1", "1"]
    )
    ref_log_probs = tu.to_reference(log_probs)
    targets = _labels(target_lengths, num_classes)
    input_length_tensor = torch.tensor(
        input_lengths, dtype=TARGETS_DTYPE, device=log_probs.device
    )
    target_length_tensor = torch.tensor(
        target_lengths, dtype=TARGETS_DTYPE, device=log_probs.device
    )
    ref_input_length_tensor = tu.to_reference(input_length_tensor)
    ref_target_length_tensor = tu.to_reference(target_length_tensor)

    ref_nll, ref_log_alpha = torch.ops.aten._cudnn_ctc_loss.Tensor(
        ref_log_probs,
        targets,
        ref_input_length_tensor,
        ref_target_length_tensor,
        0,
        True,
        True,
    )
    res_nll, res_log_alpha = flag_gems._cudnn_ctc_loss(
        log_probs,
        targets,
        input_length_tensor,
        target_length_tensor,
        0,
        True,
        True,
    )

    tu.assert_result_close(res_nll, ref_nll)
    tu.assert_result_close(res_log_alpha, ref_log_alpha)


# Backward rows use a uniform input_lengths: the native gradient is undefined for
# t >= input_lengths[i], so a padded batch cannot be compared gradient-wise.
BACKWARD_ROWS = [
    (8, 3, 6, [8] * 3, [3] * 3),
    (16, 8, 16, [16] * 8, [5] * 8),
]


@pytest.mark._cudnn_ctc_loss
@pytest.mark.parametrize("row", tu.selected_cases(BACKWARD_ROWS, quick=[]))
def test__cudnn_ctc_loss_backward(row):
    # log_alpha is a native constant (no grad_fn), so only nll is differentiated.
    time_steps, batch, num_classes, input_lengths, target_lengths = row
    log_probs = tu.make_input(
        torch.float32, (time_steps, batch, num_classes), ["-1", "1"]
    )
    ref_log_probs = tu.to_reference(log_probs).requires_grad_(True)
    res_log_probs = log_probs.clone().requires_grad_(True)
    targets = _labels(target_lengths, num_classes)
    upstream = tu.make_input(torch.float32, (batch,), ["-1", "1"])
    ref_upstream = tu.to_reference(upstream)

    ref_nll, ref_log_alpha = torch.ops.aten._cudnn_ctc_loss(
        ref_log_probs, targets, input_lengths, target_lengths, 0, True, True
    )
    res_nll, res_log_alpha = flag_gems._cudnn_ctc_loss(
        res_log_probs, targets, input_lengths, target_lengths, 0, True, True
    )
    tu.assert_result_close(res_nll, ref_nll)
    tu.assert_result_close(res_log_alpha, ref_log_alpha)

    (ref_grad,) = torch.autograd.grad(ref_nll, ref_log_probs, grad_outputs=ref_upstream)
    (res_grad,) = torch.autograd.grad(res_nll, res_log_probs, grad_outputs=upstream)
    tu.assert_result_close(res_grad, ref_grad)


def _valid_workload(time_steps=8, batch=3, num_classes=6, target_length=3):
    """A native-valid workload used as the base of the negative cases."""
    log_probs = tu.make_input(
        torch.float32, (time_steps, batch, num_classes), ["-1", "1"]
    )
    targets = _labels([target_length] * batch, num_classes)
    return log_probs, targets, [time_steps] * batch, [target_length] * batch


@pytest.mark._cudnn_ctc_loss
@pytest.mark.parametrize("dtype", REJECTED_LOG_PROBS_DTYPES)
def test__cudnn_ctc_loss_rejects_unsupported_log_probs_dtype(dtype):
    _, targets, input_lengths, target_lengths = _valid_workload()
    log_probs = tu.make_input(dtype, (8, 3, 6), ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._cudnn_ctc_loss(
            log_probs, targets, input_lengths, target_lengths, 0, False, True
        )


@pytest.mark._cudnn_ctc_loss
@pytest.mark.parametrize("shape", [(8, 6), (2, 3, 6, 4)])
def test__cudnn_ctc_loss_rejects_wrong_log_probs_rank(shape):
    log_probs = tu.make_input(torch.float32, shape, ["-1", "1"])
    targets = _labels([3, 3, 3], 6)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._cudnn_ctc_loss(
            log_probs, targets, [8, 8, 8], [3, 3, 3], 0, False, True
        )


@pytest.mark._cudnn_ctc_loss
@pytest.mark.parametrize("dtype", [torch.int64, torch.float32, torch.uint8])
def test__cudnn_ctc_loss_rejects_wrong_targets_dtype(dtype):
    log_probs, _, input_lengths, target_lengths = _valid_workload()
    targets = torch.zeros(sum(target_lengths), dtype=dtype)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._cudnn_ctc_loss(
            log_probs, targets, input_lengths, target_lengths, 0, False, True
        )


@pytest.mark._cudnn_ctc_loss
@pytest.mark.parametrize("strided", [False, True])
def test__cudnn_ctc_loss_rejects_targets_layout(strided):
    """Targets must be CPU int32: compute-device or non-contiguous both raise."""
    log_probs, _, input_lengths, target_lengths = _valid_workload()
    total = sum(target_lengths)
    if strided:
        targets = torch.zeros(2 * total, dtype=TARGETS_DTYPE)[::2]
    else:
        targets = torch.zeros(total, dtype=TARGETS_DTYPE, device=log_probs.device)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._cudnn_ctc_loss(
            log_probs, targets, input_lengths, target_lengths, 0, False, True
        )


@pytest.mark._cudnn_ctc_loss
@pytest.mark.parametrize("blank", [1, 5, 3])
def test__cudnn_ctc_loss_rejects_nonzero_blank(blank):
    log_probs, targets, input_lengths, target_lengths = _valid_workload()

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._cudnn_ctc_loss(
            log_probs, targets, input_lengths, target_lengths, blank, False, True
        )


@pytest.mark._cudnn_ctc_loss
@pytest.mark.parametrize("which", ["input_lengths", "target_lengths"])
def test__cudnn_ctc_loss_rejects_length_count_mismatch(which):
    log_probs, targets, input_lengths, target_lengths = _valid_workload()
    if which == "input_lengths":
        input_lengths = [8, 8]
    else:
        target_lengths = [3, 3]

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._cudnn_ctc_loss(
            log_probs, targets, input_lengths, target_lengths, 0, False, True
        )


@pytest.mark._cudnn_ctc_loss
def test__cudnn_ctc_loss_rejects_input_lengths_beyond_time():
    log_probs, targets, _, target_lengths = _valid_workload()

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._cudnn_ctc_loss(
            log_probs, targets, [9, 8, 8], target_lengths, 0, False, True
        )


@pytest.mark._cudnn_ctc_loss
def test__cudnn_ctc_loss_rejects_single_class():
    # C == 1 is a native BAD_PARAM; the targets are given explicitly because the
    # label builder would divide by C - 1.
    log_probs = tu.make_input(torch.float32, (8, 3, 1), ["-1", "1"])
    targets = torch.zeros(3, dtype=TARGETS_DTYPE)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._cudnn_ctc_loss(
            log_probs, targets, [8, 8, 8], [1, 1, 1], 0, False, True
        )


@pytest.mark._cudnn_ctc_loss
def test__cudnn_ctc_loss_rejects_single_class_empty_targets():
    """C == 1 with a blank-only transcript is native BAD_PARAM as well."""
    log_probs = tu.make_input(torch.float32, (8, 3, 1), ["-1", "1"])
    targets = torch.zeros(0, dtype=TARGETS_DTYPE)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._cudnn_ctc_loss(
            log_probs, targets, [8, 8, 8], [0, 0, 0], 0, False, True
        )


@pytest.mark._cudnn_ctc_loss
def test__cudnn_ctc_loss_rejects_empty_batch():
    log_probs = torch.empty((8, 0, 6), dtype=torch.float32, device=flag_gems.device)
    targets = torch.zeros(0, dtype=TARGETS_DTYPE)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._cudnn_ctc_loss(log_probs, targets, [], [], 0, False, True)

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

"""Correctness tests for torch.ops.aten._use_cudnn_ctc_loss.

Boolean capability predicate consulted by torch.nn.CTCLoss before choosing
between the cuDNN CTC path and native _ctc_loss. It inspects rank, dtype, device
and the two length lists, returns a Python bool and allocates nothing:

    _use_cudnn_ctc_loss(log_probs, targets, int[] input_lengths,
                        int[] target_lengths, int blank) -> bool
    _use_cudnn_ctc_loss.Tensor(log_probs, targets, Tensor input_lengths,
                               Tensor target_lengths, int blank) -> bool

Native conditions (LossCTC.cpp): cuDNN enabled, blank == 0, targets rank-1
int32 contiguous (on CPU for the int[] form), log_probs float32 rank-3 on CUDA,
every input_length == log_probs.size(0), every target_lengths[b] < 256 and
<= input_lengths[b]; the .Tensor overload drops the CPU-targets test, checks the
length tensors are int32, and bounds its loop by target_lengths.size(). A
rejected dtype, rank or device yields a plain False, not an exception, so those
rows are compared as booleans; only malformed calls raise. The expected value is
whatever the backend really reports, so on a device without the cuDNN path the
test still checks the honest predicate contract instead of demanding a vendor
implementation.

Inapplicable dimensions: broadcast (there is no second elementwise operand; the
operands are only inspected for rank/dtype/device/contiguity), backward (the
result is a bool with no autograd graph), parameter sweeps (blank is the only
parameter, it is required and has no schema default), and fp8 special values
further below.
"""

import pytest
import torch

import flag_gems

from . import test_utils as tu

# log_probs is read by dtype only: float32 continues through the predicate, the
# other required dtypes are rejected by the scalar_type test.
_ACCEPTED_LOG_PROBS_DTYPES = [torch.float32]

_REJECTED_LOG_PROBS_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int64,
]


def _ctc_shape(shape):
    # log_probs is rank-3 by contract, so a spec shape of another rank is folded
    # to (T, N, C) with the same element count and stays a valid input.
    if len(shape) == 3:
        return shape
    numel = 1
    for extent in shape:
        numel *= extent
    return (numel, 1, 1)


def _targets(batch, length, dtype=torch.int32):
    # The int[] overload requires rank-1 int32 contiguous targets on CPU, so
    # this operand deliberately does not use flag_gems.device.
    return torch.zeros(batch * length, dtype=dtype, device="cpu")


def _where(device_kind):
    return "cpu" if device_kind == "cpu" else flag_gems.device


def _assert_bool(res_out, ref_out):
    # The result is a Python scalar: check its type and value directly instead of
    # wrapping it in an artificial tensor.
    assert type(res_out) is bool, type(res_out)
    assert res_out == ref_out, (res_out, ref_out)


@pytest.mark.use_cudnn_ctc_loss
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _ACCEPTED_LOG_PROBS_DTYPES)
def test__use_cudnn_ctc_loss(shape, value_range, dtype):
    t, n, c = _ctc_shape(shape)
    target_length = min(3, c)
    log_probs = tu.make_input(dtype, (t, n, c), value_range)
    targets = _targets(n, target_length)
    input_lengths = [t] * n
    target_lengths = [target_length] * n

    ref_log_probs = tu.to_reference(log_probs)
    ref_targets = tu.to_reference(targets)

    ref_out = torch.ops.aten._use_cudnn_ctc_loss(
        ref_log_probs, ref_targets, input_lengths, target_lengths, 0
    )
    res_out = flag_gems._use_cudnn_ctc_loss(
        log_probs, targets, input_lengths, target_lengths, 0
    )

    _assert_bool(res_out, ref_out)


@pytest.mark.use_cudnn_ctc_loss
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _REJECTED_LOG_PROBS_DTYPES)
def test__use_cudnn_ctc_loss_rejected_log_probs_dtype(shape, dtype):
    # The scalar_type test runs before any element is read, so one value range
    # suffices: the values cannot change the outcome and it is a plain False.
    t, n, c = _ctc_shape(shape)
    log_probs = tu.make_input(dtype, (t, n, c), ["-1", "1"])
    targets = _targets(n, 3)

    ref_log_probs = tu.to_reference(log_probs)
    ref_targets = tu.to_reference(targets)

    ref_out = torch.ops.aten._use_cudnn_ctc_loss(
        ref_log_probs, ref_targets, [t] * n, [3] * n, 0
    )
    res_out = flag_gems._use_cudnn_ctc_loss(log_probs, targets, [t] * n, [3] * n, 0)

    _assert_bool(res_out, ref_out)


@pytest.mark.use_cudnn_ctc_loss
@pytest.mark.parametrize(
    "targets_dtype", [torch.int32, torch.int64, torch.int16, torch.float32]
)
def test__use_cudnn_ctc_loss_targets_dtype(targets_dtype):
    log_probs = tu.make_input(torch.float32, (8, 3, 5), ["-1", "1"])
    targets = _targets(3, 3, dtype=targets_dtype)

    ref_log_probs = tu.to_reference(log_probs)
    ref_targets = tu.to_reference(targets)

    ref_out = torch.ops.aten._use_cudnn_ctc_loss(
        ref_log_probs, ref_targets, [8, 8, 8], [3, 3, 3], 0
    )
    res_out = flag_gems._use_cudnn_ctc_loss(log_probs, targets, [8, 8, 8], [3, 3, 3], 0)

    _assert_bool(res_out, ref_out)


def _targets_variant(kind, batch, length):
    if kind == "contiguous":
        return _targets(batch, length)
    if kind == "offset":
        # A contiguous slice with a nonzero storage offset stays acceptable.
        base = torch.zeros(batch * length + 3, dtype=torch.int32, device="cpu")
        return base[1 : 1 + batch * length]
    if kind == "noncontiguous":
        return torch.zeros(2 * batch * length, dtype=torch.int32, device="cpu")[::2]
    if kind == "device":
        return torch.zeros(batch * length, dtype=torch.int32, device=flag_gems.device)
    if kind == "rank2":
        return torch.zeros((1, batch * length), dtype=torch.int32, device="cpu")
    if kind == "rank0":
        return torch.zeros((), dtype=torch.int32, device="cpu")
    raise ValueError(kind)


@pytest.mark.use_cudnn_ctc_loss
@pytest.mark.parametrize(
    "kind", ["contiguous", "offset", "noncontiguous", "device", "rank2", "rank0"]
)
def test__use_cudnn_ctc_loss_targets_layout(kind):
    log_probs = tu.make_input(torch.float32, (8, 3, 5), ["-1", "1"])
    targets = _targets_variant(kind, 3, 3)

    ref_log_probs = tu.to_reference(log_probs)
    ref_targets = tu.to_reference(targets)

    ref_out = torch.ops.aten._use_cudnn_ctc_loss(
        ref_log_probs, ref_targets, [8, 8, 8], [3, 3, 3], 0
    )
    res_out = flag_gems._use_cudnn_ctc_loss(log_probs, targets, [8, 8, 8], [3, 3, 3], 0)

    _assert_bool(res_out, ref_out)


# (log_probs shape, input_lengths, target_lengths); each length list is handed to
# the reference and the candidate unchanged.
_LENGTH_ROWS = [
    ((8, 3, 5), [8, 8, 8], [3, 3, 3]),
    ((8, 3, 5), [8, 8, 8], [0, 0, 0]),
    ((8, 3, 5), [8, 8, 8], [8, 8, 8]),
    ((8, 3, 5), [8, 8, 8], [3, 8, 2]),
    ((256, 1, 5), [256], [255]),
    ((1, 1, 5), [1], [1]),
    ((2, 4, 5), [2, 2, 2, 2], [1, 2, 0, 1]),
    ((8, 3, 5), [], []),
    ((8, 3, 5), [7, 8, 8], [3, 3, 3]),
    ((8, 3, 5), [8, 8, 8], [9, 3, 3]),
    ((256, 1, 5), [256], [256]),
]


@pytest.mark.use_cudnn_ctc_loss
@pytest.mark.parametrize(
    "log_probs_shape,input_lengths,target_lengths",
    _LENGTH_ROWS,
    ids=[
        "accepted",
        "zero_target_lengths",
        "target_equals_input",
        "mixed_target_lengths",
        "max_target_length_255",
        "single_step",
        "batch_of_four",
        "empty_lengths",
        "input_length_below_dim0",
        "target_longer_than_input",
        "target_length_256",
    ],
)
def test__use_cudnn_ctc_loss_lengths(log_probs_shape, input_lengths, target_lengths):
    log_probs = tu.make_input(torch.float32, log_probs_shape, ["-1", "1"])
    targets = _targets(log_probs_shape[1], 3)

    ref_log_probs = tu.to_reference(log_probs)
    ref_targets = tu.to_reference(targets)

    ref_out = torch.ops.aten._use_cudnn_ctc_loss(
        ref_log_probs, ref_targets, input_lengths, target_lengths, 0
    )
    res_out = flag_gems._use_cudnn_ctc_loss(
        log_probs, targets, input_lengths, target_lengths, 0
    )

    _assert_bool(res_out, ref_out)


@pytest.mark.use_cudnn_ctc_loss
@pytest.mark.parametrize("blank", [0, 1, -1, 255, 256])
def test__use_cudnn_ctc_loss_blank(blank):
    log_probs = tu.make_input(torch.float32, (8, 3, 5), ["-1", "1"])
    targets = _targets(3, 3)

    ref_log_probs = tu.to_reference(log_probs)
    ref_targets = tu.to_reference(targets)

    ref_out = torch.ops.aten._use_cudnn_ctc_loss(
        ref_log_probs, ref_targets, [8, 8, 8], [3, 3, 3], blank
    )
    res_out = flag_gems._use_cudnn_ctc_loss(
        log_probs, targets, [8, 8, 8], [3, 3, 3], blank
    )

    _assert_bool(res_out, ref_out)


# (T, N, C, input_lengths, target_lengths, lengths device, targets device, blank,
#  length dtype) for the .Tensor overload.
_TENSOR_ROWS = [
    (8, 3, 5, [8, 8, 8], [3, 3, 3], "cpu", "cpu", 0, torch.int32),
    (8, 3, 5, [8, 8, 8], [3, 3, 3], "device", "cpu", 0, torch.int32),
    (8, 3, 5, [8, 8, 8], [3, 3, 3], "cpu", "device", 0, torch.int32),
    (8, 3, 5, [7, 8, 8], [3, 3, 3], "cpu", "cpu", 0, torch.int32),
    (8, 3, 5, [8, 8, 8], [9, 3, 3], "cpu", "cpu", 0, torch.int32),
    (256, 1, 5, [256], [255], "cpu", "cpu", 0, torch.int32),
    (256, 1, 5, [256], [256], "cpu", "cpu", 0, torch.int32),
    (8, 3, 5, [8, 8, 8], [3, 3, 3], "cpu", "cpu", 1, torch.int32),
    (8, 3, 5, [8, 8, 8], [3, 3, 3], "cpu", "cpu", 0, torch.int64),
    (8, 3, 5, [8, 8, 8], [3, 3, 3], "cpu", "cpu", 0, torch.float32),
    (8, 3, 5, [], [], "cpu", "cpu", 0, torch.int32),
]


@pytest.mark.use_cudnn_ctc_loss
@pytest.mark.parametrize(
    "row",
    _TENSOR_ROWS,
    ids=[
        "cpu_lengths_cpu_targets",
        "device_lengths",
        "device_targets",
        "input_lengths_shorter_than_dim0",
        "target_longer_than_input",
        "max_target_length_255",
        "target_length_256",
        "blank_nonzero",
        "int64_lengths",
        "float32_lengths",
        "empty_lengths",
    ],
)
def test__use_cudnn_ctc_loss_tensor_lengths(row):
    (
        t,
        n,
        c,
        il_values,
        tl_values,
        lengths_where,
        targets_where,
        blank,
        lengths_dtype,
    ) = row
    log_probs = tu.make_input(torch.float32, (t, n, c), ["-1", "1"])
    targets = torch.zeros(n * 3, dtype=torch.int32, device=_where(targets_where))
    input_lengths = torch.tensor(
        il_values, dtype=lengths_dtype, device=_where(lengths_where)
    )
    target_lengths = torch.tensor(
        tl_values, dtype=lengths_dtype, device=_where(lengths_where)
    )

    ref_log_probs = tu.to_reference(log_probs)
    ref_targets = tu.to_reference(targets)
    ref_input_lengths = tu.to_reference(input_lengths)
    ref_target_lengths = tu.to_reference(target_lengths)

    ref_out = torch.ops.aten._use_cudnn_ctc_loss.Tensor(
        ref_log_probs, ref_targets, ref_input_lengths, ref_target_lengths, blank
    )
    res_out = flag_gems._use_cudnn_ctc_loss(
        log_probs, targets, input_lengths, target_lengths, blank
    )

    _assert_bool(res_out, ref_out)


# Positive special values are default-only. Only float32 applies: an fp8, int or
# half log_probs is rejected by the dtype test before any element is read, so
# nan/inf scenarios for those dtypes cannot reach a value-dependent path.
_SPECIAL_CASES = tu.selected_cases(
    tu.special_value_cases(_ACCEPTED_LOG_PROBS_DTYPES), quick=[]
)


@pytest.mark.use_cudnn_ctc_loss
@pytest.mark.parametrize("dtype, scenario", _SPECIAL_CASES)
def test__use_cudnn_ctc_loss_special_values(dtype, scenario):
    log_probs = tu.make_special_input(dtype, scenario).reshape(5, 1, 1)
    targets = _targets(1, 1)

    ref_log_probs = tu.to_reference(log_probs)
    ref_targets = tu.to_reference(targets)

    ref_out = torch.ops.aten._use_cudnn_ctc_loss(
        ref_log_probs, ref_targets, [5], [1], 0
    )
    res_out = flag_gems._use_cudnn_ctc_loss(log_probs, targets, [5], [1], 0)

    _assert_bool(res_out, ref_out)


_INVALID_CALLS = [
    "missing_blank",
    "extra_argument",
    "float_blank",
    "non_tensor_log_probs",
    "non_tensor_targets",
    "list_log_probs",
]


@pytest.mark.use_cudnn_ctc_loss
@pytest.mark.parametrize("case", _INVALID_CALLS)
def test__use_cudnn_ctc_loss_invalid_call(case):
    log_probs = tu.make_input(torch.float32, (8, 3, 5), ["-1", "1"])
    targets = _targets(3, 3)

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        if case == "missing_blank":
            flag_gems._use_cudnn_ctc_loss(log_probs, targets, [8, 8, 8], [3, 3, 3])
        elif case == "extra_argument":
            flag_gems._use_cudnn_ctc_loss(
                log_probs, targets, [8, 8, 8], [3, 3, 3], 0, 0
            )
        elif case == "float_blank":
            flag_gems._use_cudnn_ctc_loss(log_probs, targets, [8, 8, 8], [3, 3, 3], 1.5)
        elif case == "non_tensor_log_probs":
            flag_gems._use_cudnn_ctc_loss(3.14, targets, [8, 8, 8], [3, 3, 3], 0)
        elif case == "non_tensor_targets":
            flag_gems._use_cudnn_ctc_loss(log_probs, [0] * 9, [8, 8, 8], [3, 3, 3], 0)
        else:
            flag_gems._use_cudnn_ctc_loss(
                [[0.0] * 5] * 3, targets, [8, 8, 8], [3, 3, 3], 0
            )


@pytest.mark.use_cudnn_ctc_loss
@pytest.mark.parametrize("shape", [(), (8,), (8, 3), (8, 3, 5), (8, 3, 5, 2)])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("form", ["list", "tensor"])
def test_use_cudnn_ctc_loss_rank_and_backend_state(shape, enabled, form):
    log_probs = tu.make_input(torch.float32, shape, ["-1", "1"])
    targets = _targets(3, 3)
    input_lengths, target_lengths = [8] * 3, [3] * 3
    if form == "tensor":
        input_lengths = torch.tensor(input_lengths, dtype=torch.int32)
        target_lengths = torch.tensor(target_lengths, dtype=torch.int32)
    with torch.backends.cudnn.flags(enabled=enabled):
        ref = torch.ops.aten._use_cudnn_ctc_loss(
            log_probs, targets, input_lengths, target_lengths, 0
        )
        res = flag_gems._use_cudnn_ctc_loss(
            log_probs, targets, input_lengths, target_lengths, 0
        )
    _assert_bool(res, ref)

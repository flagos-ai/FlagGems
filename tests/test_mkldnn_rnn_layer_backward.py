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


"""Correctness tests for aten::mkldnn_rnn_layer_backward.

The oneDNN LSTM backward is registered for the CPU dispatch key and its last
argument is the opaque uint8 workspace the training forward allocates, so both
the reference and the injected candidate receive real CPU operands.
"""

import pytest
import torch

import flag_gems

from . import test_utils as tu

# The input is rank-3 (T, N, I) and the hidden size H drives every weight, bias,
# state and gradient shape, so each row is a (T, N, I, H) descriptor.  The spec's
# 0-dim, 1-dim, 2-dim and 5-dim shapes cannot describe a sequence operand.
LSTM_SHAPES = [
    (1, 1, 1, 1),
    (1, 256, 8, 4),
    (2, 19, 7, 8),
    (1024, 1, 2, 4),
    (20, 320, 15, 16),
    (16, 7, 57, 32),
    (16, 128, 64, 60),
]

QUICK_LSTM_SHAPES = [(1, 1, 1, 1), (2, 19, 7, 8)]

_SMALL_SHAPE = (2, 19, 7, 8)

_LSTM_MODE = 2

# Probed with the exact operator: float64/int32/int64/bool raise "get_mkldnn_dtype:
# unsupported data type", int8/uint8 raise "itensor_view_from_dense expects float,
# bfloat16 or half tensor input", and the fp8 types reach the CPU "add_stub"
# fallback.  Only these three dtypes have a kernel; the rest are negative rows.
SUPPORTED_DTYPES = [torch.float32, torch.bfloat16, torch.float16]


def _cpu(dtype, shape, value_range):
    # tu.make_input builds on flag_gems.device; this kernel is CPU-registered, so
    # its operands are CPU tensors, which is the operator's real device contract.
    return tu.make_input(dtype, shape, value_range).to("cpu")


def _is_undefined(tensor):
    # An undefined result has no elements: it arrives as None or as an empty
    # tensor, and its contents must never be compared.
    return tensor is None or tensor.numel() == 0


def _forward_backward_args(dtype, shape, value_range, **params):
    """Build the 23 positional arguments of the native backward call."""
    t, n, i, h = shape
    reverse = params.pop("reverse", False)
    mode = params.pop("mode", _LSTM_MODE)
    num_layers = params.pop("num_layers", 1)
    has_biases = params.pop("has_biases", True)
    train = params.pop("train", True)
    bidirectional = params.pop("bidirectional", False)
    batch_first = params.pop("batch_first", False)
    assert not params, params

    # The primitive behind this operator is single-layer and single-direction: it
    # describes the state as (1, 1, input.size(1), H) and the weights as
    # (4H, I) / (4H, H), so the state rows are the batch extent and the gate rows
    # are 4H with no layer or direction factor.  The multi-layer caller slices
    # layers and directions and applies batch_first before it reaches this
    # primitive, so num_layers, bidirectional and batch_first leave the operand
    # geometry alone: the input stays (T, N, I) and hx/cx stay (N, H).
    inp = _cpu(dtype, (t, n, i), value_range)
    weight_ih = _cpu(dtype, (4 * h, i), value_range)
    weight_hh = _cpu(dtype, (4 * h, h), value_range)
    bias_ih = _cpu(dtype, (4 * h,), value_range)
    bias_hh = _cpu(dtype, (4 * h,), value_range)
    hx = _cpu(dtype, (n, h), value_range)
    cx = _cpu(dtype, (n, h), value_range)

    # GradMode decides whether the forward allocates the backward workspace, whose
    # bytes cannot be derived from the other arguments.
    with torch.enable_grad():
        output, hy, cy, workspace = torch.ops.aten.mkldnn_rnn_layer(
            inp,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            hx,
            cx,
            reverse,
            [],
            mode,
            h,
            num_layers,
            has_biases,
            bidirectional,
            batch_first,
            train,
        )

    grad_output = _cpu(dtype, tuple(output.shape), value_range)
    grad_hy = _cpu(dtype, tuple(hy.shape), value_range)
    grad_cy = _cpu(dtype, tuple(cy.shape), value_range)

    return [
        inp,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
        hx,
        cx,
        output,
        hy,
        cy,
        grad_output,
        grad_hy,
        grad_cy,
        reverse,
        mode,
        h,
        num_layers,
        has_biases,
        train,
        bidirectional,
        [],
        batch_first,
        workspace,
    ]


def _clone_operands(operands):
    # Scalars and the batch-size list are immutable; every tensor is copied so the
    # candidate writes into its own storage instead of the reference's.
    return [item.clone() if torch.is_tensor(item) else item for item in operands]


def _operand_pair(dtype, shape, value_range, **params):
    """The reference operands and the candidate's own copy of them.

    The workspace is opaque oneDNN scratch handed over as an input, so it is
    copied like every other operand: both calls then start from identical values
    and the operator under test is the only difference between them.
    """
    ref_args = _forward_backward_args(dtype, shape, value_range, **params)
    return ref_args, _clone_operands(ref_args)


def _materialize(tensor):
    # ideep hands back blocked (oneDNN) tensors; their dense values are comparable
    # only after the layout itself has been asserted.
    if tensor.layout == torch.strided:
        return tensor
    return tensor.to_dense()


def _assert_outputs_close(res_out, ref_out):
    # diff_x, diff_weight_ih, diff_weight_hh, diff_bias_ih, diff_bias_hh,
    # diff_hx, diff_cx: every component the native operator defines is compared.
    assert len(res_out) == len(ref_out) == 7
    for res, ref in zip(res_out, ref_out):
        assert res.layout == ref.layout
        assert res.dtype == ref.dtype
        assert res.shape == ref.shape
        tu.assert_result_close(_materialize(res), _materialize(ref))


def _workspace_meta(workspace):
    # The workspace is oneDNN scratch: the native call rewrites its bytes, so only
    # the allocation metadata is expected to survive a call.
    return (
        workspace.data_ptr(),
        tuple(workspace.shape),
        workspace.dtype,
        workspace.numel(),
    )


def _assert_operands_unchanged(args, before, workspace_meta):
    # The operator may write into the buffers it is handed, so the candidate has
    # to leave its operands in their pre-call state.  Index 22 is the uint8
    # workspace, whose bytes are scratch (the native call rewrites them too).
    for index in range(13):
        tu.assert_result_equal(_materialize(args[index]), _materialize(before[index]))
    assert _workspace_meta(args[22]) == workspace_meta


@pytest.mark.mkldnn_rnn_layer_backward
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize(
    "shape", tu.selected_cases(LSTM_SHAPES, quick=QUICK_LSTM_SHAPES)
)
def test_mkldnn_rnn_layer_backward(dtype, value_range, shape):
    ref_args, res_args = _operand_pair(dtype, shape, value_range)
    workspace_meta = _workspace_meta(res_args[22])
    before = _clone_operands(res_args)

    ref_out = torch.ops.aten.mkldnn_rnn_layer_backward(*ref_args)
    res_out = flag_gems.mkldnn_rnn_layer_backward(*res_args)

    _assert_outputs_close(res_out, ref_out)
    _assert_operands_unchanged(res_args, before, workspace_meta)


SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(SUPPORTED_DTYPES), quick=[])


@pytest.mark.mkldnn_rnn_layer_backward
@pytest.mark.parametrize("dtype,scenario", SPECIAL_CASES)
def test_mkldnn_rnn_layer_backward_special_values(dtype, scenario):
    ref_args, res_args = _operand_pair(dtype, _SMALL_SHAPE, ["-1", "1"])
    # The value domain of a backward is its upstream gradient: a non-finite seed
    # has to reach the input, weight, bias and state gradients.
    payload = tu.make_special_input(dtype, scenario).to("cpu")
    for operand_set in (ref_args, res_args):
        operand_set[10].reshape(-1)[: payload.numel()] = payload

    ref_out = torch.ops.aten.mkldnn_rnn_layer_backward(*ref_args)
    res_out = flag_gems.mkldnn_rnn_layer_backward(*res_args)

    _assert_outputs_close(res_out, ref_out)


# Every parameter form is a cheap flag and none of them changes an operand shape,
# so quick keeps them all and the default level runs the same rows.
PARAM_ROWS = [
    {"reverse": True},
    {"batch_first": True},
    {"has_biases": False},
    {"train": False},
    {"num_layers": 2},
    {"bidirectional": True},
]

PARAM_CASES = tu.selected_cases(PARAM_ROWS, quick=PARAM_ROWS)


@pytest.mark.mkldnn_rnn_layer_backward
@pytest.mark.parametrize("params", PARAM_CASES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_mkldnn_rnn_layer_backward_params(params, dtype):
    # The forward is built with the same parameters, so the workspace it hands to
    # the backward belongs to this call.  All seven components are compared.
    ref_args, res_args = _operand_pair(dtype, _SMALL_SHAPE, ["-1", "1"], **params)
    workspace_meta = _workspace_meta(res_args[22])
    before = _clone_operands(res_args)

    ref_out = torch.ops.aten.mkldnn_rnn_layer_backward(*ref_args)
    res_out = flag_gems.mkldnn_rnn_layer_backward(*res_args)

    _assert_outputs_close(res_out, ref_out)
    _assert_operands_unchanged(res_args, before, workspace_meta)


_OUT_SENTINEL = -9876.5


def _out_buffers(shape):
    # Every buffer is prefilled so a component the candidate never writes cannot
    # pass the value comparison below.
    t, n, i, h = shape
    return [
        torch.full((t, n, i), _OUT_SENTINEL, dtype=torch.float32),
        torch.full((4 * h, i), _OUT_SENTINEL, dtype=torch.float32),
        torch.full((4 * h, h), _OUT_SENTINEL, dtype=torch.float32),
        torch.full((4 * h,), _OUT_SENTINEL, dtype=torch.float32),
        torch.full((4 * h,), _OUT_SENTINEL, dtype=torch.float32),
        torch.full((n, h), _OUT_SENTINEL, dtype=torch.float32),
        torch.full((n, h), _OUT_SENTINEL, dtype=torch.float32),
    ]


@pytest.mark.mkldnn_rnn_layer_backward
def test_mkldnn_rnn_layer_backward_out():
    ref_args, res_args = _operand_pair(torch.float32, _SMALL_SHAPE, ["-1", "1"])
    workspace_meta = _workspace_meta(res_args[22])
    before = _clone_operands(res_args)
    ref_buffers = _out_buffers(_SMALL_SHAPE)
    res_buffers = _out_buffers(_SMALL_SHAPE)

    ref_out = torch.ops.aten.mkldnn_rnn_layer_backward.out(
        *ref_args, **{f"out{index}": buffer for index, buffer in enumerate(ref_buffers)}
    )
    res_out = flag_gems.mkldnn_rnn_layer_backward(
        *res_args, **{f"out{index}": buffer for index, buffer in enumerate(res_buffers)}
    )

    assert len(res_out) == len(ref_out) == 7
    for index, (res, ref) in enumerate(zip(res_out, ref_out)):
        # return identity: the overload hands back the caller's own buffer
        assert res is res_buffers[index]
        assert res.layout == ref.layout
        assert res.dtype == torch.float32
        assert res.shape == ref.shape
        tu.assert_result_close(_materialize(res), _materialize(ref))

    _assert_operands_unchanged(res_args, before, workspace_meta)


@pytest.mark.mkldnn_rnn_layer_backward
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_mkldnn_rnn_layer_backward_without_gradients(dtype):
    ref_args, res_args = _operand_pair(dtype, _SMALL_SHAPE, ["-1", "1"])
    # grad_output / grad_hy / grad_cy are optional; with no gradient at all the
    # native contract returns seven undefined results, whose contents must never
    # be compared.  The None values are positional because a keyword None bypasses
    # the schema conversion of an optional argument.
    for operand_set in (ref_args, res_args):
        operand_set[10] = operand_set[11] = operand_set[12] = None

    ref_out = torch.ops.aten.mkldnn_rnn_layer_backward(*ref_args)
    res_out = flag_gems.mkldnn_rnn_layer_backward(*res_args)

    assert len(res_out) == len(ref_out) == 7
    assert [_is_undefined(tensor) for tensor in ref_out] == [True] * 7
    assert [_is_undefined(tensor) for tensor in res_out] == [True] * 7


@pytest.mark.mkldnn_rnn_layer_backward
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize(
    "present",
    [
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
    ],
)
def test_mkldnn_rnn_layer_backward_optional_gradients(dtype, present):
    ref_args, res_args = _operand_pair(dtype, _SMALL_SHAPE, ["-1", "1"])
    for index, supplied in enumerate(present, 10):
        if not supplied:
            ref_args[index] = res_args[index] = None

    ref_out = torch.ops.aten.mkldnn_rnn_layer_backward(*ref_args)
    res_out = flag_gems.mkldnn_rnn_layer_backward(*res_args)

    _assert_outputs_close(res_out, ref_out)


# Negative rows are collected in both modes.  Indices 0..12 are the floating
# operands and index 22 is the uint8 workspace.
@pytest.mark.mkldnn_rnn_layer_backward
@pytest.mark.parametrize(
    "dtype",
    [
        torch.int8,
        torch.uint8,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.int32,
        torch.int64,
        torch.bool,
        torch.float64,
    ],
)
def test_mkldnn_rnn_layer_backward_unsupported_dtype(dtype):
    args = _forward_backward_args(torch.float32, _SMALL_SHAPE, ["-1", "1"])
    for index in range(13):
        args[index] = args[index].to(dtype)

    with pytest.raises(RuntimeError):
        flag_gems.mkldnn_rnn_layer_backward(*args)


@pytest.mark.mkldnn_rnn_layer_backward
def test_mkldnn_rnn_layer_backward_invalid_input_dim():
    args = _forward_backward_args(torch.float32, _SMALL_SHAPE, ["-1", "1"])
    args[0] = args[0].reshape(-1)

    # The sequence descriptor reads input.size(1) without a rank check, so a
    # rank-1 input surfaces as IndexError rather than RuntimeError.
    with pytest.raises(IndexError):
        flag_gems.mkldnn_rnn_layer_backward(*args)


@pytest.mark.mkldnn_rnn_layer_backward
@pytest.mark.parametrize("mode", [1, 7])
def test_mkldnn_rnn_layer_backward_invalid_mode(mode):
    args = _forward_backward_args(torch.float32, _SMALL_SHAPE, ["-1", "1"])
    # mode reaches the oneDNN primitive preparation, where only the LSTM mode (2)
    # matches this workload's gate weights.
    args[14] = mode

    with pytest.raises(RuntimeError):
        flag_gems.mkldnn_rnn_layer_backward(*args)

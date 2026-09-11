# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE, TO_CPU

DTYPES = utils.ALL_FLOAT_DTYPES
SHAPES = [(1, 4), (4, 17), (33, 129), (2, 257)]
if QUICK_MODE:
    SHAPES = [(4, 17)]


def _make_inputs(batch_size, hidden_size, dtype, noncontiguous=False):
    device = flag_gems.device
    if noncontiguous:
        input_gates = torch.randn(
            3 * hidden_size, batch_size, dtype=dtype, device=device
        ).T
        hidden_gates = torch.randn(
            3 * hidden_size, batch_size, dtype=dtype, device=device
        ).T
        hx = torch.randn(hidden_size, batch_size, dtype=dtype, device=device).T
    else:
        input_gates = torch.randn(
            batch_size, 3 * hidden_size, dtype=dtype, device=device
        )
        hidden_gates = torch.randn_like(input_gates)
        hx = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    return input_gates, hidden_gates, hx


def _manual_reference(input_gates, hidden_gates, hx, input_bias=None, hidden_bias=None):
    if input_bias is not None and hidden_bias is None:
        raise RuntimeError("hidden_bias must be defined when input_bias is defined")
    dtype = input_gates.dtype
    compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    input_reset, input_update, input_new = (
        value.to(compute_dtype) for value in input_gates.chunk(3, dim=1)
    )
    hidden_reset, hidden_update, hidden_new = (
        value.to(compute_dtype) for value in hidden_gates.chunk(3, dim=1)
    )
    if input_bias is not None:
        bias_reset, bias_update, bias_new = (
            value.to(compute_dtype) for value in input_bias.chunk(3)
        )
        hidden_bias_reset, hidden_bias_update, hidden_bias_new = (
            value.to(compute_dtype) for value in hidden_bias.chunk(3)
        )
        input_reset = input_reset + bias_reset
        input_update = input_update + bias_update
        input_new = input_new + bias_new
        hidden_reset = hidden_reset + hidden_bias_reset
        hidden_update = hidden_update + hidden_bias_update
        hidden_new = hidden_new + hidden_bias_new
    reset_gate = torch.sigmoid(input_reset + hidden_reset)
    update_gate = torch.sigmoid(input_update + hidden_update)
    new_gate = torch.tanh(input_new + reset_gate * hidden_new)
    hx_compute = hx.to(compute_dtype)
    hy = new_gate + update_gate * (hx_compute - new_gate)
    workspace = torch.cat(
        (reset_gate, update_gate, new_gate, hx_compute, hidden_new), dim=1
    )
    return hy.to(dtype), workspace.to(dtype)


def _reference(input_gates, hidden_gates, hx, input_bias=None, hidden_bias=None):
    args = [utils.to_reference(value) for value in (input_gates, hidden_gates, hx)]
    ref_input_bias = utils.to_reference(input_bias) if input_bias is not None else None
    ref_hidden_bias = (
        utils.to_reference(hidden_bias) if hidden_bias is not None else None
    )
    if args[0].device.type == "cpu":
        return _manual_reference(
            *args, input_bias=ref_input_bias, hidden_bias=ref_hidden_bias
        )
    return torch.ops.aten._thnn_fused_gru_cell(*args, ref_input_bias, ref_hidden_bias)


def _assert_close(result, reference, dtype):
    for actual, expected in zip(result, reference):
        utils.gems_assert_close(actual, expected, dtype)


@pytest.mark.thnn_fused_gru_cell
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("with_bias", [False, True])
def test_accuracy_thnn_fused_gru_cell(shape, dtype, with_bias):
    batch_size, hidden_size = shape
    input_gates, hidden_gates, hx = _make_inputs(batch_size, hidden_size, dtype)
    input_bias = None
    hidden_bias = None
    if with_bias:
        input_bias = torch.randn(3 * hidden_size, dtype=dtype, device=flag_gems.device)
        hidden_bias = torch.randn_like(input_bias)
    reference = _reference(input_gates, hidden_gates, hx, input_bias, hidden_bias)

    result = flag_gems._thnn_fused_gru_cell(
        input_gates, hidden_gates, hx, input_bias, hidden_bias
    )

    _assert_close(result, reference, dtype)


@pytest.mark.thnn_fused_gru_cell
@pytest.mark.parametrize("dtype", DTYPES)
def test_accuracy_thnn_fused_gru_cell_noncontiguous(dtype):
    batch_size, hidden_size = 5, 19
    input_gates, hidden_gates, hx = _make_inputs(
        batch_size, hidden_size, dtype, noncontiguous=True
    )
    input_bias = torch.randn(6 * hidden_size, dtype=dtype, device=flag_gems.device)[::2]
    hidden_bias = torch.randn(6 * hidden_size, dtype=dtype, device=flag_gems.device)[
        ::2
    ]
    reference = _reference(input_gates, hidden_gates, hx, input_bias, hidden_bias)

    result = flag_gems._thnn_fused_gru_cell(
        input_gates, hidden_gates, hx, input_bias, hidden_bias
    )

    _assert_close(result, reference, dtype)


@pytest.mark.thnn_fused_gru_cell
@pytest.mark.parametrize("shape", [(0, 8), (2, 0)])
def test_accuracy_thnn_fused_gru_cell_empty(shape):
    input_gates, hidden_gates, hx = _make_inputs(*shape, torch.float32)
    reference = _reference(input_gates, hidden_gates, hx)
    result = flag_gems._thnn_fused_gru_cell(input_gates, hidden_gates, hx)
    _assert_close(result, reference, torch.float32)


@pytest.mark.thnn_fused_gru_cell
def test_accuracy_thnn_fused_gru_cell_hidden_bias_without_input_bias():
    batch_size, hidden_size = 3, 11
    input_gates, hidden_gates, hx = _make_inputs(batch_size, hidden_size, torch.float32)
    hidden_bias = torch.randn(3 * hidden_size, device=flag_gems.device)
    reference = _reference(input_gates, hidden_gates, hx, hidden_bias=hidden_bias)
    result = flag_gems._thnn_fused_gru_cell(
        input_gates, hidden_gates, hx, hidden_bias=hidden_bias
    )
    _assert_close(result, reference, torch.float32)


@pytest.mark.thnn_fused_gru_cell
def test_accuracy_thnn_fused_gru_cell_rejects_input_bias_only():
    input_gates, hidden_gates, hx = _make_inputs(2, 7, torch.float32)
    input_bias = torch.randn(21, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems._thnn_fused_gru_cell(
            input_gates, hidden_gates, hx, input_bias=input_bias
        )


@pytest.mark.skipif(TO_CPU, reason="native fused GRU cell is CUDA-only")
@pytest.mark.thnn_fused_gru_cell
@pytest.mark.parametrize("dtype", [torch.int32, torch.bool, torch.complex64])
def test_thnn_fused_gru_cell_rejects_unsupported_dtype(dtype):
    shape = (2, 21)
    if dtype == torch.bool:
        input_gates = torch.randint(
            0, 2, shape, dtype=torch.int32, device=flag_gems.device
        ).bool()
    elif dtype.is_complex:
        input_gates = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        input_gates = torch.randint(-4, 5, shape, dtype=dtype, device=flag_gems.device)
    hidden_gates = torch.zeros_like(input_gates)
    hx = torch.zeros((2, 7), dtype=dtype, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        torch.ops.aten._thnn_fused_gru_cell(input_gates, hidden_gates, hx)
    with pytest.raises(RuntimeError):
        flag_gems._thnn_fused_gru_cell(input_gates, hidden_gates, hx)


def _make_noncontiguous_outputs(batch_size, hidden_size, dtype):
    out0 = torch.empty(hidden_size, batch_size, dtype=dtype, device=flag_gems.device).T
    out1 = torch.empty(
        5 * hidden_size, batch_size, dtype=dtype, device=flag_gems.device
    ).T
    return out0, out1


@pytest.mark.thnn_fused_gru_cell_out
@pytest.mark.parametrize("dtype", DTYPES)
def test_accuracy_thnn_fused_gru_cell_out(dtype):
    batch_size, hidden_size = 7, 37
    input_gates, hidden_gates, hx = _make_inputs(batch_size, hidden_size, dtype)
    input_bias = torch.randn(3 * hidden_size, dtype=dtype, device=flag_gems.device)
    hidden_bias = torch.randn_like(input_bias)
    reference = _reference(input_gates, hidden_gates, hx, input_bias, hidden_bias)
    outputs = _make_noncontiguous_outputs(batch_size, hidden_size, dtype)

    result = flag_gems._thnn_fused_gru_cell_out(
        input_gates,
        hidden_gates,
        hx,
        input_bias,
        hidden_bias,
        out0=outputs[0],
        out1=outputs[1],
    )

    assert result[0] is outputs[0] and result[1] is outputs[1]
    _assert_close(result, reference, dtype)


@pytest.mark.thnn_fused_gru_cell_out
def test_accuracy_thnn_fused_gru_cell_out_resize():
    input_gates, hidden_gates, hx = _make_inputs(2, 7, torch.float32)
    reference = _reference(input_gates, hidden_gates, hx)
    outputs = tuple(torch.empty(0, device=flag_gems.device) for _ in range(2))
    result = flag_gems._thnn_fused_gru_cell_out(
        input_gates, hidden_gates, hx, out0=outputs[0], out1=outputs[1]
    )
    assert result[0] is outputs[0] and result[1] is outputs[1]
    assert tuple(outputs[0].shape) == (2, 7)
    assert tuple(outputs[1].shape) == (2, 35)
    _assert_close(result, reference, torch.float32)


@pytest.mark.thnn_fused_gru_cell_out
@pytest.mark.parametrize("output_index", [0, 1])
def test_thnn_fused_gru_cell_out_rejects_invalid_dtype(output_index):
    input_gates, hidden_gates, hx = _make_inputs(2, 7, torch.float32)
    outputs = [
        torch.empty((2, 7), dtype=torch.float32, device=flag_gems.device),
        torch.empty((2, 35), dtype=torch.float32, device=flag_gems.device),
    ]
    outputs[output_index] = outputs[output_index].to(torch.float64)
    with pytest.raises(RuntimeError):
        flag_gems._thnn_fused_gru_cell_out(
            input_gates,
            hidden_gates,
            hx,
            out0=outputs[0],
            out1=outputs[1],
        )


@pytest.mark.thnn_fused_gru_cell_out
@pytest.mark.parametrize("output_index", [0, 1])
def test_thnn_fused_gru_cell_out_rejects_invalid_device(output_index):
    input_gates, hidden_gates, hx = _make_inputs(2, 7, torch.float32)
    outputs = [
        torch.empty((2, 7), device=flag_gems.device),
        torch.empty((2, 35), device=flag_gems.device),
    ]
    wrong_device = "meta" if input_gates.device.type == "cpu" else "cpu"
    outputs[output_index] = torch.empty(
        outputs[output_index].shape, device=wrong_device
    )
    with pytest.raises(RuntimeError):
        flag_gems._thnn_fused_gru_cell_out(
            input_gates,
            hidden_gates,
            hx,
            out0=outputs[0],
            out1=outputs[1],
        )


@pytest.mark.thnn_fused_gru_cell_out
def test_thnn_fused_gru_cell_out_rejects_overlapping_outputs():
    input_gates, hidden_gates, hx = _make_inputs(2, 7, torch.float32)
    storage = torch.empty((2, 35), device=flag_gems.device)
    out0 = storage[:, :7]
    with pytest.raises(RuntimeError):
        flag_gems._thnn_fused_gru_cell_out(
            input_gates, hidden_gates, hx, out0=out0, out1=storage
        )


@pytest.mark.skipif(TO_CPU, reason="native fused GRU cell is CUDA-only")
@pytest.mark.thnn_fused_gru_cell
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("with_bias", [False, True])
def test_thnn_fused_gru_cell_backward_end_to_end(dtype, with_bias):
    batch_size, hidden_size = 5, 19
    input_gates, hidden_gates, hx = _make_inputs(batch_size, hidden_size, dtype)
    input_bias = None
    hidden_bias = None
    if with_bias:
        input_bias = torch.randn(3 * hidden_size, dtype=dtype, device=flag_gems.device)
        hidden_bias = torch.randn_like(input_bias)

    ref_hy, ref_workspace = torch.ops.aten._thnn_fused_gru_cell(
        input_gates, hidden_gates, hx, input_bias, hidden_bias
    )
    hy, workspace = flag_gems._thnn_fused_gru_cell(
        input_gates, hidden_gates, hx, input_bias, hidden_bias
    )
    grad_hy = torch.randn_like(hy)
    reference = torch.ops.aten._thnn_fused_gru_cell_backward(
        grad_hy, ref_workspace, with_bias
    )
    result = torch.ops.aten._thnn_fused_gru_cell_backward(grad_hy, workspace, with_bias)

    utils.gems_assert_close(hy, ref_hy, dtype)
    for actual, expected in zip(result, reference):
        if expected is None:
            assert actual is None
        else:
            utils.gems_assert_close(actual, expected, dtype)

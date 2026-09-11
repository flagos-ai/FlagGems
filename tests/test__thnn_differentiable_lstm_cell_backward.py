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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

DTYPES = utils.ALL_FLOAT_DTYPES
SHAPES = [(1, 4), (2, 17), (8, 64), (33, 129)]
if cfg.QUICK_MODE:
    SHAPES = [(2, 17)]

BIAS_MODES = [(False, False), (True, False), (False, True), (True, True)]
GRAD_MODES = [(False, False), (True, False), (False, True), (True, True)]


def _make_state(batch_size, hidden_size, dtype, *, noncontiguous=False):
    device = flag_gems.device

    def make(columns):
        if noncontiguous:
            return torch.randn(columns, batch_size, dtype=dtype, device=device).T
        return torch.randn(batch_size, columns, dtype=dtype, device=device)

    return (
        make(hidden_size),
        make(hidden_size),
        make(4 * hidden_size),
        make(4 * hidden_size),
    )


def _make_bias(hidden_size, dtype, enabled, *, noncontiguous=False):
    if not enabled:
        return None
    size = 4 * hidden_size
    if noncontiguous:
        return torch.randn(size * 2, dtype=dtype, device=flag_gems.device)[::2]
    return torch.randn(size, dtype=dtype, device=flag_gems.device)


def _make_args(
    batch_size,
    hidden_size,
    dtype,
    *,
    has_grad_hy=True,
    has_grad_cy=True,
    has_input_bias=True,
    has_hidden_bias=True,
    noncontiguous=False,
):
    cx, cy, input_gates, hidden_gates = _make_state(
        batch_size, hidden_size, dtype, noncontiguous=noncontiguous
    )
    grad_hy = (
        _make_state(batch_size, hidden_size, dtype, noncontiguous=noncontiguous)[0]
        if has_grad_hy
        else None
    )
    grad_cy = (
        _make_state(batch_size, hidden_size, dtype, noncontiguous=noncontiguous)[0]
        if has_grad_cy
        else None
    )
    input_bias = _make_bias(
        hidden_size, dtype, has_input_bias, noncontiguous=noncontiguous
    )
    hidden_bias = _make_bias(
        hidden_size, dtype, has_hidden_bias, noncontiguous=noncontiguous
    )
    return (
        grad_hy,
        grad_cy,
        input_gates,
        hidden_gates,
        input_bias,
        hidden_bias,
        cx,
        cy,
    )


def _reference_args(args):
    return tuple(None if value is None else utils.to_reference(value) for value in args)


def _assert_outputs_close(result, reference, dtype, batch_size):
    assert len(result) == len(reference) == 5
    if result[0] is not None:
        assert result[0] is result[1]
    if result[3] is not None:
        assert result[3] is result[4]

    if dtype == torch.float64:
        base_atol = 1e-8
    elif dtype == torch.bfloat16:
        base_atol = 3e-3
    elif dtype == torch.float16:
        base_atol = 1e-3
    else:
        base_atol = 1e-4
    for actual, expected in zip(result, reference):
        if expected is None:
            assert actual is None
            continue
        assert actual is not None
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert actual.is_contiguous()
        utils.gems_assert_close(
            actual,
            expected,
            dtype,
            equal_nan=True,
            reduce_dim=max(batch_size, 1),
            atol=base_atol,
        )


@pytest.mark.thnn_differentiable_lstm_cell_backward
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("has_input_bias,has_hidden_bias", BIAS_MODES)
def test_thnn_differentiable_lstm_cell_backward(
    dtype, shape, has_input_bias, has_hidden_bias
):
    batch_size, hidden_size = shape
    args = _make_args(
        batch_size,
        hidden_size,
        dtype,
        has_input_bias=has_input_bias,
        has_hidden_bias=has_hidden_bias,
    )
    reference = torch.ops.aten._thnn_differentiable_lstm_cell_backward(
        *_reference_args(args)
    )

    with flag_gems.use_gems():
        result = torch.ops.aten._thnn_differentiable_lstm_cell_backward(*args)

    _assert_outputs_close(result, reference, dtype, batch_size)


@pytest.mark.thnn_differentiable_lstm_cell_backward
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("has_grad_hy,has_grad_cy", GRAD_MODES)
def test_thnn_differentiable_lstm_cell_backward_optional_grads(
    dtype, has_grad_hy, has_grad_cy
):
    args = _make_args(
        5,
        19,
        dtype,
        has_grad_hy=has_grad_hy,
        has_grad_cy=has_grad_cy,
    )
    reference = torch.ops.aten._thnn_differentiable_lstm_cell_backward(
        *_reference_args(args)
    )

    result = flag_gems._thnn_differentiable_lstm_cell_backward(*args)

    _assert_outputs_close(result, reference, dtype, 5)


@pytest.mark.thnn_differentiable_lstm_cell_backward
@pytest.mark.parametrize("dtype", DTYPES)
def test_thnn_differentiable_lstm_cell_backward_noncontiguous(dtype):
    args = _make_args(5, 19, dtype, noncontiguous=True)
    assert all(value is None or not value.is_contiguous() for value in args)
    reference = torch.ops.aten._thnn_differentiable_lstm_cell_backward(
        *_reference_args(args)
    )

    result = flag_gems._thnn_differentiable_lstm_cell_backward(*args)

    _assert_outputs_close(result, reference, dtype, 5)


@pytest.mark.thnn_differentiable_lstm_cell_backward
@pytest.mark.parametrize("shape", [(0, 8), (2, 0)])
def test_thnn_differentiable_lstm_cell_backward_empty(shape):
    batch_size, hidden_size = shape
    args = _make_args(batch_size, hidden_size, torch.float32)
    reference = torch.ops.aten._thnn_differentiable_lstm_cell_backward(
        *_reference_args(args)
    )

    with flag_gems.use_gems():
        result = torch.ops.aten._thnn_differentiable_lstm_cell_backward(*args)

    _assert_outputs_close(result, reference, torch.float32, batch_size)


@pytest.mark.thnn_differentiable_lstm_cell_backward
def test_thnn_differentiable_lstm_cell_backward_special_values():
    args = list(_make_args(1, 4, torch.float32, has_input_bias=False))
    args[0] = torch.tensor(
        [[float("inf"), -float("inf"), 0.0, -0.0]], device=flag_gems.device
    )
    args[1] = torch.tensor([[1.0, -1.0, float("nan"), 0.0]], device=flag_gems.device)
    special = torch.tensor(
        [
            float("inf"),
            -float("inf"),
            float("nan"),
            0.0,
            -0.0,
            1.0,
            -1.0,
            20.0,
            -20.0,
            0.5,
            -0.5,
            2.0,
            -2.0,
            3.0,
            -3.0,
            0.0,
        ],
        device=flag_gems.device,
    )
    args[2] = special.reshape(1, 16)
    args[3] = torch.zeros_like(args[2])
    reference = torch.ops.aten._thnn_differentiable_lstm_cell_backward(
        *_reference_args(args)
    )

    result = flag_gems._thnn_differentiable_lstm_cell_backward(*args)

    _assert_outputs_close(result, reference, torch.float32, 1)


@pytest.mark.thnn_differentiable_lstm_cell_backward
def test_thnn_differentiable_lstm_cell_backward_is_differentiable():
    args = _make_args(2, 4, torch.float32)
    runtime_args = tuple(value.detach().requires_grad_() for value in args)
    reference_args = tuple(
        utils.to_reference(value).detach().requires_grad_() for value in args
    )
    weights = (
        torch.randn(2, 16, device=flag_gems.device),
        torch.randn(2, 4, device=flag_gems.device),
        torch.randn(16, device=flag_gems.device),
    )
    reference_weights = tuple(utils.to_reference(value) for value in weights)

    reference_outputs = torch.ops.aten._thnn_differentiable_lstm_cell_backward(
        *reference_args
    )
    reference_loss = sum(
        (reference_outputs[index] * weight).sum()
        for index, weight in zip((0, 2, 3), reference_weights)
    )
    reference_grads = torch.autograd.grad(reference_loss, reference_args)

    outputs = flag_gems._thnn_differentiable_lstm_cell_backward(*runtime_args)
    loss = sum(
        (outputs[index] * weight).sum() for index, weight in zip((0, 2, 3), weights)
    )
    grads = torch.autograd.grad(loss, runtime_args)

    for actual, expected in zip(grads, reference_grads):
        utils.gems_assert_close(actual, expected, torch.float32, atol=1e-4)


@pytest.mark.thnn_differentiable_lstm_cell_backward
def test_thnn_differentiable_lstm_cell_backward_invalid_shape():
    args = list(_make_args(2, 8, torch.float32))
    args[2] = torch.randn(2, 31, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems._thnn_differentiable_lstm_cell_backward(*args)


@pytest.mark.thnn_differentiable_lstm_cell_backward
def test_thnn_differentiable_lstm_cell_backward_all_grads_none_short_circuit():
    result = flag_gems._thnn_differentiable_lstm_cell_backward(
        None, None, None, None, None, None, None, None
    )
    assert result == (None, None, None, None, None)

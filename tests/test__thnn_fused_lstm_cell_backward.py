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

GRAD_MODES = [(True, True), (True, False), (False, True), (False, False)]


def _make_args(
    batch_size,
    hidden_size,
    dtype,
    *,
    has_grad_hy=True,
    has_grad_cy=True,
    has_bias=True,
    noncontiguous=False,
):
    device = flag_gems.device
    input_gates = torch.randn(batch_size, 4 * hidden_size, dtype=dtype, device=device)
    hidden_gates = torch.randn_like(input_gates)
    cx = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    bias = (
        torch.randn(4 * hidden_size, dtype=dtype, device=device) if has_bias else None
    )
    with torch.no_grad():
        hy, cy, workspace = torch.ops.aten._thnn_fused_lstm_cell(
            input_gates, hidden_gates, cx, bias, bias
        )
    grad_hy = torch.randn_like(hy) if has_grad_hy else None
    grad_cy = torch.randn_like(cy) if has_grad_cy else None

    if noncontiguous:

        def make_noncontiguous(tensor):
            return None if tensor is None else tensor.T.contiguous().T

        grad_hy = make_noncontiguous(grad_hy)
        grad_cy = make_noncontiguous(grad_cy)
        cx = make_noncontiguous(cx)
        cy = make_noncontiguous(cy)
        workspace = make_noncontiguous(workspace)

    return grad_hy, grad_cy, cx, cy, workspace, has_bias


def _reference_args(args):
    return tuple(
        utils.to_reference(value) if isinstance(value, torch.Tensor) else value
        for value in args
    )


def _assert_outputs(result, reference, dtype, batch_size):
    assert len(result) == len(reference) == 5
    assert result[0] is result[1]
    assert result[3] is result[4]
    atol = {
        torch.float16: 1e-3,
        torch.bfloat16: 5e-3,
        torch.float32: 1e-5,
        torch.float64: 1e-10,
    }[dtype]
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
            atol=atol,
        )


@pytest.mark.skipif(cfg.TO_CPU, reason="native fused LSTM cell is CUDA-only")
@pytest.mark.thnn_fused_lstm_cell_backward
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("has_bias", [False, True])
def test_thnn_fused_lstm_cell_backward(dtype, shape, has_bias):
    args = _make_args(*shape, dtype, has_bias=has_bias)
    with torch.no_grad():
        reference = torch.ops.aten._thnn_fused_lstm_cell_backward(
            *_reference_args(args)
        )
        with flag_gems.use_gems():
            result = torch.ops.aten._thnn_fused_lstm_cell_backward(*args)
    _assert_outputs(result, reference, dtype, shape[0])


@pytest.mark.skipif(cfg.TO_CPU, reason="native fused LSTM cell is CUDA-only")
@pytest.mark.thnn_fused_lstm_cell_backward
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("has_grad_hy,has_grad_cy", GRAD_MODES)
def test_thnn_fused_lstm_cell_backward_optional_grads(dtype, has_grad_hy, has_grad_cy):
    args = _make_args(
        5,
        19,
        dtype,
        has_grad_hy=has_grad_hy,
        has_grad_cy=has_grad_cy,
    )
    with torch.no_grad():
        reference = torch.ops.aten._thnn_fused_lstm_cell_backward(
            *_reference_args(args)
        )
        result = flag_gems._thnn_fused_lstm_cell_backward(*args)
    _assert_outputs(result, reference, dtype, 5)


@pytest.mark.skipif(cfg.TO_CPU, reason="native fused LSTM cell is CUDA-only")
@pytest.mark.thnn_fused_lstm_cell_backward
@pytest.mark.parametrize("dtype", DTYPES)
def test_thnn_fused_lstm_cell_backward_noncontiguous(dtype):
    args = _make_args(5, 19, dtype, noncontiguous=True)
    assert all(
        not value.is_contiguous()
        for value in args[:-1]
        if isinstance(value, torch.Tensor)
    )
    with torch.no_grad():
        reference = torch.ops.aten._thnn_fused_lstm_cell_backward(
            *_reference_args(args)
        )
        result = flag_gems._thnn_fused_lstm_cell_backward(*args)
    _assert_outputs(result, reference, dtype, 5)


@pytest.mark.skipif(cfg.TO_CPU, reason="native fused LSTM cell is CUDA-only")
@pytest.mark.thnn_fused_lstm_cell_backward
@pytest.mark.parametrize("shape", [(0, 8), (2, 0)])
def test_thnn_fused_lstm_cell_backward_empty(shape):
    args = _make_args(*shape, torch.float32)
    with torch.no_grad():
        reference = torch.ops.aten._thnn_fused_lstm_cell_backward(
            *_reference_args(args)
        )
        result = flag_gems._thnn_fused_lstm_cell_backward(*args)
    _assert_outputs(result, reference, torch.float32, shape[0])


@pytest.mark.skipif(cfg.TO_CPU, reason="native fused LSTM cell is CUDA-only")
@pytest.mark.thnn_fused_lstm_cell_backward
def test_thnn_fused_lstm_cell_backward_special_values():
    device = flag_gems.device
    grad_hy = torch.tensor([[float("inf"), -float("inf"), 0.0, -0.0]], device=device)
    grad_cy = torch.tensor([[1.0, -1.0, float("nan"), 0.0]], device=device)
    cx = torch.tensor([[float("inf"), -float("inf"), 0.0, -0.0]], device=device)
    cy = torch.tensor([[1.0, -1.0, float("nan"), 0.0]], device=device)
    workspace = torch.tensor(
        [
            float("nan"),
            float("inf"),
            -float("inf"),
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            -0.5,
            2.0,
            -2.0,
            0.25,
            0.75,
            0.0,
            1.0,
            -1.0,
        ],
        device=device,
    ).reshape(1, 16)
    args = grad_hy, grad_cy, cx, cy, workspace, True

    with torch.no_grad():
        reference = torch.ops.aten._thnn_fused_lstm_cell_backward(
            *_reference_args(args)
        )
        result = flag_gems._thnn_fused_lstm_cell_backward(*args)
    _assert_outputs(result, reference, torch.float32, 1)

    for actual, expected in zip(result, reference):
        valid = ~torch.isnan(expected)
        utils.gems_assert_equal(
            torch.signbit(actual)[valid], torch.signbit(expected)[valid]
        )


@pytest.mark.skipif(cfg.TO_CPU, reason="native fused LSTM cell is CUDA-only")
@pytest.mark.thnn_fused_lstm_cell_backward
def test_thnn_fused_lstm_cell_backward_requires_no_grad():
    args = _make_args(2, 4, torch.float32)
    with pytest.raises(RuntimeError, match="grad mode disabled"):
        flag_gems._thnn_fused_lstm_cell_backward(*args)

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
from .accuracy_utils import _SKIP_FP64

DTYPES = utils.ALL_FLOAT_DTYPES
SHAPES = [(1, 4), (2, 17), (8, 64), (33, 129), (513, 129)]
if cfg.QUICK_MODE:
    SHAPES = [(2, 17)]

GRAD_MODES = [(True, True), (True, False), (False, True), (False, False)]


def _use_cpu_reference(sample_tensor: torch.Tensor) -> bool:
    return cfg.TO_CPU or sample_tensor.device.type != "cuda"


def _cpu_forward_ref(
    input_gates,
    hidden_gates,
    cx,
    input_bias=None,
    hidden_bias=None,
    *,
    out_device="cpu",
):
    dtype = input_gates.dtype
    acc_dtype = torch.float64
    batch_size, hidden_size = cx.shape
    if batch_size == 0 or hidden_size == 0:
        return (
            torch.zeros((batch_size, hidden_size), dtype=dtype, device=out_device),
            torch.zeros((batch_size, hidden_size), dtype=dtype, device=out_device),
            torch.zeros((batch_size, 4 * hidden_size), dtype=dtype, device=out_device),
        )
    gates = input_gates.detach().cpu().to(acc_dtype) + hidden_gates.detach().cpu().to(
        acc_dtype
    )
    if input_bias is not None:
        gates = gates + input_bias.detach().cpu().to(acc_dtype)
    if hidden_bias is not None:
        gates = gates + hidden_bias.detach().cpu().to(acc_dtype)
    i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=1)
    i_gate = torch.sigmoid(i_gate)
    f_gate = torch.sigmoid(f_gate)
    g_gate = torch.tanh(g_gate)
    o_gate = torch.sigmoid(o_gate)
    cy = f_gate * cx.detach().cpu().to(acc_dtype) + i_gate * g_gate
    hy = o_gate * torch.tanh(cy)
    workspace = torch.cat((i_gate, f_gate, g_gate, o_gate), dim=1)
    return (
        hy.to(dtype=dtype, device=out_device),
        cy.to(dtype=dtype, device=out_device),
        workspace.to(dtype=dtype, device=out_device),
    )


def _cpu_backward_ref(
    grad_hy,
    grad_cy,
    cx,
    cy,
    workspace,
    has_bias,
    *,
    exact_dtype=False,
    out_device="cpu",
):
    if grad_hy is None and grad_cy is None:
        return None, None, None, None, None

    out_dtype = workspace.dtype
    acc_dtype = (
        out_dtype
        if (exact_dtype and out_dtype in (torch.float32, torch.float64))
        else torch.float64
    )
    batch_size, hidden_size = cx.shape
    ws_shape = workspace.shape
    if batch_size == 0 or hidden_size == 0:
        grad_gates = torch.zeros(ws_shape, dtype=out_dtype, device=out_device)
        grad_cx = torch.zeros(
            (batch_size, hidden_size), dtype=out_dtype, device=out_device
        )
        grad_bias = (
            torch.zeros((ws_shape[1],), dtype=out_dtype, device=out_device)
            if has_bias
            else None
        )
        return grad_gates, grad_gates, grad_cx, grad_bias, grad_bias

    cx_cpu = cx.detach().cpu().to(acc_dtype)
    cy_cpu = cy.detach().cpu().to(acc_dtype)
    ws_flat = (
        workspace.detach().cpu().to(acc_dtype).reshape(batch_size, 4 * hidden_size)
    )

    i_gate, f_gate, g_gate, o_gate = ws_flat.split(hidden_size, dim=1)
    tanh_cy = torch.tanh(cy_cpu)

    if grad_hy is not None:
        ghy_cpu = grad_hy.detach().cpu().to(acc_dtype)
        d_cy = ghy_cpu * o_gate * (1.0 - tanh_cy * tanh_cy)
        grad_o = ghy_cpu * tanh_cy * o_gate * (1.0 - o_gate)
    else:
        d_cy = torch.zeros_like(cy_cpu)
        grad_o = torch.zeros_like(cy_cpu)

    if grad_cy is not None:
        d_cy = d_cy + grad_cy.detach().cpu().to(acc_dtype)

    grad_i = d_cy * g_gate * i_gate * (1.0 - i_gate)
    grad_f = d_cy * cx_cpu * f_gate * (1.0 - f_gate)
    grad_g = d_cy * i_gate * (1.0 - g_gate * g_gate)
    grad_gates = (
        torch.cat((grad_i, grad_f, grad_g, grad_o), dim=1)
        .reshape(ws_shape)
        .to(dtype=out_dtype, device=out_device)
    )
    grad_cx = (d_cy * f_gate).to(dtype=out_dtype, device=out_device)
    grad_bias = (
        grad_gates.to(acc_dtype).sum(dim=0).to(dtype=out_dtype, device=out_device)
        if has_bias
        else None
    )
    return grad_gates, grad_gates, grad_cx, grad_bias, grad_bias


def _forward_ref(
    input_gates, hidden_gates, cx, bias_ih, bias_hh, *, on_input_device=False
):
    if _use_cpu_reference(cx):
        target_dev = (
            input_gates.device
            if on_input_device
            else ("cpu" if cfg.TO_CPU else input_gates.device)
        )
        return _cpu_forward_ref(
            input_gates, hidden_gates, cx, bias_ih, bias_hh, out_device=target_dev
        )
    if cfg.TO_CPU and not on_input_device:
        hy, cy, ws = torch.ops.aten._thnn_fused_lstm_cell(
            input_gates, hidden_gates, cx, bias_ih, bias_hh
        )
        return hy.cpu(), cy.cpu(), ws.cpu()
    return torch.ops.aten._thnn_fused_lstm_cell(
        input_gates, hidden_gates, cx, bias_ih, bias_hh
    )


def _backward_ref(args, *, exact_dtype=False):
    cx = args[2]
    if _use_cpu_reference(cx):
        target_dev = "cpu" if cfg.TO_CPU else cx.device
        return _cpu_backward_ref(*args, exact_dtype=exact_dtype, out_device=target_dev)
    return torch.ops.aten._thnn_fused_lstm_cell_backward(*_reference_args(args))


@pytest.mark.thnn_fused_lstm_cell_backward
@pytest.mark.parametrize(
    "dtype",
    [
        torch.int32,
        torch.int64,
        torch.complex64,
        pytest.param(torch.complex128, marks=_SKIP_FP64),
    ],
)
def test_thnn_fused_lstm_cell_backward_rejects_unsupported_dtype(dtype):
    state = torch.empty((2, 3), dtype=dtype, device=flag_gems.device)
    workspace = torch.empty((2, 12), dtype=dtype, device=flag_gems.device)
    args = (state, state, state, state, workspace, True)
    with torch.no_grad():
        if not _use_cpu_reference(state):
            with pytest.raises(RuntimeError):
                torch.ops.aten._thnn_fused_lstm_cell_backward(*_reference_args(args))
        with pytest.raises(RuntimeError, match="only supports"):
            flag_gems._thnn_fused_lstm_cell_backward(*args)


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
        hy, cy, workspace = _forward_ref(
            input_gates, hidden_gates, cx, bias, bias, on_input_device=True
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


@pytest.mark.thnn_fused_lstm_cell_backward
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("has_bias", [False, True])
def test_thnn_fused_lstm_cell_backward(dtype, shape, has_bias):
    args = _make_args(*shape, dtype, has_bias=has_bias)
    with torch.no_grad():
        reference = _backward_ref(args)
        result = flag_gems._thnn_fused_lstm_cell_backward(*args)
    _assert_outputs(result, reference, dtype, shape[0])


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
        reference = _backward_ref(args)
        result = flag_gems._thnn_fused_lstm_cell_backward(*args)
    _assert_outputs(result, reference, dtype, 5)


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
        reference = _backward_ref(args)
        result = flag_gems._thnn_fused_lstm_cell_backward(*args)
    _assert_outputs(result, reference, dtype, 5)


@pytest.mark.thnn_fused_lstm_cell_backward
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("has_bias", [False, True])
def test_thnn_fused_lstm_cell_backward_end_to_end(dtype, has_bias):
    batch_size, hidden_size = 5, 19
    input_gates = torch.randn(
        batch_size,
        4 * hidden_size,
        dtype=dtype,
        device=flag_gems.device,
    )
    hidden_gates = torch.randn_like(input_gates)
    cx = torch.randn(batch_size, hidden_size, dtype=dtype, device=flag_gems.device)
    bias = (
        torch.randn(4 * hidden_size, dtype=dtype, device=flag_gems.device)
        if has_bias
        else None
    )

    with torch.no_grad():
        ref_hy, ref_cy, _ = _forward_ref(input_gates, hidden_gates, cx, bias, bias)
        hy, cy, workspace = flag_gems._thnn_fused_lstm_cell(
            input_gates, hidden_gates, cx, bias, bias
        )
        grad_hy = torch.randn_like(hy)
        grad_cy = torch.randn_like(cy)
        args = (grad_hy, grad_cy, cx, cy, workspace, has_bias)
        reference = _backward_ref(args)
        result = flag_gems._thnn_fused_lstm_cell_backward(*args)

    utils.gems_assert_close(hy, ref_hy, dtype, reduce_dim=batch_size)
    utils.gems_assert_close(cy, ref_cy, dtype, reduce_dim=batch_size)
    _assert_outputs(result, reference, dtype, batch_size)


@pytest.mark.thnn_fused_lstm_cell_backward
@pytest.mark.parametrize("workspace_shape", [(1, 80), (10, 8), (20, 4)])
def test_thnn_fused_lstm_cell_backward_reshaped_workspace(workspace_shape):
    args = list(_make_args(5, 4, torch.float32))
    args[4] = args[4].reshape(workspace_shape)
    with torch.no_grad():
        reference = _backward_ref(tuple(args))
        result = flag_gems._thnn_fused_lstm_cell_backward(*args)

    assert result[0].shape == workspace_shape
    assert result[1].shape == workspace_shape
    _assert_outputs(result, reference, torch.float32, 5)


@pytest.mark.thnn_fused_lstm_cell_backward
@pytest.mark.parametrize("shape", [(0, 8), (2, 0)])
def test_thnn_fused_lstm_cell_backward_empty(shape):
    args = _make_args(*shape, torch.float32)
    with torch.no_grad():
        reference = _backward_ref(args)
        result = flag_gems._thnn_fused_lstm_cell_backward(*args)
    _assert_outputs(result, reference, torch.float32, shape[0])


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
        reference = _backward_ref(args, exact_dtype=True)
        result = flag_gems._thnn_fused_lstm_cell_backward(*args)
    _assert_outputs(result, reference, torch.float32, 1)

    for actual, expected in zip(result, reference):
        actual_cpu = actual.cpu()
        expected_cpu = expected.cpu()
        valid = ~torch.isnan(expected_cpu)
        utils.gems_assert_equal(
            torch.signbit(actual_cpu)[valid], torch.signbit(expected_cpu)[valid]
        )


@pytest.mark.thnn_fused_lstm_cell_backward
def test_thnn_fused_lstm_cell_backward_requires_no_grad():
    args = _make_args(2, 4, torch.float32)
    with pytest.raises(RuntimeError, match="grad mode disabled"):
        flag_gems._thnn_fused_lstm_cell_backward(*args)

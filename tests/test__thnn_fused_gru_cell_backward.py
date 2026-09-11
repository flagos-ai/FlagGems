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
from .conftest import QUICK_MODE

DTYPES = utils.ALL_FLOAT_DTYPES
SHAPES = [(1, 4), (2, 17), (8, 64), (33, 129), (513, 129)]
if QUICK_MODE:
    SHAPES = [(2, 17)]


def _make_workspace(batch_size, hidden_size, dtype):
    input_gates = torch.randn(
        batch_size, 3 * hidden_size, device=flag_gems.device, dtype=dtype
    )
    hidden_gates = torch.randn_like(input_gates)
    hx = torch.randn(batch_size, hidden_size, device=flag_gems.device, dtype=dtype)
    input_bias = torch.randn(3 * hidden_size, device=flag_gems.device, dtype=dtype)
    hidden_bias = torch.randn_like(input_bias)
    return torch.ops.aten._thnn_fused_gru_cell(
        input_gates, hidden_gates, hx, input_bias, hidden_bias
    )[1]


def _manual_reference(grad_hy, workspace, has_bias):
    dtype = grad_hy.dtype
    compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    grad_hy_compute = grad_hy.to(compute_dtype)
    reset_gate, input_gate, new_gate, hx, hidden_new = (
        value.to(compute_dtype) for value in workspace.chunk(5, dim=1)
    )
    grad_input_gate = grad_hy_compute * (hx - new_gate)
    grad_input_gate *= 1.0 - input_gate
    grad_input_gate *= input_gate
    grad_hx = grad_hy_compute * input_gate
    grad_new_input = grad_hy_compute * (1.0 - input_gate)
    grad_new_input *= 1.0 - new_gate * new_gate
    grad_hidden_new = grad_new_input * reset_gate
    grad_reset = grad_new_input * hidden_new
    grad_reset *= 1.0 - reset_gate
    grad_reset *= reset_gate
    grad_input_gates = torch.cat(
        (grad_reset, grad_input_gate, grad_new_input), dim=1
    ).to(dtype)
    grad_hidden_gates = torch.cat(
        (grad_reset, grad_input_gate, grad_hidden_new), dim=1
    ).to(dtype)
    grad_hx = grad_hx.to(dtype)
    if not has_bias:
        return grad_input_gates, grad_hidden_gates, grad_hx, None, None
    grad_input_bias = grad_input_gates.to(compute_dtype).sum(dim=0).to(dtype)
    grad_hidden_bias = grad_hidden_gates.to(compute_dtype).sum(dim=0).to(dtype)
    return (
        grad_input_gates,
        grad_hidden_gates,
        grad_hx,
        grad_input_bias,
        grad_hidden_bias,
    )


def _reference(grad_hy, workspace, has_bias):
    ref_grad_hy = utils.to_reference(grad_hy)
    ref_workspace = utils.to_reference(workspace)
    if ref_grad_hy.device.type == "cpu":
        return _manual_reference(ref_grad_hy, ref_workspace, has_bias)
    return torch.ops.aten._thnn_fused_gru_cell_backward(
        ref_grad_hy, ref_workspace, has_bias
    )


def _assert_outputs_close(result, reference, dtype, batch_size):
    base_atol = 1e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-4
    for index, (actual, expected) in enumerate(zip(result, reference)):
        if expected is None:
            assert actual is None
            continue
        atol = 2e-3 if dtype == torch.bfloat16 and index >= 3 else base_atol
        utils.gems_assert_close(
            actual,
            expected,
            dtype,
            reduce_dim=max(batch_size, 1) if index >= 3 else 1,
            atol=atol,
        )


@pytest.mark.thnn_fused_gru_cell_backward
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("has_bias", [False, True])
def test_thnn_fused_gru_cell_backward(dtype, shape, has_bias):
    batch_size, hidden_size = shape
    workspace = _make_workspace(batch_size, hidden_size, dtype)
    grad_hy = torch.randn(batch_size, hidden_size, device=flag_gems.device, dtype=dtype)
    reference = _reference(grad_hy, workspace, has_bias)

    with flag_gems.use_gems():
        result = torch.ops.aten._thnn_fused_gru_cell_backward(
            grad_hy, workspace, has_bias
        )

    _assert_outputs_close(result, reference, dtype, batch_size)


@pytest.mark.thnn_fused_gru_cell_backward
@pytest.mark.parametrize("dtype", DTYPES)
def test_thnn_fused_gru_cell_backward_noncontiguous(dtype):
    batch_size, hidden_size = 5, 19
    grad_hy = torch.randn(
        hidden_size, batch_size, device=flag_gems.device, dtype=dtype
    ).T
    workspace_value = _make_workspace(batch_size, hidden_size, dtype)
    workspace = torch.empty(
        batch_size,
        10 * hidden_size,
        device=flag_gems.device,
        dtype=dtype,
    )[:, ::2]
    workspace.copy_(workspace_value)
    assert not grad_hy.is_contiguous() and not workspace.is_contiguous()
    reference = _reference(grad_hy, workspace, True)

    result = flag_gems._thnn_fused_gru_cell_backward(grad_hy, workspace, True)

    _assert_outputs_close(result, reference, dtype, batch_size)


@pytest.mark.thnn_fused_gru_cell_backward
@pytest.mark.parametrize("shape", [(0, 8), (2, 0)])
def test_thnn_fused_gru_cell_backward_empty(shape):
    batch_size, hidden_size = shape
    grad_hy = torch.empty(shape, device=flag_gems.device)
    workspace = torch.empty(batch_size, 5 * hidden_size, device=flag_gems.device)
    reference = _reference(grad_hy, workspace, True)

    with flag_gems.use_gems():
        result = torch.ops.aten._thnn_fused_gru_cell_backward(grad_hy, workspace, True)

    _assert_outputs_close(result, reference, torch.float32, batch_size)


@pytest.mark.thnn_fused_gru_cell_backward
def test_thnn_fused_gru_cell_backward_invalid_workspace():
    grad_hy = torch.randn(2, 8, device=flag_gems.device)
    workspace = torch.randn(2, 39, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems._thnn_fused_gru_cell_backward(grad_hy, workspace, False)


def _make_noncontiguous_outputs(batch_size, hidden_size, dtype):
    device = flag_gems.device
    out0 = torch.empty(3 * hidden_size, batch_size, device=device, dtype=dtype).T
    out1 = torch.empty(3 * hidden_size, batch_size, device=device, dtype=dtype).T
    out2 = torch.empty(hidden_size, batch_size, device=device, dtype=dtype).T
    out3 = torch.empty(6 * hidden_size, device=device, dtype=dtype)[::2]
    out4 = torch.empty(6 * hidden_size, device=device, dtype=dtype)[::2]
    return out0, out1, out2, out3, out4


@pytest.mark.thnn_fused_gru_cell_backward_out
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_thnn_fused_gru_cell_backward_out(dtype, shape):
    batch_size, hidden_size = shape
    workspace = _make_workspace(batch_size, hidden_size, dtype)
    grad_hy = torch.randn(batch_size, hidden_size, device=flag_gems.device, dtype=dtype)
    reference = _reference(grad_hy, workspace, True)
    outputs = _make_noncontiguous_outputs(batch_size, hidden_size, dtype)

    with flag_gems.use_gems():
        result = torch.ops.aten._thnn_fused_gru_cell_backward.out(
            grad_hy,
            workspace,
            True,
            out0=outputs[0],
            out1=outputs[1],
            out2=outputs[2],
            out3=outputs[3],
            out4=outputs[4],
        )

    assert all(actual is out for actual, out in zip(result, outputs))
    _assert_outputs_close(result, reference, dtype, batch_size)


@pytest.mark.thnn_fused_gru_cell_backward_out
def test_thnn_fused_gru_cell_backward_out_resize():
    batch_size, hidden_size = 2, 7
    workspace = _make_workspace(batch_size, hidden_size, torch.float32)
    grad_hy = torch.randn(batch_size, hidden_size, device=flag_gems.device)
    outputs = tuple(torch.empty(0, device=flag_gems.device) for _ in range(5))

    result = flag_gems._thnn_fused_gru_cell_backward_out(
        grad_hy,
        workspace,
        True,
        out0=outputs[0],
        out1=outputs[1],
        out2=outputs[2],
        out3=outputs[3],
        out4=outputs[4],
    )

    assert all(actual is out for actual, out in zip(result, outputs))
    assert [tuple(out.shape) for out in outputs] == [
        (batch_size, 3 * hidden_size),
        (batch_size, 3 * hidden_size),
        (batch_size, hidden_size),
        (3 * hidden_size,),
        (3 * hidden_size,),
    ]

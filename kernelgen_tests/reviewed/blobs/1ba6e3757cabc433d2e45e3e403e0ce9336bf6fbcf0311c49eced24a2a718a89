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

# LSTM cell backward test shapes
LSTM_SHAPES = [
    (1, 4),  # batch=1, hidden=4
    (2, 8),  # batch=2, hidden=8
    (4, 16),  # batch=4, hidden=16
    (8, 32),  # batch=8, hidden=32
    (16, 64),  # batch=16, hidden=64
]


def _cpu_reference(grad_hy, grad_cy, cx, cy, workspace):
    dtype = cx.dtype
    compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    grad_hy, grad_cy, cx, cy, workspace = (
        value.detach().cpu().to(compute_dtype)
        for value in (grad_hy, grad_cy, cx, cy, workspace)
    )
    input_gate, forget_gate, cell_gate, output_gate = workspace.chunk(4, dim=1)
    tanh_cy = cy.tanh()
    grad_cell = grad_cy + grad_hy * output_gate * (1 - tanh_cy * tanh_cy)
    grad_gates = torch.cat(
        (
            grad_cell * cell_gate * input_gate * (1 - input_gate),
            grad_cell * cx * forget_gate * (1 - forget_gate),
            grad_cell * input_gate * (1 - cell_gate * cell_gate),
            grad_hy * tanh_cy * output_gate * (1 - output_gate),
        ),
        dim=1,
    ).to(dtype)
    grad_cx = (grad_cell * forget_gate).to(dtype)
    grad_bias = grad_gates.to(compute_dtype).sum(dim=0).to(dtype)
    return grad_gates, grad_cx, grad_bias


@pytest.mark.thnn_fused_lstm_cell_backward_impl
@pytest.mark.underscore_thnn_fused_lstm_cell_backward_impl
@pytest.mark.parametrize("shape", LSTM_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_thnn_fused_lstm_cell_backward_impl(shape, dtype):
    """Test accuracy for _thnn_fused_lstm_cell_backward_impl."""
    if flag_gems.vendor_name == "cambricon" and dtype == torch.bfloat16:
        pytest.skip("Issue #5254: Not supported")
    if flag_gems.vendor_name == "ascend" and dtype == torch.bfloat16:
        pytest.skip("Ascend native LSTM forward setup supports only float and half")
    torch.manual_seed(0)
    batch_size, hidden_size = shape
    dev = flag_gems.device

    # Create input tensors on CUDA (forward op is CUDA-only)
    input_gates = torch.randn(batch_size, 4 * hidden_size, dtype=dtype, device=dev)
    hidden_gates = torch.randn(batch_size, 4 * hidden_size, dtype=dtype, device=dev)
    cx = torch.randn(batch_size, hidden_size, dtype=dtype, device=dev)
    input_bias = torch.zeros(4 * hidden_size, dtype=dtype, device=dev)
    hidden_bias = torch.randn(4 * hidden_size, dtype=dtype, device=dev)

    # Forward pass (CUDA-only; _thnn_fused_lstm_cell has no CPU kernel)
    hx, cy, workspace = torch.ops.aten._thnn_fused_lstm_cell(
        input_gates, hidden_gates, cx, input_bias, hidden_bias
    )

    # Gradient tensors for backward
    grad_hy = torch.randn_like(hx)
    grad_cy = torch.randn_like(cy)

    # Backward pass — ATen reference (outside use_gems, on CUDA)
    if utils.TO_CPU:
        ref_out = _cpu_reference(grad_hy, grad_cy, cx, cy, workspace)
    else:
        ref_out = torch.ops.aten._thnn_fused_lstm_cell_backward_impl(
            grad_hy, grad_cy, cx, cy, workspace, True
        )
    res_out = flag_gems._thnn_fused_lstm_cell_backward_impl(
        grad_hy, grad_cy, cx, cy, workspace, True
    )

    # Compare outputs — ref_out order: (grad_input_gates, grad_cx, grad_biases)
    for i, (ref, res) in enumerate(zip(ref_out, res_out)):
        assert res.device == cx.device
        assert (
            res.shape == ref.shape
        ), f"Shape mismatch at output[{i}]: {res.shape} vs {ref.shape}"
        assert (
            res.dtype == ref.dtype == dtype
        ), f"Dtype mismatch at output[{i}]: {res.dtype} vs {ref.dtype}"
        ref = utils.to_reference(ref, dtype != torch.float64)
        utils.gems_assert_close(res.cpu(), ref.cpu(), dtype, atol=5e-2)

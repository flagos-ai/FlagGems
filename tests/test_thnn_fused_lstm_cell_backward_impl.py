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


@pytest.mark.thnn_fused_lstm_cell_backward_impl
@pytest.mark.parametrize("shape", LSTM_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_thnn_fused_lstm_cell_backward_impl(shape, dtype):
    """Test accuracy for _thnn_fused_lstm_cell_backward_impl."""
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
    ref_out = torch.ops.aten._thnn_fused_lstm_cell_backward_impl(
        grad_hy, grad_cy, cx, cy, workspace, True
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten._thnn_fused_lstm_cell_backward_impl(
            grad_hy, grad_cy, cx, cy, workspace, True
        )

    # Compare outputs — ref_out order: (grad_input_gates, grad_cx, grad_biases)
    for i, (ref, res) in enumerate(zip(ref_out, res_out)):
        assert (
            res.shape == ref.shape
        ), f"Shape mismatch at output[{i}]: {res.shape} vs {ref.shape}"
        assert (
            res.dtype == ref.dtype
        ), f"Dtype mismatch at output[{i}]: {res.dtype} vs {ref.dtype}"
        utils.gems_assert_close(res.cpu(), ref.cpu(), dtype, atol=5e-2)

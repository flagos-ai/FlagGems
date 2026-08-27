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

# Shapes for LSTM cell: (batch, hidden_size) for cx, (batch, 4*hidden_size) for gates
LSTM_SHAPES = [
    (1, 4),
    (4, 16),
    (8, 32),
    (16, 64),
    (32, 128),
]


@pytest.mark.thnn_fused_lstm_cell
@pytest.mark.parametrize("shape", LSTM_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_thnn_fused_lstm_cell(shape, dtype):
    """Test _thnn_fused_lstm_cell accuracy against PyTorch reference."""
    batch_size, hidden_size = shape
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Create inputs
    input_gates = torch.randn(
        batch_size, 4 * hidden_size, dtype=dtype, device=flag_gems.device
    )
    hidden_gates = torch.randn(
        batch_size, 4 * hidden_size, dtype=dtype, device=flag_gems.device
    )
    cx = torch.randn(batch_size, hidden_size, dtype=dtype, device=flag_gems.device)

    # Reference implementation using PyTorch
    ref_input_gates = utils.to_reference(input_gates)
    ref_hidden_gates = utils.to_reference(hidden_gates)
    ref_cx = utils.to_reference(cx)

    # Compute reference
    ref_gates = ref_input_gates + ref_hidden_gates
    i, f, g, o = ref_gates.chunk(4, dim=-1)
    i_ref = torch.sigmoid(i)
    f_ref = torch.sigmoid(f)
    g_ref = torch.tanh(g)
    o_ref = torch.sigmoid(o)
    ref_cy = f_ref * ref_cx + i_ref * g_ref
    ref_hy = o_ref * torch.tanh(ref_cy)

    # Compute with FlagGems
    with flag_gems.use_gems():
        res_hy, res_cy, res_workspace = torch.ops.aten._thnn_fused_lstm_cell(
            input_gates, hidden_gates, cx
        )

    # Use looser tolerance for float16/bfloat16 due to accumulated errors from multiple ops
    atol = (
        1e-2
        if dtype == torch.bfloat16
        else (1.5e-3 if dtype == torch.float16 else 1e-4)
    )
    utils.gems_assert_close(res_hy, ref_hy, dtype, atol=atol)
    utils.gems_assert_close(res_cy, ref_cy, dtype, atol=atol)


@pytest.mark.thnn_fused_lstm_cell
@pytest.mark.parametrize("shape", LSTM_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_thnn_fused_lstm_cell_with_bias(shape, dtype):
    """Test _thnn_fused_lstm_cell with biases."""
    batch_size, hidden_size = shape
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Create inputs with biases
    input_gates = torch.randn(
        batch_size, 4 * hidden_size, dtype=dtype, device=flag_gems.device
    )
    hidden_gates = torch.randn(
        batch_size, 4 * hidden_size, dtype=dtype, device=flag_gems.device
    )
    cx = torch.randn(batch_size, hidden_size, dtype=dtype, device=flag_gems.device)
    input_bias = torch.randn(4 * hidden_size, dtype=dtype, device=flag_gems.device)
    hidden_bias = torch.randn(4 * hidden_size, dtype=dtype, device=flag_gems.device)

    # Reference implementation
    ref_input_gates = utils.to_reference(input_gates)
    ref_hidden_gates = utils.to_reference(hidden_gates)
    ref_cx = utils.to_reference(cx)
    ref_input_bias = utils.to_reference(input_bias)
    ref_hidden_bias = utils.to_reference(hidden_bias)

    ref_gates = ref_input_gates + ref_hidden_gates + ref_input_bias + ref_hidden_bias
    i, f, g, o = ref_gates.chunk(4, dim=-1)
    i_ref = torch.sigmoid(i)
    f_ref = torch.sigmoid(f)
    g_ref = torch.tanh(g)
    o_ref = torch.sigmoid(o)
    ref_cy = f_ref * ref_cx + i_ref * g_ref
    ref_hy = o_ref * torch.tanh(ref_cy)

    # Compute with FlagGems
    with flag_gems.use_gems():
        res_hy, res_cy, res_workspace = torch.ops.aten._thnn_fused_lstm_cell(
            input_gates, hidden_gates, cx, input_bias, hidden_bias
        )

    # Use looser tolerance for float16/bfloat16 due to accumulated errors from multiple ops
    atol = (
        1e-2
        if dtype == torch.bfloat16
        else (1.5e-3 if dtype == torch.float16 else 1e-4)
    )
    utils.gems_assert_close(res_hy, ref_hy, dtype, atol=atol)
    utils.gems_assert_close(res_cy, ref_cy, dtype, atol=atol)

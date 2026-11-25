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
    (5, 193),
]


def _cpu_fused_lstm_cell_ref(
    input_gates, hidden_gates, cx, input_bias=None, hidden_bias=None
):
    dtype = input_gates.dtype
    ref_ig = utils.to_reference(input_gates, upcast=True)
    ref_hg = utils.to_reference(hidden_gates, upcast=True)
    ref_cx = utils.to_reference(cx, upcast=True)
    ref_gates = ref_ig + ref_hg
    if input_bias is not None:
        ref_gates = ref_gates + utils.to_reference(input_bias, upcast=True)
    if hidden_bias is not None:
        ref_gates = ref_gates + utils.to_reference(hidden_bias, upcast=True)
    i, f, g, o = ref_gates.chunk(4, dim=-1)
    i_ref = torch.sigmoid(i)
    f_ref = torch.sigmoid(f)
    g_ref = torch.tanh(g)
    o_ref = torch.sigmoid(o)
    ref_cy = f_ref * ref_cx + i_ref * g_ref
    ref_hy = o_ref * torch.tanh(ref_cy)
    ref_workspace = torch.cat((i_ref, f_ref, g_ref, o_ref), dim=-1)
    return (
        ref_hy.to(dtype),
        ref_cy.to(dtype),
        ref_workspace.to(dtype),
    )


def _atol_for(dtype):
    if dtype == torch.bfloat16:
        return 1e-2
    if dtype == torch.float16:
        return 1.5e-3
    return 1e-4


@pytest.mark.thnn_fused_lstm_cell
@pytest.mark.parametrize("shape", LSTM_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_thnn_fused_lstm_cell(shape, dtype):
    """Test _thnn_fused_lstm_cell accuracy (including activated gate workspace)."""
    batch_size, hidden_size = shape
    torch.manual_seed(42)

    input_gates = torch.randn(
        batch_size, 4 * hidden_size, dtype=dtype, device=flag_gems.device
    )
    hidden_gates = torch.randn(
        batch_size, 4 * hidden_size, dtype=dtype, device=flag_gems.device
    )
    cx = torch.randn(batch_size, hidden_size, dtype=dtype, device=flag_gems.device)

    ref_hy, ref_cy, ref_workspace = _cpu_fused_lstm_cell_ref(
        input_gates, hidden_gates, cx
    )
    res_hy, res_cy, res_workspace = flag_gems._thnn_fused_lstm_cell(
        input_gates, hidden_gates, cx
    )

    atol = _atol_for(dtype)
    utils.gems_assert_close(res_hy, ref_hy, dtype, atol=atol)
    utils.gems_assert_close(res_cy, ref_cy, dtype, atol=atol)
    utils.gems_assert_close(res_workspace, ref_workspace, dtype, atol=atol)


@pytest.mark.thnn_fused_lstm_cell
@pytest.mark.parametrize("shape", LSTM_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_thnn_fused_lstm_cell_with_bias(shape, dtype, noncontiguous):
    """Test _thnn_fused_lstm_cell with biases and non-contiguous inputs."""
    batch_size, hidden_size = shape
    torch.manual_seed(42)

    input_gates = torch.randn(
        batch_size, 4 * hidden_size, dtype=dtype, device=flag_gems.device
    )
    hidden_gates = torch.randn(
        batch_size, 4 * hidden_size, dtype=dtype, device=flag_gems.device
    )
    cx = torch.randn(batch_size, hidden_size, dtype=dtype, device=flag_gems.device)
    input_bias = torch.randn(4 * hidden_size, dtype=dtype, device=flag_gems.device)
    hidden_bias = torch.randn(4 * hidden_size, dtype=dtype, device=flag_gems.device)

    if noncontiguous:
        input_gates = input_gates.T.contiguous().T
        hidden_gates = hidden_gates.T.contiguous().T
        cx = cx.T.contiguous().T
        input_bias = torch.stack((input_bias, input_bias), dim=-1)[:, 0]
        hidden_bias = torch.stack((hidden_bias, hidden_bias), dim=-1)[:, 1]

    ref_hy, ref_cy, ref_workspace = _cpu_fused_lstm_cell_ref(
        input_gates, hidden_gates, cx, input_bias, hidden_bias
    )
    res_hy, res_cy, res_workspace = flag_gems._thnn_fused_lstm_cell(
        input_gates, hidden_gates, cx, input_bias, hidden_bias
    )

    atol = _atol_for(dtype)
    utils.gems_assert_close(res_hy, ref_hy, dtype, atol=atol)
    utils.gems_assert_close(res_cy, ref_cy, dtype, atol=atol)
    utils.gems_assert_close(res_workspace, ref_workspace, dtype, atol=atol)

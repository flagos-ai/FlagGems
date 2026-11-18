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

"""GCU400-specific ``_thnn_fused_lstm_cell_backward_impl``.

The common implementation splits the LSTM cell backward into a dozen PyTorch tensor ops (slicing, tanh,
elementwise multiply, cat, column sums), and on GCU each launch's fixed overhead far exceeds the actual compute.
Here CTAs are partitioned by hidden column block: one program handles all batch rows for several columns,
writing the four gate gradients and grad_cx while reducing over batch in registers for the bias gradients,
so all outputs are produced by a single Triton launch.

``FUSED_TRITON=0`` keeps a PyTorch path word-for-word equivalent to the common implementation as baseline and fallback;
after optimization the main path is fixed to the ``FUSED_TRITON=1`` Triton kernel.
"""

import logging

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---
# Compile-time config, adjusted per optimization round; semantics unchanged.
FUSED_TRITON = 1
BLOCK_B = 32
BLOCK_H = 32
NUM_WARPS = 2
TANH_VIA_SIGMOID = 1
# --- TUNING PARAMS END ---


@libentry()
@triton.jit(do_not_specialize=["B", "H"])
def _lstm_cell_backward_fused_kernel(
    grad_hy_ptr,
    grad_cy_ptr,
    cx_ptr,
    cy_ptr,
    workspace_ptr,
    grad_gates_ptr,
    grad_cx_ptr,
    grad_bias_ptr,
    B,
    H,
    HAS_BIAS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_H: tl.constexpr,
    TANH_VIA_SIGMOID: tl.constexpr,
):
    # 2D tile: rows cover batch rows, cols cover a contiguous range of hidden columns.
    # workspace is laid out contiguously along dim=1 as [i, f, g, o], so the four base addresses differ by H.
    rows = tl.arange(0, BLOCK_B)
    cols = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
    valid = (rows[:, None] < B) & (cols[None, :] < H)
    hidden_offset = rows[:, None] * H + cols[None, :]
    gate_offset = rows[:, None] * (4 * H) + cols[None, :]

    grad_hy = tl.load(grad_hy_ptr + hidden_offset, mask=valid, other=0.0).to(tl.float32)
    grad_cy = tl.load(grad_cy_ptr + hidden_offset, mask=valid, other=0.0).to(tl.float32)
    cx = tl.load(cx_ptr + hidden_offset, mask=valid, other=0.0).to(tl.float32)
    cy = tl.load(cy_ptr + hidden_offset, mask=valid, other=0.0).to(tl.float32)
    i_gate = tl.load(workspace_ptr + gate_offset, mask=valid, other=0.0).to(tl.float32)
    f_gate = tl.load(workspace_ptr + gate_offset + H, mask=valid, other=0.0).to(
        tl.float32
    )
    g_gate = tl.load(workspace_ptr + gate_offset + 2 * H, mask=valid, other=0.0).to(
        tl.float32
    )
    o_gate = tl.load(workspace_ptr + gate_offset + 3 * H, mask=valid, other=0.0).to(
        tl.float32
    )

    # tanh(cy) is expressed with the identity 2*sigmoid(2*cy)-1 to avoid depending on device-side libdevice.
    if TANH_VIA_SIGMOID:
        tanh_cy = 2.0 * tl.sigmoid(2.0 * cy) - 1.0
    else:
        exp_neg = tl.exp(-2.0 * cy)
        tanh_cy = (1.0 - exp_neg) / (1.0 + exp_neg)

    d_cy = grad_hy * o_gate * (1.0 - tanh_cy * tanh_cy) + grad_cy
    grad_i = d_cy * g_gate * i_gate * (1.0 - i_gate)
    grad_f = d_cy * cx * f_gate * (1.0 - f_gate)
    grad_g = d_cy * i_gate * (1.0 - g_gate * g_gate)
    grad_o = grad_hy * tanh_cy * o_gate * (1.0 - o_gate)

    tl.store(grad_gates_ptr + gate_offset, grad_i, mask=valid)
    tl.store(grad_gates_ptr + gate_offset + H, grad_f, mask=valid)
    tl.store(grad_gates_ptr + gate_offset + 2 * H, grad_g, mask=valid)
    tl.store(grad_gates_ptr + gate_offset + 3 * H, grad_o, mask=valid)
    tl.store(grad_cx_ptr + hidden_offset, d_cy * f_gate, mask=valid)

    # The bias gradient is the column sum of grad_input_gates over batch; masked-out elements are already zero,
    # so it can be reduced over rows inside the same kernel without atomics or a second launch.
    if HAS_BIAS:
        bias_mask = cols < H
        tl.store(grad_bias_ptr + cols, tl.sum(grad_i, axis=0), mask=bias_mask)
        tl.store(grad_bias_ptr + H + cols, tl.sum(grad_f, axis=0), mask=bias_mask)
        tl.store(grad_bias_ptr + 2 * H + cols, tl.sum(grad_g, axis=0), mask=bias_mask)
        tl.store(grad_bias_ptr + 3 * H + cols, tl.sum(grad_o, axis=0), mask=bias_mask)


def _public_equivalent(
    grad_hy: torch.Tensor,
    grad_cy: torch.Tensor,
    cx: torch.Tensor,
    cy: torch.Tensor,
    workspace: torch.Tensor,
    has_bias: bool,
):
    """Baseline path word-for-word equivalent to the common implementation, used only with ``FUSED_TRITON=0``."""
    _, hidden_size = cx.shape
    grad_hy = grad_hy.contiguous()
    grad_cy = grad_cy.contiguous()
    cx = cx.contiguous()
    cy = cy.contiguous()
    workspace = workspace.contiguous()

    i_gate = workspace[:, :hidden_size]
    f_gate = workspace[:, hidden_size : 2 * hidden_size]
    g_gate = workspace[:, 2 * hidden_size : 3 * hidden_size]
    o_gate = workspace[:, 3 * hidden_size : 4 * hidden_size]

    tanh_cy = flag_gems.tanh(cy)
    d_cy = grad_hy * o_gate * (1.0 - tanh_cy * tanh_cy) + grad_cy
    grad_i = d_cy * g_gate * i_gate * (1.0 - i_gate)
    grad_f = d_cy * cx * f_gate * (1.0 - f_gate)
    grad_g = d_cy * i_gate * (1.0 - g_gate * g_gate)
    grad_o = grad_hy * tanh_cy * o_gate * (1.0 - o_gate)
    grad_input_gates = flag_gems.cat([grad_i, grad_f, grad_g, grad_o], dim=1)
    grad_cx = d_cy * f_gate
    if has_bias:
        grad_biases = grad_input_gates.sum(dim=0)
    else:
        grad_biases = torch.zeros(0, dtype=cx.dtype, device=cx.device)
    return grad_input_gates, grad_cx, grad_biases


def _thnn_fused_lstm_cell_backward_impl(
    grad_hy: torch.Tensor,
    grad_cy: torch.Tensor,
    cx: torch.Tensor,
    cy: torch.Tensor,
    workspace: torch.Tensor,
    has_bias: bool,
):
    """Compute the three outputs of fused LSTM cell backward; signature matches the common implementation."""
    logger.debug("GEMS_ENFLAME_GCU400 _THNN_FUSED_LSTM_CELL_BACKWARD_IMPL")
    if FUSED_TRITON == 0:
        return _public_equivalent(grad_hy, grad_cy, cx, cy, workspace, has_bias)

    batch_size, hidden_size = cx.shape
    grad_hy = grad_hy.contiguous()
    grad_cy = grad_cy.contiguous()
    cx = cx.contiguous()
    cy = cy.contiguous()
    workspace = workspace.contiguous()

    grad_input_gates = torch.empty(
        (batch_size, 4 * hidden_size), dtype=cx.dtype, device=cx.device
    )
    grad_cx = torch.empty((batch_size, hidden_size), dtype=cx.dtype, device=cx.device)
    if has_bias:
        grad_biases = torch.empty(4 * hidden_size, dtype=cx.dtype, device=cx.device)
    else:
        grad_biases = torch.empty(0, dtype=cx.dtype, device=cx.device)

    # Grid cap: prevents exceeding GCU's hardware grid limit when BLOCK_H is small.
    grid_size = min(triton.cdiv(hidden_size, BLOCK_H), 65535)
    with torch_device_fn.device(cx.device):
        _lstm_cell_backward_fused_kernel[(grid_size,)](
            grad_hy,
            grad_cy,
            cx,
            cy,
            workspace,
            grad_input_gates,
            grad_cx,
            grad_biases,
            batch_size,
            hidden_size,
            HAS_BIAS=bool(has_bias),
            BLOCK_B=max(BLOCK_B, triton.next_power_of_2(batch_size)),
            BLOCK_H=BLOCK_H,
            TANH_VIA_SIGMOID=TANH_VIA_SIGMOID,
            num_warps=NUM_WARPS,
        )
    return grad_input_gates, grad_cx, grad_biases

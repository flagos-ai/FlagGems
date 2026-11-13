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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim
from flag_gems.utils.libentry import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _thnn_fused_lstm_cell_kernel(
    input_gates_ptr,
    hidden_gates_ptr,
    cx_ptr,
    input_bias_ptr,
    hidden_bias_ptr,
    hy_ptr,
    cy_ptr,
    workspace_ptr,
    batch_size: tl.constexpr,
    hidden_size: tl.constexpr,
    has_input_bias: tl.constexpr,
    has_hidden_bias: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Grid: (batch_size, cdiv(hidden_size, BLOCK_SIZE))
    batch_idx = tl.program_id(0)
    hblk = tl.program_id(1)

    # Pointers for this batch
    ig_base = input_gates_ptr + batch_idx * 4 * hidden_size
    hg_base = hidden_gates_ptr + batch_idx * 4 * hidden_size
    cx_base = cx_ptr + batch_idx * hidden_size
    hy_base = hy_ptr + batch_idx * hidden_size
    cy_base = cy_ptr + batch_idx * hidden_size
    ws_base = workspace_ptr + batch_idx * 4 * hidden_size

    # Offsets for this block
    offsets = hblk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size

    # Issue every load before any activation so the four gate computations
    # below are independent expressions: their sigmoid/tanh latency chains
    # then overlap instead of serializing behind the per-gate stores.
    ig0 = tl.load(ig_base + 0 * hidden_size + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    ig1 = tl.load(ig_base + 1 * hidden_size + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    ig2 = tl.load(ig_base + 2 * hidden_size + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    ig3 = tl.load(ig_base + 3 * hidden_size + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    hg0 = tl.load(hg_base + 0 * hidden_size + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    hg1 = tl.load(hg_base + 1 * hidden_size + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    hg2 = tl.load(hg_base + 2 * hidden_size + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    hg3 = tl.load(hg_base + 3 * hidden_size + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    cx_vals = tl.load(cx_base + offsets, mask=mask, other=0.0).to(tl.float32)

    if has_input_bias:
        ib0 = tl.load(
            input_bias_ptr + 0 * hidden_size + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        ib1 = tl.load(
            input_bias_ptr + 1 * hidden_size + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        ib2 = tl.load(
            input_bias_ptr + 2 * hidden_size + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        ib3 = tl.load(
            input_bias_ptr + 3 * hidden_size + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        ig0 = ig0 + ib0
        ig1 = ig1 + ib1
        ig2 = ig2 + ib2
        ig3 = ig3 + ib3
    if has_hidden_bias:
        hb0 = tl.load(
            hidden_bias_ptr + 0 * hidden_size + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        hb1 = tl.load(
            hidden_bias_ptr + 1 * hidden_size + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        hb2 = tl.load(
            hidden_bias_ptr + 2 * hidden_size + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        hb3 = tl.load(
            hidden_bias_ptr + 3 * hidden_size + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        hg0 = hg0 + hb0
        hg1 = hg1 + hb1
        hg2 = hg2 + hb2
        hg3 = hg3 + hb3

    # gate 0=input, 1=forget, 2=cell, 3=output
    i_gate = tl.sigmoid(ig0 + hg0)
    f_gate = tl.sigmoid(ig1 + hg1)
    g_gate = tl_extra_shim.tanh(ig2 + hg2)
    o_gate = tl.sigmoid(ig3 + hg3)

    # Compute cy = f * cx + i * g
    cy_acc = f_gate * cx_vals + i_gate * g_gate
    # Compute hy = o * tanh(cy)
    hy_acc = o_gate * tl_extra_shim.tanh(cy_acc)

    # The backward operator consumes activated gates, matching ATen's
    # workspace contract.
    tl.store(
        ws_base + 0 * hidden_size + offsets,
        i_gate.to(workspace_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(
        ws_base + 1 * hidden_size + offsets,
        f_gate.to(workspace_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(
        ws_base + 2 * hidden_size + offsets,
        g_gate.to(workspace_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(
        ws_base + 3 * hidden_size + offsets,
        o_gate.to(workspace_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(cy_base + offsets, cy_acc.to(cy_ptr.dtype.element_ty), mask=mask)
    tl.store(hy_base + offsets, hy_acc.to(hy_ptr.dtype.element_ty), mask=mask)


def _thnn_fused_lstm_cell(
    input_gates: torch.Tensor,
    hidden_gates: torch.Tensor,
    cx: torch.Tensor,
    input_bias: torch.Tensor = None,
    hidden_bias: torch.Tensor = None,
):
    """Fused LSTM cell computation.

    Args:
        input_gates: Input gates tensor of shape (batch, 4 * hidden_size)
        hidden_gates: Hidden gates tensor of shape (batch, 4 * hidden_size)
        cx: Cell state tensor of shape (batch, hidden_size)
        input_bias: Optional input bias of shape (4 * hidden_size)
        hidden_bias: Optional hidden bias of shape (4 * hidden_size)

    Returns:
        hy: New hidden state of shape (batch, hidden_size)
        cy: New cell state of shape (batch, hidden_size)
        workspace: Workspace tensor for backward of shape (batch, 4 * hidden_size)
    """
    logger.debug("GEMS _THNN_FUSED_LSTM_CELL")

    batch_size, gates_dim = input_gates.shape
    assert gates_dim % 4 == 0, "gates_dim must be divisible by 4"
    hidden_size = gates_dim // 4

    # Validate input shapes
    assert (
        hidden_gates.shape == input_gates.shape
    ), "hidden_gates must have same shape as input_gates"
    assert cx.shape == (
        batch_size,
        hidden_size,
    ), f"cx must have shape ({batch_size}, {hidden_size})"

    if input_bias is not None:
        assert input_bias.shape == (
            4 * hidden_size,
        ), f"input_bias must have shape ({4 * hidden_size},)"
    if hidden_bias is not None:
        assert hidden_bias.shape == (
            4 * hidden_size,
        ), f"hidden_bias must have shape ({4 * hidden_size},)"

    # Allocate outputs
    hy = torch.empty(
        (batch_size, hidden_size), device=input_gates.device, dtype=input_gates.dtype
    )
    cy = torch.empty(
        (batch_size, hidden_size), device=input_gates.device, dtype=input_gates.dtype
    )
    workspace = torch.empty(
        (batch_size, gates_dim), device=input_gates.device, dtype=input_gates.dtype
    )

    # Determine bias flags
    has_input_bias = input_bias is not None
    has_hidden_bias = hidden_bias is not None

    # Choose BLOCK_SIZE based on hidden_size
    BLOCK_SIZE = triton.next_power_of_2(hidden_size)
    BLOCK_SIZE = max(32, min(BLOCK_SIZE, 128))

    # Launch kernel: one program per (batch, hidden block); BLOCK_SIZE is
    # capped at 128, so hidden_size > BLOCK_SIZE is covered by the second
    # grid dimension instead of being truncated.
    grid = (batch_size, triton.cdiv(hidden_size, BLOCK_SIZE))
    _thnn_fused_lstm_cell_kernel[grid](
        input_gates,
        hidden_gates,
        cx,
        input_bias if input_bias is not None else input_gates.new_empty(0),
        hidden_bias if hidden_bias is not None else input_gates.new_empty(0),
        hy,
        cy,
        workspace,
        batch_size,
        hidden_size,
        has_input_bias,
        has_hidden_bias,
        BLOCK_SIZE,
        num_warps=4,
    )

    return hy, cy, workspace

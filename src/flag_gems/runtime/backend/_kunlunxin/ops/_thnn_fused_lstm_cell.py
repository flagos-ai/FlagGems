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
#
# Kunlunxin (XPU) override of ``aten::_thnn_fused_lstm_cell``.
#
# Why this file exists
# --------------------
# The generic implementation (``flag_gems/ops/_thnn_fused_lstm_cell.py``) maps
# one program to one batch element and covers the hidden dimension with a single
# ``tl.arange(0, BLOCK_SIZE)`` tile, where
#
#     BLOCK_SIZE = max(32, min(next_power_of_2(hidden_size), 128))
#
# so the tile width is *capped at 128 lanes*.  For ``hidden_size > 128`` the
# masked lanes (``offsets < hidden_size``) silently drop the columns
# ``[128, hidden_size)``: ``hy`` / ``cy`` / ``workspace`` are allocated with
# ``torch.empty`` and those columns are never written, so the operator returns
# **uninitialized memory** instead of raising.  Measured on XPU (2026-09-19):
#
#     hidden_size   128   129   160   200   256   512
#     wrong cy cols   0     1    32    72   128   384   (== hidden_size - 128)
#
# ``hidden_size`` of 256/512/1024 is the standard size of real LSTM layers, and
# the reference XPU kernel for the same op is exact on all of them
# (``maxdiff == 0`` against the composed-op semantics).  The official test
# matrix stops at ``hidden_size == 128``, i.e. exactly at the cap, which is why
# the defect is invisible there.
#
# The fix keeps the same per-batch tile shape (the ``workspace`` store stays
# affine/stride-1, which is what makes it cheap on this backend) and walks the
# hidden dimension in ``NUM_H`` chunks of ``BLOCK`` lanes.  ``hidden_size <= 128``
# gives ``NUM_H == 1`` and therefore an identical code path to the generic
# implementation, so the verified fast path is preserved.
#
# Input addresses are computed from the real tensor strides (including negative
# and broadcast/zero strides), which also removes the generic implementation's
# silent wrong values on non-contiguous inputs; the outputs are freshly
# allocated and kept contiguous.
#
# Second XPU constraint found while wiring the stride-aware loads: an
# innermost-stride != 1 load is **miscompiled at tile width 128** -- every odd
# lane returns the even lane's element, i.e. exactly half of ``cy`` is wrong
# (2026-09-19, XPU 5).  Measured with a fixed H=16 / B=4 fp32 input and the
# innermost stride swept over {2, 3, 5, 7, 10, 13}:
#
#     innermost stride   2    3    5    7   10   13
#     BLOCK=32 bad lanes 0    0    0    0    0    0
#     BLOCK=64 bad lanes 0    0    0    0    0    0
#     BLOCK=128 bad     32   32   32   32   32   32
#
# so the strided path is pinned to at most 64 lanes.  Contiguous inputs (the
# only layout the official matrix uses) keep the 128-lane fast path.
import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim
from flag_gems.utils.libentry import libentry

logger = logging.getLogger(__name__)

_tanh = tl_extra_shim.tanh

# Keep the unrolled hidden-chunk loop shallow: grow the tile instead so the
# static unroll never exceeds this many chunks.
_MAX_H_BLOCKS = 8
_MAX_BLOCK = 2048


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
    stride_ig_row,
    stride_ig_col,
    stride_hg_row,
    stride_hg_col,
    stride_cx_row,
    stride_cx_col,
    stride_ib,
    stride_hb,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_H: tl.constexpr,
    HAS_IB: tl.constexpr,
    HAS_HB: tl.constexpr,
):
    # Grid: (batch_size,); one program per batch element.  The outputs are
    # fresh contiguous allocations, so their addressing stays affine (stride-1)
    # in ``offs`` and keeps the block-DMA store path.
    batch_idx = tl.program_id(0)

    ig_row = input_gates_ptr + batch_idx * stride_ig_row
    hg_row = hidden_gates_ptr + batch_idx * stride_hg_row
    cx_row = cx_ptr + batch_idx * stride_cx_row
    hy_row = hy_ptr + batch_idx * H
    cy_row = cy_ptr + batch_idx * H
    ws_row = workspace_ptr + batch_idx * 4 * H

    for h_blk in tl.static_range(NUM_H):
        offs = h_blk * BLOCK + tl.arange(0, BLOCK)
        mask = offs < H

        cx_vals = tl.load(cx_row + offs * stride_cx_col, mask=mask, other=0.0).to(
            tl.float32
        )

        i_gate = tl.zeros([BLOCK], dtype=tl.float32)
        f_gate = tl.zeros([BLOCK], dtype=tl.float32)
        g_gate = tl.zeros([BLOCK], dtype=tl.float32)
        o_gate = tl.zeros([BLOCK], dtype=tl.float32)

        # gate_idx: 0=input (sigmoid), 1=forget (sigmoid), 2=cell (tanh),
        # 3=output (sigmoid).  The workspace keeps the activated gates in that
        # order, matching the ATen workspace contract consumed by the backward.
        for gate_idx in tl.static_range(4):
            col = gate_idx * H + offs

            ig = tl.load(ig_row + col * stride_ig_col, mask=mask, other=0.0).to(
                tl.float32
            )
            hg = tl.load(hg_row + col * stride_hg_col, mask=mask, other=0.0).to(
                tl.float32
            )

            if HAS_IB:
                ib = tl.load(input_bias_ptr + col * stride_ib, mask=mask, other=0.0).to(
                    tl.float32
                )
                ig = ig + ib
            if HAS_HB:
                hb = tl.load(
                    hidden_bias_ptr + col * stride_hb, mask=mask, other=0.0
                ).to(tl.float32)
                hg = hg + hb

            g = ig + hg

            if gate_idx == 0:
                i_gate = tl.sigmoid(g)
                activated_gate = i_gate
            elif gate_idx == 1:
                f_gate = tl.sigmoid(g)
                activated_gate = f_gate
            elif gate_idx == 2:
                g_gate = _tanh(g)
                activated_gate = g_gate
            else:
                o_gate = tl.sigmoid(g)
                activated_gate = o_gate

            tl.store(
                ws_row + col,
                activated_gate.to(workspace_ptr.dtype.element_ty),
                mask=mask,
            )

        cy_acc = f_gate * cx_vals + i_gate * g_gate
        hy_acc = o_gate * _tanh(cy_acc)

        tl.store(cy_row + offs, cy_acc.to(cy_ptr.dtype.element_ty), mask=mask)
        tl.store(hy_row + offs, hy_acc.to(hy_ptr.dtype.element_ty), mask=mask)


def _pick_tile(hidden_size, strided):
    """Return ``(BLOCK, NUM_H)`` covering ``hidden_size`` columns."""
    if strided:
        # Non-unit innermost stride: stay at or below the 64-lane tile width
        # that is not affected by the strided-load miscompile documented above.
        block = min(64, max(32, triton.next_power_of_2(hidden_size)))
        return block, triton.cdiv(hidden_size, block)

    block = triton.next_power_of_2(hidden_size)
    block = max(32, min(block, 128))
    num_h = triton.cdiv(hidden_size, block)
    if num_h > _MAX_H_BLOCKS:
        block = min(
            triton.next_power_of_2(triton.cdiv(hidden_size, _MAX_H_BLOCKS)), _MAX_BLOCK
        )
        block = max(block, 32)
        num_h = triton.cdiv(hidden_size, block)
    return block, num_h


def _thnn_fused_lstm_cell(
    input_gates: torch.Tensor,
    hidden_gates: torch.Tensor,
    cx: torch.Tensor,
    input_bias: torch.Tensor = None,
    hidden_bias: torch.Tensor = None,
):
    """Fused LSTM cell computation (Kunlunxin override).

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
    logger.debug("GEMS_KUNLUNXIN _THNN_FUSED_LSTM_CELL")

    batch_size, gates_dim = input_gates.shape
    assert gates_dim % 4 == 0, "gates_dim must be divisible by 4"
    hidden_size = gates_dim // 4

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

    hy = torch.empty(
        (batch_size, hidden_size), device=input_gates.device, dtype=input_gates.dtype
    )
    cy = torch.empty(
        (batch_size, hidden_size), device=input_gates.device, dtype=input_gates.dtype
    )
    workspace = torch.empty(
        (batch_size, gates_dim), device=input_gates.device, dtype=input_gates.dtype
    )

    has_input_bias = input_bias is not None
    has_hidden_bias = hidden_bias is not None

    # A non-unit innermost stride forces the narrower tile (see the notes at the
    # top of this file).
    strided = (
        input_gates.stride(1) != 1
        or hidden_gates.stride(1) != 1
        or cx.stride(1) != 1
        or (has_input_bias and input_bias.stride(0) != 1)
        or (has_hidden_bias and hidden_bias.stride(0) != 1)
    )

    BLOCK, num_h = _pick_tile(hidden_size, strided)

    dummy = input_gates.new_empty(0)
    _thnn_fused_lstm_cell_kernel[(batch_size,)](
        input_gates,
        hidden_gates,
        cx,
        input_bias if has_input_bias else dummy,
        hidden_bias if has_hidden_bias else dummy,
        hy,
        cy,
        workspace,
        input_gates.stride(0),
        input_gates.stride(1),
        hidden_gates.stride(0),
        hidden_gates.stride(1),
        cx.stride(0),
        cx.stride(1),
        input_bias.stride(0) if has_input_bias else 1,
        hidden_bias.stride(0) if has_hidden_bias else 1,
        H=hidden_size,
        BLOCK=BLOCK,
        NUM_H=num_h,
        HAS_IB=has_input_bias,
        HAS_HB=has_hidden_bias,
        num_warps=4,
    )

    return hy, cy, workspace

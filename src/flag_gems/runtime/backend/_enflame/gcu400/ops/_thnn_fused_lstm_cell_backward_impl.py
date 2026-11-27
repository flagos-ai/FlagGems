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
Here all outputs are produced by a single Triton launch.

Two kernels cover the input space:

* ``_lstm_cell_backward_flat_kernel`` — fast path for power-of-two hidden sizes with ``B*H <= FLAT_BLOCK_MAX``:
  one CTA covers every (row, column) element in a single linear range, so all load/store offsets stay
  1-D affine and the only predicate is the flat bounds check; the bias reduction reshapes the tile back to
  ``(BLOCK // H, H)`` and sums over rows in registers. This removes the 2-D tile address/predicate tax that
  dominates at the small LSTM-cell shapes (measured ~5.5us per launch on GCU400).
  NOTE: the column index must be derived with ``offs % H``; the mathematically equivalent
  ``offs - (offs // H) * H`` form miscompiles on Triton GCU 4.1.6 (masked gate stores silently dropped /
  wrong values once row > 0), archived as a P13 index-derived-addressing instance
  (flaggems_enflame_lstm_bwd_mulsub_index).
* ``_lstm_cell_backward_fused_kernel`` — generic 2-D tiled fallback (any H, any B): CTAs are partitioned by
  hidden column block and loop over row chunks of at most 64 rows, keeping the bias partials in registers,
  so the per-CTA tile (and DSM footprint) stays bounded for arbitrarily large batch sizes.

``FUSED_TRITON=0`` keeps a PyTorch path word-for-word equivalent to the common implementation as baseline and fallback;
after optimization the main path is fixed to the ``FUSED_TRITON=1`` Triton kernel.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---
# Compile-time config, adjusted per optimization round; semantics unchanged.
FUSED_TRITON = 1
FLAT_BLOCK_MAX = 2048
BLOCK_B = 32
BLOCK_H = 32
NUM_WARPS = 2
# 0: exp form (nan for large negative cy), 1: 2*sigmoid(2x)-1 (loses precision near 0),
# 2: stable exp form on |cy| (default).
TANH_MODE = 2
# Both kernels select the tanh formulation with a single boolean: only mode 1
# computes tanh via sigmoid; modes 0/2 use the exponential form, which is the
# closest in-device match to the public `torch.tanh(cy)` reference.
TANH_VIA_SIGMOID = TANH_MODE == 1
# --- TUNING PARAMS END ---


def _is_pow2(n):
    return n > 0 and (n & (n - 1)) == 0


@libentry()
@triton.jit(do_not_specialize=["B"])
def _lstm_cell_backward_flat_kernel(
    grad_hy_ptr,
    grad_cy_ptr,
    cx_ptr,
    cy_ptr,
    workspace_ptr,
    grad_gates_ptr,
    grad_cx_ptr,
    grad_bias_ptr,
    B,
    H: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
    TANH_VIA_SIGMOID: tl.constexpr,
):
    # Single linear range over whole rows: gate g of row r lives at r*4H + g*H + c,
    # so the flat offset addresses both the hidden tensors directly and the gates
    # through an affine remap. The only mask is the flat bounds check; masked lanes
    # load 0.0 and therefore reduce to exactly 0 in the bias sums.
    offs = tl.arange(0, BLOCK)
    m = offs < B * H
    # Column index must use % (see module docstring: the `offs - row * H`
    # mul-sub form miscompiles on Triton GCU 4.1.6 for row > 0).
    row = offs // H
    col = offs % H
    gate_off = row * (4 * H) + col

    grad_hy = tl.load(grad_hy_ptr + offs, mask=m, other=0.0).to(tl.float32)
    grad_cy = tl.load(grad_cy_ptr + offs, mask=m, other=0.0).to(tl.float32)
    cx = tl.load(cx_ptr + offs, mask=m, other=0.0).to(tl.float32)
    cy = tl.load(cy_ptr + offs, mask=m, other=0.0).to(tl.float32)
    i_gate = tl.load(workspace_ptr + gate_off, mask=m, other=0.0).to(tl.float32)
    f_gate = tl.load(workspace_ptr + gate_off + H, mask=m, other=0.0).to(tl.float32)
    g_gate = tl.load(workspace_ptr + gate_off + 2 * H, mask=m, other=0.0).to(tl.float32)
    o_gate = tl.load(workspace_ptr + gate_off + 3 * H, mask=m, other=0.0).to(tl.float32)

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

    tl.store(grad_gates_ptr + gate_off, grad_i, mask=m)
    tl.store(grad_gates_ptr + gate_off + H, grad_f, mask=m)
    tl.store(grad_gates_ptr + gate_off + 2 * H, grad_g, mask=m)
    tl.store(grad_gates_ptr + gate_off + 3 * H, grad_o, mask=m)
    tl.store(grad_cx_ptr + offs, d_cy * f_gate, mask=m)

    # The tile covers BLOCK // H complete rows (BLOCK is a multiple of H and at
    # least B*H), so a plain reshape recovers the (rows, columns) view for the
    # over-batch reduction; padded rows are zero and drop out of the sums.
    if HAS_BIAS:
        ROWS: tl.constexpr = BLOCK // H
        cols_h = tl.arange(0, H)
        tl.store(grad_bias_ptr + cols_h, tl.sum(tl.reshape(grad_i, (ROWS, H)), axis=0))
        tl.store(
            grad_bias_ptr + H + cols_h, tl.sum(tl.reshape(grad_f, (ROWS, H)), axis=0)
        )
        tl.store(
            grad_bias_ptr + 2 * H + cols_h,
            tl.sum(tl.reshape(grad_g, (ROWS, H)), axis=0),
        )
        tl.store(
            grad_bias_ptr + 3 * H + cols_h,
            tl.sum(tl.reshape(grad_o, (ROWS, H)), axis=0),
        )


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
    ATOMIC_BIAS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_H: tl.constexpr,
    TANH_VIA_SIGMOID: tl.constexpr,
):
    # 2D tile: rows cover a chunk of batch rows, cols cover a contiguous range of hidden columns.
    # workspace is laid out contiguously along dim=1 as [i, f, g, o], so the four base addresses differ by H.
    # The row loop keeps the tile bounded for arbitrarily large B; for B <= BLOCK_B it runs once and matches
    # the previous non-looped kernel exactly.
    cols = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
    col_mask = cols < H
    if HAS_BIAS:
        acc_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
        acc_f = tl.zeros((BLOCK_H,), dtype=tl.float32)
        acc_g = tl.zeros((BLOCK_H,), dtype=tl.float32)
        acc_o = tl.zeros((BLOCK_H,), dtype=tl.float32)

    for row_base in range(0, B, BLOCK_B):
        rows = row_base + tl.arange(0, BLOCK_B)
        valid = (rows[:, None] < B) & col_mask[None, :]
        hidden_offset = rows[:, None] * H + cols[None, :]
        gate_offset = rows[:, None] * (4 * H) + cols[None, :]

        grad_hy = tl.load(grad_hy_ptr + hidden_offset, mask=valid, other=0.0).to(
            tl.float32
        )
        grad_cy = tl.load(grad_cy_ptr + hidden_offset, mask=valid, other=0.0).to(
            tl.float32
        )
        cx = tl.load(cx_ptr + hidden_offset, mask=valid, other=0.0).to(tl.float32)
        cy = tl.load(cy_ptr + hidden_offset, mask=valid, other=0.0).to(tl.float32)
        i_gate = tl.load(workspace_ptr + gate_offset, mask=valid, other=0.0).to(
            tl.float32
        )
        f_gate = tl.load(workspace_ptr + gate_offset + H, mask=valid, other=0.0).to(
            tl.float32
        )
        g_gate = tl.load(workspace_ptr + gate_offset + 2 * H, mask=valid, other=0.0).to(
            tl.float32
        )
        o_gate = tl.load(workspace_ptr + gate_offset + 3 * H, mask=valid, other=0.0).to(
            tl.float32
        )

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

        if HAS_BIAS:
            acc_i += tl.sum(grad_i, axis=0)
            acc_f += tl.sum(grad_f, axis=0)
            acc_g += tl.sum(grad_g, axis=0)
            acc_o += tl.sum(grad_o, axis=0)

    # The bias gradient is the column sum of grad_input_gates over batch; masked-out elements are already zero,
    # so it is reduced over rows inside the same kernel without atomics or a second launch.
    if HAS_BIAS:
        tl.store(grad_bias_ptr + cols, acc_i, mask=col_mask)
        tl.store(grad_bias_ptr + H + cols, acc_f, mask=col_mask)
        tl.store(grad_bias_ptr + 2 * H + cols, acc_g, mask=col_mask)
        tl.store(grad_bias_ptr + 3 * H + cols, acc_o, mask=col_mask)


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

    tanh_cy = torch.tanh(cy)
    d_cy = grad_hy * o_gate * (1.0 - tanh_cy * tanh_cy) + grad_cy
    grad_i = d_cy * g_gate * i_gate * (1.0 - i_gate)
    grad_f = d_cy * cx * f_gate * (1.0 - f_gate)
    grad_g = d_cy * i_gate * (1.0 - g_gate * g_gate)
    grad_o = grad_hy * tanh_cy * o_gate * (1.0 - o_gate)
    grad_input_gates = torch.cat([grad_i, grad_f, grad_g, grad_o], dim=1)
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
    logger.debug("GEMS_ENFLAME _THNN_FUSED_LSTM_CELL_BACKWARD_IMPL")
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

    with torch_device_fn.device(cx.device):
        # B == 0 must stay on the tiled path: next_power_of_2(0) collapses BLOCK
        # below one row and breaks the reshape, while the 2-D kernel handles the
        # empty batch with zero loop trips and zero bias partials.
        if (
            batch_size > 0
            and _is_pow2(hidden_size)
            and batch_size * hidden_size <= FLAT_BLOCK_MAX
        ):
            # Fast path: one CTA, flat linear addressing, in-tile bias reduction.
            _lstm_cell_backward_flat_kernel[(1,)](
                grad_hy,
                grad_cy,
                cx,
                cy,
                workspace,
                grad_input_gates,
                grad_cx,
                grad_biases,
                batch_size,
                H=hidden_size,
                HAS_BIAS=bool(has_bias),
                BLOCK=triton.next_power_of_2(batch_size * hidden_size),
                TANH_VIA_SIGMOID=TANH_VIA_SIGMOID,
                num_warps=NUM_WARPS,
            )
        else:
            # Generic 2-D tiled fallback. BLOCK_B is capped so the per-CTA tile stays
            # within the DSM budget; the kernel loops over row chunks for large B.
            # Grid cap: prevents exceeding GCU's hardware grid limit when BLOCK_H is small.
            grid_size = min(triton.cdiv(hidden_size, BLOCK_H), 65535)
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
                BLOCK_B=min(max(BLOCK_B, triton.next_power_of_2(batch_size)), 64),
                BLOCK_H=BLOCK_H,
                TANH_VIA_SIGMOID=TANH_VIA_SIGMOID,
                num_warps=NUM_WARPS,
            )
    return grad_input_gates, grad_cx, grad_biases

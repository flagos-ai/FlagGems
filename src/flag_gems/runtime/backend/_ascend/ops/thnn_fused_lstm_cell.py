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

import torch
import triton
import triton.language as tl


@triton.jit
def _thnn_fused_lstm_cell_kernel(
    input_gates_ptr,
    hidden_gates_ptr,
    cx_ptr,
    input_bias_ptr,
    hidden_bias_ptr,
    workspace_ptr,
    hy_ptr,
    cy_ptr,
    N,
    H: tl.constexpr,
    HAS_INPUT_BIAS: tl.constexpr,
    HAS_HIDDEN_BIAS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * ROWS_PER_PROG + tl.arange(0, ROWS_PER_PROG)  # (R,)
    cols = tl.arange(0, BLOCK_H)  # (BH,)

    valid = (rows[:, None] < N) & (cols[None, :] < H)

    r_off = rows[:, None] * (4 * H) + cols[None, :]  # chunk-0 base, (R, BH)
    c_off = rows[:, None] * H + cols[None, :]  # cell base, (R, BH)

    # ---- gate pre-activation sums (one contiguous chunk per gate) ----
    g0 = tl.load(input_gates_ptr + r_off, mask=valid) + tl.load(
        hidden_gates_ptr + r_off, mask=valid
    )
    g1 = tl.load(input_gates_ptr + r_off + H, mask=valid) + tl.load(
        hidden_gates_ptr + r_off + H, mask=valid
    )
    g2 = tl.load(input_gates_ptr + r_off + 2 * H, mask=valid) + tl.load(
        hidden_gates_ptr + r_off + 2 * H, mask=valid
    )
    g3 = tl.load(input_gates_ptr + r_off + 3 * H, mask=valid) + tl.load(
        hidden_gates_ptr + r_off + 3 * H, mask=valid
    )

    if HAS_INPUT_BIAS:
        bcols = cols[None, :]
        g0 += tl.load(input_bias_ptr + bcols, mask=cols[None, :] < H)
        g1 += tl.load(input_bias_ptr + bcols + H, mask=cols[None, :] < H)
        g2 += tl.load(input_bias_ptr + bcols + 2 * H, mask=cols[None, :] < H)
        g3 += tl.load(input_bias_ptr + bcols + 3 * H, mask=cols[None, :] < H)

    if HAS_HIDDEN_BIAS:
        bcols = cols[None, :]
        g0 += tl.load(hidden_bias_ptr + bcols, mask=cols[None, :] < H)
        g1 += tl.load(hidden_bias_ptr + bcols + H, mask=cols[None, :] < H)
        g2 += tl.load(hidden_bias_ptr + bcols + 2 * H, mask=cols[None, :] < H)
        g3 += tl.load(hidden_bias_ptr + bcols + 3 * H, mask=cols[None, :] < H)

    tl.store(workspace_ptr + r_off, g0, mask=valid)
    tl.store(workspace_ptr + r_off + H, g1, mask=valid)
    tl.store(workspace_ptr + r_off + 2 * H, g2, mask=valid)
    tl.store(workspace_ptr + r_off + 3 * H, g3, mask=valid)

    # ---- cell update ----
    cx_v = tl.load(cx_ptr + c_off, mask=valid)

    i = tl.sigmoid(g0)
    f = tl.sigmoid(g1)
    gg = tl.math.tanh(g2)
    o = tl.sigmoid(g3)

    cy = f * cx_v + i * gg
    hy = o * tl.math.tanh(cy)

    tl.store(cy_ptr + c_off, cy, mask=valid)
    tl.store(hy_ptr + c_off, hy, mask=valid)


def thnn_fused_lstm_cell(input_gates, hidden_gates, cx, input_bias, hidden_bias):
    N = input_gates.shape[0]
    H = input_gates.shape[1] // 4
    BLOCK_H = triton.next_power_of_2(H)
    R = 4

    workspace = torch.empty_like(input_gates)
    hy = torch.empty_like(cx)
    cy = torch.empty_like(cx)

    ib = input_bias if input_bias is not None else cx
    hb = hidden_bias if hidden_bias is not None else cx

    _thnn_fused_lstm_cell_kernel[(triton.cdiv(N, R),)](
        input_gates,
        hidden_gates,
        cx,
        ib,
        hb,
        workspace,
        hy,
        cy,
        N,
        H=H,
        HAS_INPUT_BIAS=input_bias is not None,
        HAS_HIDDEN_BIAS=hidden_bias is not None,
        BLOCK_H=BLOCK_H,
        ROWS_PER_PROG=R,
    )
    return hy, cy, workspace

# Copyright 2026, The FlagOS Contributors.
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
"""GCU400 专属 ``_thnn_fused_lstm_cell_backward_impl``。

公共实现把 LSTM cell 反向拆成十余次 PyTorch 张量操作（切片、tanh、逐元素乘、
cat、按列求和），在 GCU 上每次 launch 的固定开销远大于实际计算量。这里按
hidden 列块划分 CTA：一个 program 负责若干列的全部 batch 行，同时写出四个门的
梯度、grad_cx，并在寄存器内对 batch 维归约得到 bias 梯度，从而用一次 Triton
launch 完成全部输出。

``FUSED_TRITON=0`` 保留与公共实现逐字等价的 PyTorch 路径，作为基线与回退；
优化完成后主路径固定为 ``FUSED_TRITON=1`` 的 Triton kernel。
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---
# 编译期配置，由优化迭代按轮次调整，不改变算子语义。
FUSED_TRITON = 1
BLOCK_B = 16
BLOCK_H = 64
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
    # 二维 tile：rows 覆盖 batch 行，cols 覆盖连续的一段 hidden 列。
    # workspace 沿 dim=1 按 [i, f, g, o] 连续排布，故四段基址相差 H。
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

    # tanh(cy) 用 2*sigmoid(2*cy)-1 的恒等式表达，避免依赖设备侧 libdevice。
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

    # bias 梯度是 grad_input_gates 沿 batch 维的列和；掩码外的元素已置零，
    # 因此可在同一 kernel 内对 rows 归约，无需 atomic 或第二次 launch。
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
    """与公共实现逐字等价的基线路径，仅在 ``FUSED_TRITON=0`` 时使用。"""
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
    """计算 fused LSTM cell 反向的三个输出，签名与公共实现一致。"""
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

    grid_size = triton.cdiv(hidden_size, BLOCK_H)
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

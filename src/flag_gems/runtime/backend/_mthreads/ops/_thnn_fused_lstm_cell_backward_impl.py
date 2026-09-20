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

from flag_gems.ops._thnn_fused_lstm_cell_backward_impl import (
    _thnn_fused_lstm_cell_backward_impl as default__thnn_fused_lstm_cell_backward_impl,
)

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@triton.jit
def _lstm_cell_bwd_impl_kernel(
    grad_hy_ptr,
    grad_cy_ptr,
    cx_ptr,
    cy_ptr,
    ws_ptr,
    grad_ig_ptr,
    grad_cx_ptr,
    grad_bias_ptr,
    B,
    H,
    HAS_BIAS: tl.constexpr,
    UPCAST: tl.constexpr,
    USE_FAST: tl.constexpr,
    EXACT: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_h = tl.program_id(0)
    r = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    if EXACT:
        mask_r = None
    else:
        mask_r = r < H

    comp_dtype = tl.float32 if UPCAST else ws_ptr.dtype.element_ty
    acc_i = tl.zeros((BLOCK_H,), dtype=comp_dtype)
    acc_f = tl.zeros((BLOCK_H,), dtype=comp_dtype)
    acc_g = tl.zeros((BLOCK_H,), dtype=comp_dtype)
    acc_o = tl.zeros((BLOCK_H,), dtype=comp_dtype)

    for b0 in range(0, B, BLOCK_B):
        bb = b0 + tl.arange(0, BLOCK_B)
        base = bb[:, None] * (4 * H)
        sbase = bb[:, None] * H
        if EXACT:
            i = tl.load(ws_ptr + base + r[None, :])
            f = tl.load(ws_ptr + base + H + r[None, :])
            g = tl.load(ws_ptr + base + 2 * H + r[None, :])
            o = tl.load(ws_ptr + base + 3 * H + r[None, :])
            cx = tl.load(cx_ptr + sbase + r[None, :])
            cy = tl.load(cy_ptr + sbase + r[None, :])
            ghy = tl.load(grad_hy_ptr + sbase + r[None, :])
            gcy = tl.load(grad_cy_ptr + sbase + r[None, :])
        else:
            m = (bb < B)[:, None] & mask_r[None, :]
            i = tl.load(ws_ptr + base + r[None, :], mask=m, other=0.0)
            f = tl.load(ws_ptr + base + H + r[None, :], mask=m, other=0.0)
            g = tl.load(ws_ptr + base + 2 * H + r[None, :], mask=m, other=0.0)
            o = tl.load(ws_ptr + base + 3 * H + r[None, :], mask=m, other=0.0)
            cx = tl.load(cx_ptr + sbase + r[None, :], mask=m, other=0.0)
            cy = tl.load(cy_ptr + sbase + r[None, :], mask=m, other=0.0)
            ghy = tl.load(grad_hy_ptr + sbase + r[None, :], mask=m, other=0.0)
            gcy = tl.load(grad_cy_ptr + sbase + r[None, :], mask=m, other=0.0)

        if UPCAST:
            i = i.to(tl.float32)
            f = f.to(tl.float32)
            g = g.to(tl.float32)
            o = o.to(tl.float32)
            cx = cx.to(tl.float32)
            cy = cy.to(tl.float32)
            ghy = ghy.to(tl.float32)
            gcy = gcy.to(tl.float32)

        one = 1.0
        if USE_FAST:
            tcy = tl.extra.musa.libdevice.fast_tanh(cy)
        else:
            tcy = tl.extra.musa.libdevice.tanh(cy)
        gc = gcy + ghy * o * (one - tcy * tcy)
        di = gc * g * (one - i) * i
        df = gc * cx * (one - f) * f
        dg = gc * i * (one - g * g)
        do = ghy * tcy * (one - o) * o
        gcx = gc * f

        if EXACT:
            tl.store(grad_ig_ptr + base + r[None, :], di)
            tl.store(grad_ig_ptr + base + H + r[None, :], df)
            tl.store(grad_ig_ptr + base + 2 * H + r[None, :], dg)
            tl.store(grad_ig_ptr + base + 3 * H + r[None, :], do)
            tl.store(grad_cx_ptr + sbase + r[None, :], gcx)
        else:
            tl.store(grad_ig_ptr + base + r[None, :], di, mask=m)
            tl.store(grad_ig_ptr + base + H + r[None, :], df, mask=m)
            tl.store(grad_ig_ptr + base + 2 * H + r[None, :], dg, mask=m)
            tl.store(grad_ig_ptr + base + 3 * H + r[None, :], do, mask=m)
            tl.store(grad_cx_ptr + sbase + r[None, :], gcx, mask=m)

        acc_i += tl.sum(di, axis=0)
        acc_f += tl.sum(df, axis=0)
        acc_g += tl.sum(dg, axis=0)
        acc_o += tl.sum(do, axis=0)

    if HAS_BIAS:
        if EXACT:
            tl.store(grad_bias_ptr + r, acc_i)
            tl.store(grad_bias_ptr + H + r, acc_f)
            tl.store(grad_bias_ptr + 2 * H + r, acc_g)
            tl.store(grad_bias_ptr + 3 * H + r, acc_o)
        else:
            tl.store(grad_bias_ptr + r, acc_i, mask=mask_r)
            tl.store(grad_bias_ptr + H + r, acc_f, mask=mask_r)
            tl.store(grad_bias_ptr + 2 * H + r, acc_g, mask=mask_r)
            tl.store(grad_bias_ptr + 3 * H + r, acc_o, mask=mask_r)


_BLOCK_B_MAX = 16
_BLOCK_H_MAX = 16


def _specialized__thnn_fused_lstm_cell_backward_impl(
    grad_hy, grad_cy, cx, cy, workspace, has_bias
):
    B, H = cx.shape
    grad_input_gates = torch.empty_like(workspace)
    grad_cx = torch.empty_like(cx)
    if has_bias:
        grad_biases = torch.empty(4 * H, device=cx.device, dtype=cx.dtype)
    else:
        grad_biases = None
    block_b = min(triton.next_power_of_2(B), _BLOCK_B_MAX)
    block_h = min(triton.next_power_of_2(H), _BLOCK_H_MAX)
    num_warps = max(1, min(32, (block_b * block_h) // 32))
    exact = (B == block_b) and (H % block_h == 0) and (B % block_b == 0)
    _lstm_cell_bwd_impl_kernel[(triton.cdiv(H, block_h),)](
        grad_hy,
        grad_cy,
        cx,
        cy,
        workspace,
        grad_input_gates,
        grad_cx,
        grad_biases if has_bias else grad_input_gates,
        B,
        H,
        HAS_BIAS=has_bias,
        UPCAST=cx.dtype in (torch.float16, torch.bfloat16),
        USE_FAST=cx.dtype != torch.float64,
        EXACT=exact,
        BLOCK_B=block_b,
        BLOCK_H=block_h,
        num_warps=num_warps,
    )
    return grad_input_gates, grad_cx, grad_biases


def _thnn_fused_lstm_cell_backward_impl(grad_hy, grad_cy, cx, cy, workspace, has_bias):
    logger.debug("GEMS_MTHREADS _THNN_FUSED_LSTM_CELL_BACKWARD_IMPL")
    if (
        isinstance(grad_hy, torch.Tensor)
        and grad_hy.device.type == "musa"
        and grad_hy.dtype in _SUPPORTED_DTYPES
        and isinstance(grad_cy, torch.Tensor)
        and grad_cy.device.type == "musa"
        and grad_cy.dtype in _SUPPORTED_DTYPES
        and isinstance(cx, torch.Tensor)
        and cx.device.type == "musa"
        and cx.dtype in _SUPPORTED_DTYPES
        and isinstance(cy, torch.Tensor)
        and cy.device.type == "musa"
        and cy.dtype in _SUPPORTED_DTYPES
        and isinstance(workspace, torch.Tensor)
        and workspace.device.type == "musa"
        and workspace.dtype in _SUPPORTED_DTYPES
    ):
        return _specialized__thnn_fused_lstm_cell_backward_impl(
            grad_hy, grad_cy, cx, cy, workspace, has_bias
        )
    return default__thnn_fused_lstm_cell_backward_impl(
        grad_hy, grad_cy, cx, cy, workspace, has_bias
    )

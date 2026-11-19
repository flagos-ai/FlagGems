# Copyright 2026 The FlagOS Contributors
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

"""Single-step tanh RNN cell, kunlunxin override.

Three things the shared kernel cannot do unchanged on this backend:

* its activation is ``tl_extra_shim.tanh``, which lowers to a libdevice extern
  whose elementary-function lowering is not available for every shape the
  backend asks it to serve, so the kernel fails to compile;
* evaluating ``tanh`` through ``tl.exp`` *inside the fused tile kernel* is not
  reliable either: measured on this backend, the exponential of a positive
  argument comes back as ``1`` there, which silently zeroes every positive
  activation while leaving the negative ones exact.  The same expression in a
  flat elementwise kernel is correct, so the cell is split: the two projections
  and their sum are produced by the tile kernel, and the activation runs in a
  separate flat pass;
* when ``input_size == 0`` the input projection's ``tl.dot`` reduction makes
  zero trips, which this backend compiles to an undefined accumulator (the
  whole output tile is garbage).  A one-column zero input and a matching zero
  weight column are exactly equivalent -- the projection contributes only its
  bias -- so the reduction then has a single trip.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.rnn_tanh_cell import (
    _ACC_DTYPE,
    _ROUND_DTYPE,
    _block,
    _check_cell_args,
    _round,
)
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_ACT_BLOCK = 1024


@libentry()
@triton.jit
def _rnn_tanh_cell_gates_kernel(
    x_ptr,
    h_ptr,
    w_ih_ptr,
    w_hh_ptr,
    b_ih_ptr,
    b_hh_ptr,
    z_ptr,
    B,
    I,
    H,
    s_xb,
    s_xk,
    s_hb,
    s_hk,
    s_wi_n,
    s_wi_k,
    s_wh_n,
    s_wh_k,
    HAS_BIH: tl.constexpr,
    HAS_BHH: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    ROUND_DTYPE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_b = tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_n = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    b_mask = offs_b < B
    n_mask = offs_n < H

    # ---- linear(input, w_ih, b_ih): (B, I) @ (I, H) ----------------------
    acc_i = tl.zeros((BLOCK_B, BLOCK_H), dtype=ACC_DTYPE)
    for k0 in range(0, I, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < I
        x = tl.load(
            x_ptr + offs_b[:, None] * s_xb + offs_k[None, :] * s_xk,
            mask=b_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        w = tl.load(
            w_ih_ptr + offs_n[None, :] * s_wi_n + offs_k[:, None] * s_wi_k,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        acc_i = tl.dot(x, w, acc_i, input_precision="ieee", out_dtype=ACC_DTYPE)
    if HAS_BIH:
        acc_i += tl.load(b_ih_ptr + offs_n, mask=n_mask, other=0.0).to(ACC_DTYPE)[
            None, :
        ]
    gate_i = _round(acc_i, ROUND_DTYPE, ACC_DTYPE)

    # ---- linear(hx, w_hh, b_hh): (B, H) @ (H, H) -------------------------
    acc_h = tl.zeros((BLOCK_B, BLOCK_H), dtype=ACC_DTYPE)
    for k0 in range(0, H, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < H
        hx = tl.load(
            h_ptr + offs_b[:, None] * s_hb + offs_k[None, :] * s_hk,
            mask=b_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        w = tl.load(
            w_hh_ptr + offs_n[None, :] * s_wh_n + offs_k[:, None] * s_wh_k,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        acc_h = tl.dot(hx, w, acc_h, input_precision="ieee", out_dtype=ACC_DTYPE)
    if HAS_BHH:
        acc_h += tl.load(b_hh_ptr + offs_n, mask=n_mask, other=0.0).to(ACC_DTYPE)[
            None, :
        ]
    gate_h = _round(acc_h, ROUND_DTYPE, ACC_DTYPE)

    z = _round(gate_i + gate_h, ROUND_DTYPE, ACC_DTYPE)
    tl.store(
        z_ptr + offs_b[:, None] * H + offs_n[None, :],
        z.to(tl.float32),
        mask=b_mask[:, None] & n_mask[None, :],
    )


@libentry()
@triton.jit
def _rnn_tanh_cell_act_kernel(z_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    z = tl.load(z_ptr + offsets, mask=mask, other=0.0)
    # tanh(z) = 1 - 2 / (exp(2z) + 1); saturates correctly for |z| -> inf.
    hy = 1.0 - 2.0 / (tl.exp(2.0 * z) + 1.0)
    tl.store(out_ptr + offsets, hy.to(out_ptr.dtype.element_ty), mask=mask)


def rnn_tanh_cell(input, hx, w_ih, w_hh, b_ih=None, b_hh=None):
    logger.debug("GEMS_KUNLUNXIN RNN_TANH_CELL")
    _check_cell_args("rnn_tanh_cell", input, hx, w_ih, w_hh, b_ih, b_hh, 1)

    if input.dim() >= 2 and input.shape[1] == 0:
        input = torch.zeros((input.shape[0], 1), dtype=input.dtype, device=input.device)
        w_ih = torch.zeros((w_ih.shape[0], 1), dtype=w_ih.dtype, device=w_ih.device)

    batch, input_size = input.shape
    hidden_size = w_hh.shape[1]
    out = torch.empty((batch, hidden_size), dtype=input.dtype, device=input.device)
    if batch == 0 or hidden_size == 0:
        return out

    block_b = _block(batch, 16)
    block_h = _block(hidden_size, 32)
    block_k = _block(max(input_size, hidden_size), 64)

    n_elements = batch * hidden_size
    z = torch.empty((batch, hidden_size), dtype=torch.float32, device=input.device)
    _rnn_tanh_cell_gates_kernel[
        (triton.cdiv(batch, block_b), triton.cdiv(hidden_size, block_h))
    ](
        input,
        hx,
        w_ih,
        w_hh,
        b_ih if b_ih is not None else input,
        b_hh if b_hh is not None else input,
        z,
        batch,
        input_size,
        hidden_size,
        input.stride(0),
        input.stride(1),
        hx.stride(0),
        hx.stride(1),
        w_ih.stride(0),
        w_ih.stride(1),
        w_hh.stride(0),
        w_hh.stride(1),
        HAS_BIH=b_ih is not None,
        HAS_BHH=b_hh is not None,
        ACC_DTYPE=_ACC_DTYPE[input.dtype],
        ROUND_DTYPE=_ROUND_DTYPE[input.dtype],
        BLOCK_B=block_b,
        BLOCK_H=block_h,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=3,
    )
    _rnn_tanh_cell_act_kernel[(triton.cdiv(n_elements, _ACT_BLOCK),)](
        z,
        out,
        n_elements,
        BLOCK=_ACT_BLOCK,
        num_warps=4,
    )
    return out

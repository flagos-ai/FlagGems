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

"""Kunlunxin backend override for ``lstm_cell``.

The generic implementation is functionally correct but slow on XPU for two
reasons, both fixed here:

1. The generic combine kernel shares the projection kernels' ``BLOCK_B=16`` /
   ``BLOCK_H=32`` tile.  That tile is tuned for the GEMMs (four live
   accumulator tiles), but for the elementwise gate chain it is far too narrow:
   each program issues eight masked 32-wide strided gate loads, and the poor
   coalescing makes the combine kernel dominate the whole operator (pre-
   profiling: combine ~2.9 ms vs ih 0.14 ms + hh 0.10 ms at 128x512x512).
   This override launches the combine kernel with its own wide tile (one batch
   row per program, hidden tile up to 256) so the four packed gate slices of
   ``gi``/``gh`` are loaded as wide contiguous chunks.

2. The generic combine kernel calls ``tl_extra_shim.tanh``, which lowers to the
   slow XPU libdevice extern ``_ZN3xpu5tanhfEf`` / ``_ZN3xpu5htanhEDF16_``.
   This override expresses ``tanh`` through the core ``tl.exp`` fast path
   (``tanh(x) = 2*sigmoid(2x) - 1``), the same identity ``tl.sigmoid`` already
   uses (``1/(1+exp(-x))``), so it matches ATen's ``torch.tanh`` within the
   tolerance the accuracy tests already accept for the sigmoid terms.

The projection GEMMs (ih/hh) are reused unchanged; they are already fast.
"""

import torch
import triton
import triton.language as tl

from flag_gems.ops.lstm_cell import (
    _ACC_DTYPE,
    _ROUND_DTYPE,
    _block,
    _check_cell_args,
    _lstm_cell_hh_kernel,
    _lstm_cell_ih_kernel,
)

__all__ = ["lstm_cell"]

_COMBINE_BLOCK_H_LIMIT = 256


@triton.jit(do_not_specialize=["B", "H"])
def _lstm_cell_combine_kernel_xpu(
    gi_ptr,
    gh_ptr,
    c_ptr,
    b_ih_ptr,
    b_hh_ptr,
    hy_ptr,
    cy_ptr,
    B,
    H,
    s_gb,
    s_gn,
    s_cb,
    s_ck,
    s_ob,
    s_on,
    HAS_BIH: tl.constexpr,
    HAS_BHH: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    NEED_MASK: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # One batch row per program, wide tile over the hidden dimension.
    offs_b = tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_n = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    b_mask = offs_b < B
    n_mask = offs_n < H

    row_i = offs_n
    row_f = offs_n + H
    row_g = offs_n + 2 * H
    row_o = offs_n + 3 * H

    out_mask = b_mask[:, None] & n_mask[None, :]
    base = offs_b[:, None] * s_gb
    if NEED_MASK:
        gate_i = tl.load(
            gi_ptr + base + row_i[None, :] * s_gn, mask=out_mask, other=0.0
        )
        gate_f = tl.load(
            gi_ptr + base + row_f[None, :] * s_gn, mask=out_mask, other=0.0
        )
        gate_g = tl.load(
            gi_ptr + base + row_g[None, :] * s_gn, mask=out_mask, other=0.0
        )
        gate_o = tl.load(
            gi_ptr + base + row_o[None, :] * s_gn, mask=out_mask, other=0.0
        )
        gate_i += tl.load(
            gh_ptr + base + row_i[None, :] * s_gn, mask=out_mask, other=0.0
        )
        gate_f += tl.load(
            gh_ptr + base + row_f[None, :] * s_gn, mask=out_mask, other=0.0
        )
        gate_g += tl.load(
            gh_ptr + base + row_g[None, :] * s_gn, mask=out_mask, other=0.0
        )
        gate_o += tl.load(
            gh_ptr + base + row_o[None, :] * s_gn, mask=out_mask, other=0.0
        )
    else:
        gate_i = tl.load(gi_ptr + base + row_i[None, :] * s_gn)
        gate_f = tl.load(gi_ptr + base + row_f[None, :] * s_gn)
        gate_g = tl.load(gi_ptr + base + row_g[None, :] * s_gn)
        gate_o = tl.load(gi_ptr + base + row_o[None, :] * s_gn)
        gate_i += tl.load(gh_ptr + base + row_i[None, :] * s_gn)
        gate_f += tl.load(gh_ptr + base + row_f[None, :] * s_gn)
        gate_g += tl.load(gh_ptr + base + row_g[None, :] * s_gn)
        gate_o += tl.load(gh_ptr + base + row_o[None, :] * s_gn)

    if HAS_BIH:
        gate_i += tl.load(b_ih_ptr + row_i, mask=n_mask, other=0.0).to(ACC_DTYPE)[
            None, :
        ]
        gate_f += tl.load(b_ih_ptr + row_f, mask=n_mask, other=0.0).to(ACC_DTYPE)[
            None, :
        ]
        gate_g += tl.load(b_ih_ptr + row_g, mask=n_mask, other=0.0).to(ACC_DTYPE)[
            None, :
        ]
        gate_o += tl.load(b_ih_ptr + row_o, mask=n_mask, other=0.0).to(ACC_DTYPE)[
            None, :
        ]
    if HAS_BHH:
        gate_i += tl.load(b_hh_ptr + row_i, mask=n_mask, other=0.0).to(ACC_DTYPE)[
            None, :
        ]
        gate_f += tl.load(b_hh_ptr + row_f, mask=n_mask, other=0.0).to(ACC_DTYPE)[
            None, :
        ]
        gate_g += tl.load(b_hh_ptr + row_g, mask=n_mask, other=0.0).to(ACC_DTYPE)[
            None, :
        ]
        gate_o += tl.load(b_hh_ptr + row_o, mask=n_mask, other=0.0).to(ACC_DTYPE)[
            None, :
        ]

    # This backend's tl.sigmoid/tl.exp only accept fp32/fp64, so promote the
    # gates explicitly (they only widen implicitly when a bias is present).
    gate_i = gate_i.to(ACC_DTYPE)
    gate_f = gate_f.to(ACC_DTYPE)
    gate_g = gate_g.to(ACC_DTYPE)
    gate_o = gate_o.to(ACC_DTYPE)

    c_prev = tl.load(
        c_ptr + offs_b[:, None] * s_cb + offs_n[None, :] * s_ck,
        mask=out_mask,
        other=0.0,
    ).to(ACC_DTYPE)

    # Fast tanh through the core tl.exp path (tanh(x) = 2*sigmoid(2x) - 1),
    # replacing tl_extra_shim.tanh which lowers to the slow libdevice extern.
    cy = tl.sigmoid(gate_f) * c_prev + tl.sigmoid(gate_i) * (
        2.0 * tl.sigmoid(2.0 * gate_g) - 1.0
    )
    hy = tl.sigmoid(gate_o) * (2.0 * tl.sigmoid(2.0 * cy) - 1.0)

    out_off = offs_b[:, None] * s_ob + offs_n[None, :] * s_on
    tl.store(hy_ptr + out_off, hy.to(hy_ptr.dtype.element_ty), mask=out_mask)
    tl.store(cy_ptr + out_off, cy.to(cy_ptr.dtype.element_ty), mask=out_mask)


def _combine_block_h(hidden_size):
    """Hidden tile for the elementwise combine kernel.

    Wide and contiguous is what matters here (no ``tl.dot`` accumulator tiles),
    so unlike the projection kernels the hidden tile is not clamped to 32.
    """
    return min(triton.next_power_of_2(max(int(hidden_size), 1)), _COMBINE_BLOCK_H_LIMIT)


def lstm_cell(input, hx, w_ih, w_hh, b_ih=None, b_hh=None):
    """Single LSTM cell step (Kunlunxin override, wide-tile combine kernel)."""
    if len(hx) != 2:
        raise RuntimeError("lstm_cell expects two hidden states")
    h_prev, c_prev = hx[0], hx[1]
    _check_cell_args("lstm_cell", input, h_prev, w_ih, w_hh, b_ih, b_hh, 4)
    if c_prev.dim() != 2 or c_prev.shape[0] != input.shape[0]:
        raise RuntimeError(
            "Input batch size %d doesn't match hidden1 batch size %d"
            % (input.shape[0], c_prev.shape[0])
        )
    if c_prev.shape[1] != w_hh.shape[1]:
        raise RuntimeError(
            "hidden1 has inconsistent hidden_size: got %d, expected %d"
            % (c_prev.shape[1], w_hh.shape[1])
        )

    batch, input_size = input.shape
    hidden_size = w_hh.shape[1]
    hy = torch.empty((batch, hidden_size), dtype=input.dtype, device=input.device)
    cy = torch.empty((batch, hidden_size), dtype=input.dtype, device=input.device)
    if batch == 0 or hidden_size == 0:
        return hy, cy

    block_b = _block(batch, 16)
    block_h = _block(hidden_size, 32)
    # XPU's Triton pipeline crashes ("Inliner" / pipeline-event limit) when a
    # kernel carries four ``tl.dot`` ops in one loop with a K tile below 64, so
    # the reduction tile is fixed at 64 and the edges are masked instead.
    block_k = 64
    grid = (triton.cdiv(batch, block_b), triton.cdiv(hidden_size, block_h))

    # Rounded ``linear(input, w_ih)`` and ``linear(hx[0], w_hh)`` tiles, in the
    # input dtype, consumed by the combine kernel (which widens to the
    # accumulator dtype on load).  Holding them in the accumulator dtype is what
    # the generic op does, but on XPU the projection kernels hang the device
    # task when the rounded store widens the input dtype to a wider buffer (e.g.
    # fp16 -> fp32), so the tiles are kept in the input dtype instead; the
    # rounded values are identical.
    gi = torch.empty(
        (batch, 4 * hidden_size),
        dtype=input.dtype,
        device=input.device,
    )
    gh = torch.empty_like(gi)

    if input_size == 0:
        # An empty reduction contributes nothing (the bias is added by the
        # combine kernel), and a zero-trip loop is miscompiled on XPU, so write
        # the zeros directly instead of launching the projection kernel.
        gi.zero_()
    else:
        _lstm_cell_ih_kernel[grid](
            input,
            w_ih,
            gi,
            batch,
            input_size,
            hidden_size,
            input.stride(0),
            input.stride(1),
            w_ih.stride(0),
            w_ih.stride(1),
            gi.stride(0),
            gi.stride(1),
            ACC_DTYPE=_ACC_DTYPE[input.dtype],
            ROUND_DTYPE=_ROUND_DTYPE[input.dtype],
            BLOCK_B=block_b,
            BLOCK_H=block_h,
            BLOCK_K=block_k,
            num_warps=4,
            num_stages=3,
        )
    _lstm_cell_hh_kernel[grid](
        h_prev,
        w_hh,
        gh,
        batch,
        hidden_size,
        h_prev.stride(0),
        h_prev.stride(1),
        w_hh.stride(0),
        w_hh.stride(1),
        gh.stride(0),
        gh.stride(1),
        ACC_DTYPE=_ACC_DTYPE[input.dtype],
        ROUND_DTYPE=_ROUND_DTYPE[input.dtype],
        BLOCK_B=block_b,
        BLOCK_H=block_h,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=3,
    )

    combine_block_b = 1
    combine_block_h = _combine_block_h(hidden_size)
    combine_grid = (
        triton.cdiv(batch, combine_block_b),
        triton.cdiv(hidden_size, combine_block_h),
    )
    _lstm_cell_combine_kernel_xpu[combine_grid](
        gi,
        gh,
        c_prev,
        b_ih if b_ih is not None else input,
        b_hh if b_hh is not None else input,
        hy,
        cy,
        batch,
        hidden_size,
        gi.stride(0),
        gi.stride(1),
        c_prev.stride(0),
        c_prev.stride(1),
        hy.stride(0),
        hy.stride(1),
        HAS_BIH=b_ih is not None,
        HAS_BHH=b_hh is not None,
        ACC_DTYPE=_ACC_DTYPE[input.dtype],
        NEED_MASK=(batch % combine_block_b != 0)
        or (hidden_size % combine_block_h != 0),
        BLOCK_B=combine_block_b,
        BLOCK_H=combine_block_h,
        num_warps=4,
        num_stages=1,
    )
    return hy, cy

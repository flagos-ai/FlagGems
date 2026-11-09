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

"""Triton implementation of the ATen ``lstm.input`` operator.

Layout of the computation (PyTorch semantics):

  for layer l in 0..L-1:
      for direction d in 0..ndir-1:
          z_t = x_t @ W_ih[l,d]^T + b_ih + h_{t-1} @ W_hh[l,d]^T + b_hh
          i, f, g, o = sigmoid(z_i), sigmoid(z_f), tanh(z_g), sigmoid(z_o)
          c_t = f * c_{t-1} + i * g
          h_t = o * tanh(c_t)
          output[t, d*H:(d+1)*H] = h_t

One Triton program owns a single (batch row, direction) pair and keeps the whole
hidden/cell vector in registers across the sequence; the rectangular weight
matrices for the four gates of that direction are loaded once into registers
before the time loop.  The intermediate layer buffer and the final output are
both laid out as (T, B, H*ndir); a zero-copy permute produces the batch-first
result.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _sigmoid(x):
    return 1.0 / (1.0 + tl.exp(-x))


@triton.jit
def _tanh(x):
    # tanh(x) = 1 - 2 / (exp(2x) + 1); saturates exactly to +-1.
    return 1.0 - 2.0 / (tl.exp(2.0 * x) + 1.0)


@triton.jit
def _lstm_layer_kernel(
    x_ptr,
    out_ptr,
    hy_ptr,
    cy_ptr,
    h0_ptr,
    c0_ptr,
    wih_ptr,
    whh_ptr,
    bih_ptr,
    bhh_ptr,
    T,
    B,
    x_st,
    x_sb,
    x_sk,
    wih_s0,
    wih_s1,
    whh_s0,
    whh_s1,
    out_st,
    out_sb,
    out_col,
    hy_s0,
    hy_s1,
    cy_s0,
    cy_s1,
    h0_s0,
    h0_s1,
    c0_s0,
    c0_s1,
    state_idx,
    t_start,
    t_step,
    K: tl.constexpr,
    H: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)

    j = tl.arange(0, BLOCK_H)
    jm = j < H
    ki = tl.arange(0, BLOCK_K)
    kim = ki < K

    r0 = j
    r1 = H + j
    r2 = 2 * H + j
    r3 = 3 * H + j

    m_ih = jm[:, None] & kim[None, :]
    m_hh = jm[:, None] & jm[None, :]

    # ---- gate weight tiles (hoisted out of the time loop) ----
    wi = tl.load(
        wih_ptr + r0[:, None] * wih_s0 + ki[None, :] * wih_s1, mask=m_ih, other=0.0
    ).to(tl.float32)
    wf = tl.load(
        wih_ptr + r1[:, None] * wih_s0 + ki[None, :] * wih_s1, mask=m_ih, other=0.0
    ).to(tl.float32)
    wg = tl.load(
        wih_ptr + r2[:, None] * wih_s0 + ki[None, :] * wih_s1, mask=m_ih, other=0.0
    ).to(tl.float32)
    wo = tl.load(
        wih_ptr + r3[:, None] * wih_s0 + ki[None, :] * wih_s1, mask=m_ih, other=0.0
    ).to(tl.float32)

    ui = tl.load(
        whh_ptr + r0[:, None] * whh_s0 + j[None, :] * whh_s1, mask=m_hh, other=0.0
    ).to(tl.float32)
    uf = tl.load(
        whh_ptr + r1[:, None] * whh_s0 + j[None, :] * whh_s1, mask=m_hh, other=0.0
    ).to(tl.float32)
    ug = tl.load(
        whh_ptr + r2[:, None] * whh_s0 + j[None, :] * whh_s1, mask=m_hh, other=0.0
    ).to(tl.float32)
    uo = tl.load(
        whh_ptr + r3[:, None] * whh_s0 + j[None, :] * whh_s1, mask=m_hh, other=0.0
    ).to(tl.float32)

    if HAS_BIAS:
        b0 = (
            tl.load(bih_ptr + r0, mask=jm, other=0.0)
            + tl.load(bhh_ptr + r0, mask=jm, other=0.0)
        ).to(tl.float32)
        b1 = (
            tl.load(bih_ptr + r1, mask=jm, other=0.0)
            + tl.load(bhh_ptr + r1, mask=jm, other=0.0)
        ).to(tl.float32)
        b2 = (
            tl.load(bih_ptr + r2, mask=jm, other=0.0)
            + tl.load(bhh_ptr + r2, mask=jm, other=0.0)
        ).to(tl.float32)
        b3 = (
            tl.load(bih_ptr + r3, mask=jm, other=0.0)
            + tl.load(bhh_ptr + r3, mask=jm, other=0.0)
        ).to(tl.float32)
    else:
        b0 = tl.zeros([BLOCK_H], tl.float32)
        b1 = tl.zeros([BLOCK_H], tl.float32)
        b2 = tl.zeros([BLOCK_H], tl.float32)
        b3 = tl.zeros([BLOCK_H], tl.float32)

    h = tl.load(h0_ptr + state_idx * h0_s0 + b * h0_s1 + j, mask=jm, other=0.0).to(
        tl.float32
    )
    c = tl.load(c0_ptr + state_idx * c0_s0 + b * c0_s1 + j, mask=jm, other=0.0).to(
        tl.float32
    )

    x_base = x_ptr + b * x_sb
    o_base = out_ptr + b * out_sb + out_col + j

    for it in range(0, T):
        t = t_start + it * t_step
        xt = tl.load(x_base + t * x_st + ki * x_sk, mask=kim, other=0.0).to(tl.float32)

        zi = tl.sum(wi * xt[None, :], axis=1) + tl.sum(ui * h[None, :], axis=1) + b0
        zf = tl.sum(wf * xt[None, :], axis=1) + tl.sum(uf * h[None, :], axis=1) + b1
        zg = tl.sum(wg * xt[None, :], axis=1) + tl.sum(ug * h[None, :], axis=1) + b2
        zo = tl.sum(wo * xt[None, :], axis=1) + tl.sum(uo * h[None, :], axis=1) + b3

        i_g = _sigmoid(zi)
        f_g = _sigmoid(zf)
        g_g = _tanh(zg)
        o_g = _sigmoid(zo)

        c = f_g * c + i_g * g_g
        h = o_g * _tanh(c)

        tl.store(o_base + t * out_st, h.to(out_ptr.dtype.element_ty), mask=jm)

    tl.store(
        hy_ptr + state_idx * hy_s0 + b * hy_s1 + j,
        h.to(hy_ptr.dtype.element_ty),
        mask=jm,
    )
    tl.store(
        cy_ptr + state_idx * cy_s0 + b * cy_s1 + j,
        c.to(cy_ptr.dtype.element_ty),
        mask=jm,
    )


def _next_pow2(n):
    return 1 << (int(n) - 1).bit_length() if int(n) > 1 else 1


def _num_warps(block_h):
    if block_h >= 16:
        return 4
    if block_h >= 8:
        return 2
    return 1


def lstm(
    input,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first,
):
    logger.debug("GEMS_HYGON LSTM")
    ndir = 2 if bidirectional else 1
    num_layers = int(num_layers)

    unbatched = input.dim() == 2
    if unbatched:
        x = input.unsqueeze(1)
        batch_first = False
    else:
        x = input

    if batch_first:
        T, B = x.size(1), x.size(0)
        x_st, x_sb = x.stride(1), x.stride(0)
    else:
        T, B = x.size(0), x.size(1)
        x_st, x_sb = x.stride(0), x.stride(1)
    x_sk = x.stride(2)

    if hx is None:
        hidden = int(params[1].size(1))
        h0 = torch.zeros((num_layers * ndir, B, hidden), dtype=x.dtype, device=x.device)
        c0 = torch.zeros_like(h0)
        h0_st = (B * hidden, hidden)
        c0_st = h0_st
    else:
        h0 = hx[0]
        c0 = hx[1]
        if unbatched:
            h0 = h0.unsqueeze(1)
            c0 = c0.unsqueeze(1)
        h0_st = (h0.stride(0), h0.stride(1))
        c0_st = (c0.stride(0), c0.stride(1))

    hidden = h0.size(-1)
    HN = hidden * ndir
    dev = x.device
    dtype = x.dtype

    out = torch.empty((T, B, HN), dtype=dtype, device=dev)
    hy = torch.empty((num_layers * ndir, B, hidden), dtype=dtype, device=dev)
    cy = torch.empty((num_layers * ndir, B, hidden), dtype=dtype, device=dev)

    mid_a = mid_b = None
    if num_layers > 1:
        mid_a = torch.empty((T, B, HN), dtype=torch.float32, device=dev)
        mid_b = torch.empty((T, B, HN), dtype=torch.float32, device=dev)

    hy_st = (hy.stride(0), hy.stride(1))
    cy_st = (cy.stride(0), cy.stride(1))
    out_st = B * HN

    block_h = _next_pow2(hidden)
    nw = _num_warps(block_h)

    idx_stride = 4 if has_biases else 2

    prev = x
    p_st, p_sb, p_sk = x_st, x_sb, x_sk

    for layer in range(num_layers):
        last = layer == num_layers - 1
        if last:
            dst = out
        else:
            dst = mid_a if layer % 2 == 0 else mid_b

        K = x.size(2) if layer == 0 else HN
        block_k = _next_pow2(K)

        for d in range(ndir):
            pidx = (layer * ndir + d) * idx_stride
            w_ih = params[pidx]
            w_hh = params[pidx + 1]
            if has_biases:
                b_ih = params[pidx + 2]
                b_hh = params[pidx + 3]
            else:
                b_ih = w_ih
                b_hh = w_hh

            t_start = T - 1 if d == 1 else 0
            t_step = -1 if d == 1 else 1

            _lstm_layer_kernel[(B,)](
                prev,
                dst,
                hy,
                cy,
                h0,
                c0,
                w_ih,
                w_hh,
                b_ih,
                b_hh,
                T,
                B,
                p_st,
                p_sb,
                p_sk,
                w_ih.stride(0),
                w_ih.stride(1),
                w_hh.stride(0),
                w_hh.stride(1),
                out_st,
                HN,
                d * hidden,
                hy_st[0],
                hy_st[1],
                cy_st[0],
                cy_st[1],
                h0_st[0],
                h0_st[1],
                c0_st[0],
                c0_st[1],
                layer * ndir + d,
                t_start,
                t_step,
                K=K,
                H=hidden,
                HAS_BIAS=bool(has_biases),
                BLOCK_H=block_h,
                BLOCK_K=block_k,
                num_warps=nw,
                num_stages=1,
            )

        prev = dst
        p_st, p_sb, p_sk = B * HN, HN, 1

    if unbatched:
        return out.squeeze(1), hy.squeeze(1), cy.squeeze(1)
    if batch_first:
        return out.permute(1, 0, 2), hy, cy
    return out, hy, cy

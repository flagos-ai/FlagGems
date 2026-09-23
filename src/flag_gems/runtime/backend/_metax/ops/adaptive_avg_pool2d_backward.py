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

logger = logging.getLogger(__name__)


def _adaptive_avg_pool2d_backward_rows(
    gptr,
    optr,
    TOTAL,
    s_g_nc,
    s_g_h,
    s_g_w,
    s_o_nc,
    s_o_h,
    s_o_w,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    acc_dtype: tl.constexpr,
    out_dtype: tl.constexpr,
    BLOCK_W: tl.constexpr,
    MAXR_H: tl.constexpr,
    MAXR_W: tl.constexpr,
    ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    widx = tl.program_id(1)
    f0 = pid * ROWS

    iw = widx * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_iw = iw < W_in

    # Covering output columns for each input column: [ow_lo, ow_hi)
    ow_lo = (iw * W_out) // W_in
    ow_hi = ((iw + 1) * W_out + W_in - 1) // W_in

    for r in tl.static_range(ROWS):
        f = f0 + r
        valid = f < TOTAL
        nc = f // H_in
        ih = f % H_in

        # Covering output rows for this input row: [oh_lo, oh_hi)
        oh_lo = (ih * H_out) // H_in
        oh_hi = ((ih + 1) * H_out + H_in - 1) // H_in

        gbase = gptr + nc * s_g_nc
        obase = optr + nc * s_o_nc + ih * s_o_h

        acc = tl.zeros([BLOCK_W], dtype=acc_dtype)

        for it_h in tl.static_range(MAXR_H):
            oh = oh_lo + it_h
            if oh < oh_hi:
                kh = ((oh + 1) * H_in + H_out - 1) // H_out - (oh * H_in) // H_out
                grow = gbase + oh * s_g_h
                for it_w in tl.static_range(MAXR_W):
                    ow = ow_lo + it_w
                    m = (ow < ow_hi) & mask_iw & valid
                    g = tl.load(grow + ow * s_g_w, mask=m, other=0.0)
                    kw = ((ow + 1) * W_in + W_out - 1) // W_out - (ow * W_in) // W_out
                    acc += g.to(acc_dtype) / (kh.to(acc_dtype) * kw.to(acc_dtype))

        tl.store(obase + iw * s_o_w, acc.to(out_dtype), mask=mask_iw & valid)


@triton.jit
def _adaptive_avg_pool2d_backward_tile(
    gptr,
    optr,
    TOTAL,
    s_g_nc,
    s_g_h,
    s_g_w,
    s_o_nc,
    s_o_h,
    s_o_w,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    acc_dtype: tl.constexpr,
    out_dtype: tl.constexpr,
    BLOCK_W: tl.constexpr,
    MAXR_H: tl.constexpr,
    MAXR_W: tl.constexpr,
    ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    widx = tl.program_id(1)

    # Flat row ids covered by this program: [pid*ROWS, pid*ROWS + ROWS)
    f = pid * ROWS + tl.arange(0, ROWS)
    mask_r = f < TOTAL
    nc = f // H_in
    ih = f % H_in

    # Covering output rows for each input row: [oh_lo, oh_hi)
    oh_lo = (ih * H_out) // H_in
    oh_hi = ((ih + 1) * H_out + H_in - 1) // H_in

    iw = widx * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_iw = iw < W_in

    # Covering output columns for each input column: [ow_lo, ow_hi)
    ow_lo = (iw * W_out) // W_in
    ow_hi = ((iw + 1) * W_out + W_in - 1) // W_in

    gbase = gptr + nc * s_g_nc
    obase = optr + nc * s_o_nc + ih * s_o_h

    acc = tl.zeros([ROWS, BLOCK_W], dtype=acc_dtype)

    for it_h in tl.static_range(MAXR_H):
        oh = oh_lo + it_h
        m_h = (oh < oh_hi) & mask_r
        kh = ((oh + 1) * H_in + H_out - 1) // H_out - (oh * H_in) // H_out
        for it_w in tl.static_range(MAXR_W):
            ow = ow_lo + it_w
            m = m_h[:, None] & (ow < ow_hi)[None, :] & mask_iw[None, :]
            addr = (gbase + oh * s_g_h)[:, None] + ow[None, :] * s_g_w
            g = tl.load(addr, mask=m, other=0.0)
            kw = ((ow + 1) * W_in + W_out - 1) // W_out - (ow * W_in) // W_out
            acc += g.to(acc_dtype) / (
                kh.to(acc_dtype)[:, None] * kw.to(acc_dtype)[None, :]
            )

    tl.store(
        obase[:, None] + iw[None, :] * s_o_w,
        acc.to(out_dtype),
        mask=mask_r[:, None] & mask_iw[None, :],
    )


@triton.jit
def _adaptive_avg_pool2d_backward_exact(
    gptr,
    optr,
    TOTAL,
    s_g_nc,
    s_g_h,
    s_g_w,
    s_o_nc,
    s_o_h,
    s_o_w,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    acc_dtype: tl.constexpr,
    out_dtype: tl.constexpr,
    BLOCK_W: tl.constexpr,
    ROWS: tl.constexpr,
):
    # Exact-division downsampling specialization:
    #   out[ih, iw] = grad[ih // KH, iw // KW] / (KH * KW)
    # No covering loop, no window computations, one masked 2D gather + store.
    pid = tl.program_id(0)
    widx = tl.program_id(1)

    f = pid * ROWS + tl.arange(0, ROWS)
    mask_r = f < TOTAL
    nc = f // H_in
    ih = f % H_in

    iw = widx * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_iw = iw < W_in

    gbase = gptr + nc * s_g_nc
    grow = gbase + (ih // KH) * s_g_h
    addr = grow[:, None] + (iw // KW)[None, :] * s_g_w
    m = mask_r[:, None] & mask_iw[None, :]
    g = tl.load(addr, mask=m, other=0.0)

    inv = 1.0 / (KH * KW)
    acc = g.to(acc_dtype) * inv

    obase = optr + nc * s_o_nc + ih * s_o_h
    tl.store(obase[:, None] + iw[None, :] * s_o_w, acc.to(out_dtype), mask=m)


def _triton_dtype(dt):
    if dt is torch.float16:
        return tl.float16
    if dt is torch.bfloat16:
        return tl.bfloat16
    if dt is torch.float32:
        return tl.float32
    if dt is torch.float64:
        return tl.float64
    return dt


def _cover_max(input_size, output_size):
    """Exact max over i in [0, input_size) of ceil((i+1)*output_size/input_size) - floor(i*output_size/input_size)."""
    if input_size % output_size == 0:
        return 1
    if output_size <= input_size:
        # non-exact downsampling: window overlap (covering count 2) is guaranteed
        return 2
    # output_size > input_size (upsampling): exact via loop when input_size is small, else safe loose bound
    if input_size <= 4096:
        m = 1
        for i in range(input_size):
            d = ((i + 1) * output_size + input_size - 1) // input_size - (
                i * output_size
            ) // input_size
            if d > m:
                m = d
        return m
    return min(output_size, (output_size + input_size - 1) // input_size + 1)


def _adaptive_avg_pool2d_backward(grad_output, self):
    g = grad_output
    s = self

    orig_shape = s.shape
    if g.dim() == 3:
        g = g.unsqueeze(0)
    if s.dim() == 3:
        s = s.unsqueeze(0)

    N, C = g.shape[0], g.shape[1]
    H_out, W_out = g.shape[2], g.shape[3]
    H_in, W_in = s.shape[2], s.shape[3]

    out = torch.empty_like(s)

    NCH = N * C
    TOTAL = NCH * H_in
    if TOTAL * W_in == 0:
        return out.view(orig_shape)

    if s.dtype == torch.float64:
        acc_dtype = tl.float64
    else:
        acc_dtype = tl.float32

    MAXR_H = triton.next_power_of_2(_cover_max(H_in, H_out))
    MAXR_W = triton.next_power_of_2(_cover_max(W_in, W_out))

    BLOCK_W = min(max(64, triton.next_power_of_2(W_in)), 256)
    while BLOCK_W * MAXR_W > 1024 and BLOCK_W > 64:
        BLOCK_W //= 2

    ROWS = 8
    grid = (triton.cdiv(TOTAL, ROWS), triton.cdiv(W_in, BLOCK_W))

    if TOTAL >= 4096 and H_in % H_out == 0 and W_in % W_out == 0:
        ROWS = 16 if BLOCK_W * 16 <= 1024 else 8
        grid = (triton.cdiv(TOTAL, ROWS), triton.cdiv(W_in, BLOCK_W))
        _adaptive_avg_pool2d_backward_exact[grid](
            g,
            out,
            TOTAL,
            g.stride(1),
            g.stride(2),
            g.stride(3),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            H_in=H_in,
            W_in=W_in,
            KH=H_in // H_out,
            KW=W_in // W_out,
            acc_dtype=acc_dtype,
            out_dtype=_triton_dtype(out.dtype),
            BLOCK_W=BLOCK_W,
            ROWS=ROWS,
            num_warps=1,
        )
    elif TOTAL >= 4096:
        _adaptive_avg_pool2d_backward_tile[grid](
            g,
            out,
            TOTAL,
            g.stride(1),
            g.stride(2),
            g.stride(3),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            H_in=H_in,
            W_in=W_in,
            H_out=H_out,
            W_out=W_out,
            acc_dtype=acc_dtype,
            out_dtype=_triton_dtype(out.dtype),
            BLOCK_W=BLOCK_W,
            MAXR_H=MAXR_H,
            MAXR_W=MAXR_W,
            ROWS=ROWS,
            num_warps=1,
        )
    else:
        ROWS = max(1, min(16, triton.next_power_of_2(TOTAL // 8192)))
        grid = (triton.cdiv(TOTAL, ROWS), triton.cdiv(W_in, BLOCK_W))
        _adaptive_avg_pool2d_backward_rows[grid](
            g,
            out,
            TOTAL,
            g.stride(1),
            g.stride(2),
            g.stride(3),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            H_in=H_in,
            W_in=W_in,
            H_out=H_out,
            W_out=W_out,
            acc_dtype=acc_dtype,
            out_dtype=_triton_dtype(out.dtype),
            BLOCK_W=BLOCK_W,
            MAXR_H=MAXR_H,
            MAXR_W=MAXR_W,
            ROWS=ROWS,
            num_warps=1,
        )

    if len(orig_shape) == 3:
        return out.view(orig_shape)
    return out

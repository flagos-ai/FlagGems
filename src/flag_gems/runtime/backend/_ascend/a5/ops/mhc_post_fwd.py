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
def _row_piece(
    xrow,
    rrow,
    orow,
    OFF: tl.constexpr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MASKED: tl.constexpr,
    p0,
    p1,
    p2,
    p3,
    c00,
    c01,
    c02,
    c03,
    c10,
    c11,
    c12,
    c13,
    c20,
    c21,
    c22,
    c23,
    c30,
    c31,
    c32,
    c33,
):
    col = OFF + tl.arange(0, BLOCK_H)
    if MASKED:
        mask = col < H
    else:
        mask = None

    # issue all five row loads before consuming any of them
    xb = tl.load(xrow + col, mask=mask)
    b0 = tl.load(rrow + col + 0 * H, mask=mask)
    b1 = tl.load(rrow + col + 1 * H, mask=mask)
    b2 = tl.load(rrow + col + 2 * H, mask=mask)
    b3 = tl.load(rrow + col + 3 * H, mask=mask)

    xf = xb.to(tl.float32)
    a0 = p0 * xf
    a1 = p1 * xf
    a2 = p2 * xf
    a3 = p3 * xf

    v = b0.to(tl.float32)
    a0 += c00 * v
    a1 += c01 * v
    a2 += c02 * v
    a3 += c03 * v
    v = b1.to(tl.float32)
    a0 += c10 * v
    a1 += c11 * v
    a2 += c12 * v
    a3 += c13 * v
    v = b2.to(tl.float32)
    a0 += c20 * v
    a1 += c21 * v
    a2 += c22 * v
    a3 += c23 * v
    v = b3.to(tl.float32)
    a0 += c30 * v
    a1 += c31 * v
    a2 += c32 * v
    a3 += c33 * v

    tl.store(orow + col + 0 * H, a0.to(tl.bfloat16), mask=mask)
    tl.store(orow + col + 1 * H, a1.to(tl.bfloat16), mask=mask)
    tl.store(orow + col + 2 * H, a2.to(tl.bfloat16), mask=mask)
    tl.store(orow + col + 3 * H, a3.to(tl.bfloat16), mask=mask)


@triton.jit
def _load_coeffs(post_ptr, comb_ptr, tn):
    pbase = tn * 4
    p0 = tl.load(post_ptr + pbase + 0).to(tl.float32)
    p1 = tl.load(post_ptr + pbase + 1).to(tl.float32)
    p2 = tl.load(post_ptr + pbase + 2).to(tl.float32)
    p3 = tl.load(post_ptr + pbase + 3).to(tl.float32)

    cbase = tn * 16
    c00 = tl.load(comb_ptr + cbase + 0).to(tl.float32)
    c01 = tl.load(comb_ptr + cbase + 1).to(tl.float32)
    c02 = tl.load(comb_ptr + cbase + 2).to(tl.float32)
    c03 = tl.load(comb_ptr + cbase + 3).to(tl.float32)
    c10 = tl.load(comb_ptr + cbase + 4).to(tl.float32)
    c11 = tl.load(comb_ptr + cbase + 5).to(tl.float32)
    c12 = tl.load(comb_ptr + cbase + 6).to(tl.float32)
    c13 = tl.load(comb_ptr + cbase + 7).to(tl.float32)
    c20 = tl.load(comb_ptr + cbase + 8).to(tl.float32)
    c21 = tl.load(comb_ptr + cbase + 9).to(tl.float32)
    c22 = tl.load(comb_ptr + cbase + 10).to(tl.float32)
    c23 = tl.load(comb_ptr + cbase + 11).to(tl.float32)
    c30 = tl.load(comb_ptr + cbase + 12).to(tl.float32)
    c31 = tl.load(comb_ptr + cbase + 13).to(tl.float32)
    c32 = tl.load(comb_ptr + cbase + 14).to(tl.float32)
    c33 = tl.load(comb_ptr + cbase + 15).to(tl.float32)
    return (
        p0,
        p1,
        p2,
        p3,
        c00,
        c01,
        c02,
        c03,
        c10,
        c11,
        c12,
        c13,
        c20,
        c21,
        c22,
        c23,
        c30,
        c31,
        c32,
        c33,
    )


@triton.jit
def _mhc_post_fwd_pieces(
    x_ptr,
    res_ptr,
    post_ptr,
    comb_ptr,
    out_ptr,
    H: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
    B3: tl.constexpr,
):

    tn = tl.program_id(0)

    (
        p0,
        p1,
        p2,
        p3,
        c00,
        c01,
        c02,
        c03,
        c10,
        c11,
        c12,
        c13,
        c20,
        c21,
        c22,
        c23,
        c30,
        c31,
        c32,
        c33,
    ) = _load_coeffs(post_ptr, comb_ptr, tn)

    xrow = x_ptr + tn * H
    rrow = res_ptr + tn * (4 * H)
    orow = out_ptr + tn * (4 * H)

    _row_piece(
        xrow,
        rrow,
        orow,
        0,
        H,
        B0,
        False,
        p0,
        p1,
        p2,
        p3,
        c00,
        c01,
        c02,
        c03,
        c10,
        c11,
        c12,
        c13,
        c20,
        c21,
        c22,
        c23,
        c30,
        c31,
        c32,
        c33,
    )
    if B1 > 0:
        _row_piece(
            xrow,
            rrow,
            orow,
            B0,
            H,
            B1,
            False,
            p0,
            p1,
            p2,
            p3,
            c00,
            c01,
            c02,
            c03,
            c10,
            c11,
            c12,
            c13,
            c20,
            c21,
            c22,
            c23,
            c30,
            c31,
            c32,
            c33,
        )
        if B2 > 0:
            _row_piece(
                xrow,
                rrow,
                orow,
                B0 + B1,
                H,
                B2,
                True,
                p0,
                p1,
                p2,
                p3,
                c00,
                c01,
                c02,
                c03,
                c10,
                c11,
                c12,
                c13,
                c20,
                c21,
                c22,
                c23,
                c30,
                c31,
                c32,
                c33,
            )


@triton.jit
def _mhc_post_fwd_win(
    x_ptr,
    res_ptr,
    post_ptr,
    comb_ptr,
    out_ptr,
    H: tl.constexpr,
    W: tl.constexpr,
    NW: tl.constexpr,
):

    tn = tl.program_id(0)

    (
        p0,
        p1,
        p2,
        p3,
        c00,
        c01,
        c02,
        c03,
        c10,
        c11,
        c12,
        c13,
        c20,
        c21,
        c22,
        c23,
        c30,
        c31,
        c32,
        c33,
    ) = _load_coeffs(post_ptr, comb_ptr, tn)

    xrow = x_ptr + tn * H
    rrow = res_ptr + tn * (4 * H)
    orow = out_ptr + tn * (4 * H)

    for win in range(NW):
        off = tl.minimum(win * W, H - W)
        col = off + tl.arange(0, W)

        xb = tl.load(xrow + col)
        b0 = tl.load(rrow + col + 0 * H)
        b1 = tl.load(rrow + col + 1 * H)
        b2 = tl.load(rrow + col + 2 * H)
        b3 = tl.load(rrow + col + 3 * H)

        xf = xb.to(tl.float32)
        a0 = p0 * xf
        a1 = p1 * xf
        a2 = p2 * xf
        a3 = p3 * xf

        v = b0.to(tl.float32)
        a0 += c00 * v
        a1 += c01 * v
        a2 += c02 * v
        a3 += c03 * v
        v = b1.to(tl.float32)
        a0 += c10 * v
        a1 += c11 * v
        a2 += c12 * v
        a3 += c13 * v
        v = b2.to(tl.float32)
        a0 += c20 * v
        a1 += c21 * v
        a2 += c22 * v
        a3 += c23 * v
        v = b3.to(tl.float32)
        a0 += c30 * v
        a1 += c31 * v
        a2 += c32 * v
        a3 += c33 * v

        tl.store(orow + col + 0 * H, a0.to(tl.bfloat16))
        tl.store(orow + col + 1 * H, a1.to(tl.bfloat16))
        tl.store(orow + col + 2 * H, a2.to(tl.bfloat16))
        tl.store(orow + col + 3 * H, a3.to(tl.bfloat16))


@triton.jit
def _mhc_post_fwd_loop(
    x_ptr,
    res_ptr,
    post_ptr,
    comb_ptr,
    out_ptr,
    H: tl.constexpr,
    NHT: tl.constexpr,
    BLOCK_H: tl.constexpr,
):

    tn = tl.program_id(0)

    (
        p0,
        p1,
        p2,
        p3,
        c00,
        c01,
        c02,
        c03,
        c10,
        c11,
        c12,
        c13,
        c20,
        c21,
        c22,
        c23,
        c30,
        c31,
        c32,
        c33,
    ) = _load_coeffs(post_ptr, comb_ptr, tn)

    xrow = x_ptr + tn * H
    rrow = res_ptr + tn * (4 * H)
    orow = out_ptr + tn * (4 * H)

    for tile in range(NHT):
        col = tile * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = col < H

        xf = tl.load(xrow + col, mask=mask).to(tl.float32)
        a0 = p0 * xf
        a1 = p1 * xf
        a2 = p2 * xf
        a3 = p3 * xf

        v = tl.load(rrow + col + 0 * H, mask=mask).to(tl.float32)
        a0 += c00 * v
        a1 += c01 * v
        a2 += c02 * v
        a3 += c03 * v
        v = tl.load(rrow + col + 1 * H, mask=mask).to(tl.float32)
        a0 += c10 * v
        a1 += c11 * v
        a2 += c12 * v
        a3 += c13 * v
        v = tl.load(rrow + col + 2 * H, mask=mask).to(tl.float32)
        a0 += c20 * v
        a1 += c21 * v
        a2 += c22 * v
        a3 += c23 * v
        v = tl.load(rrow + col + 3 * H, mask=mask).to(tl.float32)
        a0 += c30 * v
        a1 += c31 * v
        a2 += c32 * v
        a3 += c33 * v

        tl.store(orow + col + 0 * H, a0.to(tl.bfloat16), mask=mask)
        tl.store(orow + col + 1 * H, a1.to(tl.bfloat16), mask=mask)
        tl.store(orow + col + 2 * H, a2.to(tl.bfloat16), mask=mask)
        tl.store(orow + col + 3 * H, a3.to(tl.bfloat16), mask=mask)


_MAX_PIECE = 4096
_MAX_PIECES = 4
_MAX_PIECES_INLINE = 2
_WIN_W = 4096


def _split_row(h):

    pieces = []
    rem = h
    while rem > 0:
        b = _MAX_PIECE
        while b > rem:
            b >>= 1
        if b < 128:
            return None
        pieces.append(b)
        rem -= b
        if len(pieces) > _MAX_PIECES:
            return None
    if not pieces:
        return None
    return pieces


def _use_pieces(x, residual, post_layer_mix, comb_res_mix, out, n, h, pieces):
    p = pieces + [0] * (_MAX_PIECES - len(pieces))
    _mhc_post_fwd_pieces[(n,)](
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        out,
        H=h,
        B0=p[0],
        B1=p[1],
        B2=p[2],
        B3=p[3],
    )


@triton.jit
def _mhc_post_fwd_persist(
    x_ptr,
    res_ptr,
    post_ptr,
    comb_ptr,
    out_ptr,
    N,
    NT,
    H: tl.constexpr,
    W: tl.constexpr,
):

    pid = tl.program_id(0)
    first = pid * NT
    col = tl.arange(0, W)

    for i in range(NT):
        tn = first + i
        tn = tl.minimum(tn, N - 1)

        (
            p0,
            p1,
            p2,
            p3,
            c00,
            c01,
            c02,
            c03,
            c10,
            c11,
            c12,
            c13,
            c20,
            c21,
            c22,
            c23,
            c30,
            c31,
            c32,
            c33,
        ) = _load_coeffs(post_ptr, comb_ptr, tn)

        xrow = x_ptr + tn * H
        rrow = res_ptr + tn * (4 * H)
        orow = out_ptr + tn * (4 * H)

        xb = tl.load(xrow + col)
        b0 = tl.load(rrow + col + 0 * H)
        b1 = tl.load(rrow + col + 1 * H)
        b2 = tl.load(rrow + col + 2 * H)
        b3 = tl.load(rrow + col + 3 * H)

        xf = xb.to(tl.float32)
        a0 = p0 * xf
        a1 = p1 * xf
        a2 = p2 * xf
        a3 = p3 * xf

        v = b0.to(tl.float32)
        a0 += c00 * v
        a1 += c01 * v
        a2 += c02 * v
        a3 += c03 * v
        v = b1.to(tl.float32)
        a0 += c10 * v
        a1 += c11 * v
        a2 += c12 * v
        a3 += c13 * v
        v = b2.to(tl.float32)
        a0 += c20 * v
        a1 += c21 * v
        a2 += c22 * v
        a3 += c23 * v
        v = b3.to(tl.float32)
        a0 += c30 * v
        a1 += c31 * v
        a2 += c32 * v
        a3 += c33 * v

        tl.store(orow + col + 0 * H, a0.to(tl.bfloat16))
        tl.store(orow + col + 1 * H, a1.to(tl.bfloat16))
        tl.store(orow + col + 2 * H, a2.to(tl.bfloat16))
        tl.store(orow + col + 3 * H, a3.to(tl.bfloat16))


_PERSIST_PROGRAMS = 64


def _use_persist(x, residual, post_layer_mix, comb_res_mix, out, n, h):
    npg = _PERSIST_PROGRAMS
    if n < npg:
        npg = n
    nt = (n + npg - 1) // npg
    _mhc_post_fwd_persist[(npg,)](
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        out,
        n,
        nt,
        H=h,
        W=h,
    )


def mhc_post_fwd(x, residual, post_layer_mix, comb_res_mix, out=None):
    """Apply four-stream mHC residual and post-layer mixing."""
    if (
        residual.shape != (*x.shape[:2], 4, x.shape[-1])
        or post_layer_mix.shape != (*x.shape[:2], 4, 1)
        or comb_res_mix.shape != (*x.shape[:2], 4, 4)
    ):
        raise ValueError("expected four mHC streams per token")
    if not all(
        t.is_contiguous() for t in (x, residual, post_layer_mix, comb_res_mix)
    ) or (
        out is not None
        and (
            out.shape != residual.shape
            or not out.is_contiguous()
            or out.dtype != residual.dtype
        )
    ):
        raise ValueError("expected contiguous inputs and a matching output")
    n0, n1, mhc, hidden = residual.shape
    if out is None:
        out = torch.empty_like(residual)

    n = n0 * n1
    h = hidden

    pieces = _split_row(h)
    nw = (h + _WIN_W - 1) // _WIN_W
    if pieces is not None and len(pieces) == 1 and pieces[0] == _WIN_W:
        # Row is exactly one 4096-wide window (h=4096): a single straight-line
        # body per token has no second iteration to overlap its prologue with,
        # so run a persistent loop over a chunk of tokens per program instead.
        _use_persist(x, residual, post_layer_mix, comb_res_mix, out, n, h)
        return out

    if pieces is not None and len(pieces) <= _MAX_PIECES_INLINE:
        # Rows that decompose into <=2 exact pieces (4096; 5120 = 4096+1024;
        # 2560 = 2048+512; 1280 = 1024+256) keep the straight-line path: the
        # measured window cover over them would spend 60% redundant lanes
        # (5120) and lose more than the loop gains.
        _use_pieces(x, residual, post_layer_mix, comb_res_mix, out, n, h, pieces)
        return out

    if pieces is not None and nw * _WIN_W * 4 <= h * 5:
        # Rows wider than one window whose cover costs at most 25% redundant
        # lanes (7168 -> 2 windows, 14.3%): equal 4096-wide shifted windows in
        # one rotating loop remove the inter-piece fence at a bounded price.
        _mhc_post_fwd_win[(n,)](
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            out,
            H=h,
            W=_WIN_W,
            NW=nw,
        )
        return out

    if pieces is not None:
        _use_pieces(x, residual, post_layer_mix, comb_res_mix, out, n, h, pieces)
        return out

    block_h = _MAX_PIECE
    nht = (h + block_h - 1) // block_h
    _mhc_post_fwd_loop[(n,)](
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        out,
        H=h,
        NHT=nht,
        BLOCK_H=block_h,
    )
    return out

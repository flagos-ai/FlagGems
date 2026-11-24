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


@triton.jit
def _conv_depthwise2d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    NCOUT,
    H,
    W,
    OH,
    OW,
    sx0,
    sx1,
    sx2,
    sx3,
    sw0,
    sw2,
    sw3,
    so0,
    so1,
    so2,
    so3,
    SH,
    SW,
    PH,
    PW,
    DH,
    DW,
    MULT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Small-plane path: a 2-D output tile of one (n, oc) plane.
    pid_w = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_nc = tl.program_id(2)

    oc = pid_nc % NCOUT
    n = pid_nc // NCOUT
    ic = oc // MULT

    x_base = x_ptr + n * sx0 + ic * sx1
    o_base = out_ptr + n * so0 + oc * so1
    w_base = w_ptr + oc * sw0

    oh = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    ow = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    o_mask = (oh[:, None] < OH) & (ow[None, :] < OW)

    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    for kh in tl.static_range(KH):
        ih = oh * SH - PH + kh * DH
        mh = (ih >= 0) & (ih < H)
        for kw in tl.static_range(KW):
            iw = ow * SW - PW + kw * DW
            mw = (iw >= 0) & (iw < W)
            m = mh[:, None] & mw[None, :] & o_mask
            x = tl.load(
                x_base + ih[:, None] * sx2 + iw[None, :] * sx3,
                mask=m,
                other=0.0,
            )
            wv = tl.load(w_base + kh * sw2 + kw * sw3)
            acc += x.to(tl.float32) * wv.to(tl.float32)

    if HAS_BIAS:
        acc = acc + tl.load(b_ptr + oc).to(tl.float32)

    tl.store(
        o_base + oh[:, None] * so2 + ow[None, :] * so3,
        acc.to(out_ptr.dtype.element_ty),
        mask=o_mask,
    )


@triton.jit
def _conv_depthwise2d_rows_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    NCOUT,
    H,
    W,
    OH,
    OW,
    sx0,
    sx1,
    sx2,
    sx3,
    sw0,
    sw2,
    sw3,
    so0,
    so1,
    so2,
    so3,
    PH,
    PW,
    SW,
    DW,
    MULT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SH: tl.constexpr,
    DH: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Row-reuse path for large planes.  Output row oh and tap kh read input row
    # oh*SH - PH + kh*DH, so ROWS consecutive output rows share
    # (ROWS-1)*SH + (KH-1)*DH + 1 input rows instead of reading KH of their own.
    # Each shared input row is loaded once per kw as a 1-D vector (one output
    # element per lane, the coalesced mapping) and its contribution is folded
    # into every output row r that uses it with the compile-time weight
    # index kh = (j - r*SH) / DH.
    pid_w = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_nc = tl.program_id(2)

    oc = pid_nc % NCOUT
    n = pid_nc // NCOUT
    ic = oc // MULT

    x_base = x_ptr + n * sx0 + ic * sx1
    o_base = out_ptr + n * so0 + oc * so1
    w_base = w_ptr + oc * sw0

    ow = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    col = ow < OW
    oh0 = pid_h * ROWS
    ih_base = oh0 * SH - PH

    acc0 = tl.zeros((BLOCK_W,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_W,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_W,), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_W,), dtype=tl.float32)
    acc4 = tl.zeros((BLOCK_W,), dtype=tl.float32)
    acc5 = tl.zeros((BLOCK_W,), dtype=tl.float32)
    acc6 = tl.zeros((BLOCK_W,), dtype=tl.float32)
    acc7 = tl.zeros((BLOCK_W,), dtype=tl.float32)

    for j in tl.static_range((ROWS - 1) * SH + (KH - 1) * DH + 1):
        ih = ih_base + j
        row_ok = (ih >= 0) & (ih < H)
        for kw in tl.static_range(KW):
            iw = ow * SW - PW + kw * DW
            m = row_ok & col & (iw >= 0) & (iw < W)
            xv = tl.load(x_base + ih * sx2 + iw * sx3, mask=m, other=0.0).to(tl.float32)
            for r in tl.static_range(ROWS):
                off = j - r * SH
                if off >= 0:
                    if off % DH == 0:
                        kh = off // DH
                        if kh < KH:
                            wv = tl.load(w_base + kh * sw2 + kw * sw3).to(tl.float32)
                            c = xv * wv
                            if r == 0:
                                acc0 += c
                            if r == 1:
                                acc1 += c
                            if r == 2:
                                acc2 += c
                            if r == 3:
                                acc3 += c
                            if r == 4:
                                acc4 += c
                            if r == 5:
                                acc5 += c
                            if r == 6:
                                acc6 += c
                            if r == 7:
                                acc7 += c

    if HAS_BIAS:
        bv = tl.load(b_ptr + oc).to(tl.float32)
        acc0 = acc0 + bv
        acc1 = acc1 + bv
        acc2 = acc2 + bv
        acc3 = acc3 + bv
        acc4 = acc4 + bv
        acc5 = acc5 + bv
        acc6 = acc6 + bv
        acc7 = acc7 + bv

    for r in tl.static_range(ROWS):
        oh = oh0 + r
        sm = col & (oh < OH)
        addr = o_base + oh * so2 + ow * so3
        if r == 0:
            tl.store(addr, acc0.to(out_ptr.dtype.element_ty), mask=sm)
        if r == 1:
            tl.store(addr, acc1.to(out_ptr.dtype.element_ty), mask=sm)
        if r == 2:
            tl.store(addr, acc2.to(out_ptr.dtype.element_ty), mask=sm)
        if r == 3:
            tl.store(addr, acc3.to(out_ptr.dtype.element_ty), mask=sm)
        if r == 4:
            tl.store(addr, acc4.to(out_ptr.dtype.element_ty), mask=sm)
        if r == 5:
            tl.store(addr, acc5.to(out_ptr.dtype.element_ty), mask=sm)
        if r == 6:
            tl.store(addr, acc6.to(out_ptr.dtype.element_ty), mask=sm)
        if r == 7:
            tl.store(addr, acc7.to(out_ptr.dtype.element_ty), mask=sm)


# Row-reuse search space for large planes: rows per program x tile width.
# num_warps = BLOCK_W / 64 keeps one output element per lane; the extra
# (BLOCK_W=128, num_warps=1) and (BLOCK_W=256, num_warps=2) entries give two
# elements per lane, which the 16-bit dtypes profited from in earlier rounds.
# ROWS=1 reproduces the plain one-row-per-program mapping as a fallback.
_ROWS_CONFIGS = [
    triton.Config({"ROWS": 1, "BLOCK_W": 128}, num_warps=2, num_stages=1),
    triton.Config({"ROWS": 2, "BLOCK_W": 64}, num_warps=1, num_stages=1),
    triton.Config({"ROWS": 2, "BLOCK_W": 128}, num_warps=2, num_stages=1),
    triton.Config({"ROWS": 2, "BLOCK_W": 128}, num_warps=1, num_stages=1),
    triton.Config({"ROWS": 2, "BLOCK_W": 256}, num_warps=4, num_stages=1),
    triton.Config({"ROWS": 4, "BLOCK_W": 64}, num_warps=1, num_stages=1),
    triton.Config({"ROWS": 4, "BLOCK_W": 128}, num_warps=2, num_stages=1),
    triton.Config({"ROWS": 4, "BLOCK_W": 256}, num_warps=4, num_stages=1),
    triton.Config({"ROWS": 4, "BLOCK_W": 256}, num_warps=2, num_stages=1),
    triton.Config({"ROWS": 8, "BLOCK_W": 64}, num_warps=1, num_stages=1),
    triton.Config({"ROWS": 8, "BLOCK_W": 128}, num_warps=2, num_stages=1),
]

_conv_depthwise2d_rows_kernel_auto = triton.autotune(
    configs=_ROWS_CONFIGS,
    key=["H", "W", "OH", "OW", "KH", "KW", "MULT", "HAS_BIAS"],
)(_conv_depthwise2d_rows_kernel)


def _pair(v):
    if isinstance(v, (list, tuple)):
        return int(v[0]), int(v[1])
    return int(v), int(v)


def _conv_depthwise2d(self, weight, kernel_size, bias, stride, padding, dilation):
    logger.debug("GEMS_HYGON _CONV_DEPTHWISE2D")
    x = self

    if x.dim() != 4:
        raise RuntimeError(
            "_conv_depthwise2d: expected 4D input tensor, got " + str(x.dim()) + "D"
        )
    if weight.dim() != 4:
        raise RuntimeError(
            "_conv_depthwise2d: expected 4D weight tensor, got "
            + str(weight.dim())
            + "D"
        )
    if weight.size(1) != 1:
        raise RuntimeError(
            "_conv_depthwise2d: expected weight.size(1) == 1, got "
            + str(weight.size(1))
        )

    ks = _pair(kernel_size)
    if ks[0] != int(weight.size(2)) or ks[1] != int(weight.size(3)):
        raise RuntimeError(
            "_conv_depthwise2d: kernel_size "
            + str([ks[0], ks[1]])
            + " does not match weight spatial size "
            + str([int(weight.size(2)), int(weight.size(3))])
        )

    KH = ks[0]
    KW = ks[1]
    SH, SW = _pair(stride)
    PH, PW = _pair(padding)
    DH, DW = _pair(dilation)

    N, C, H, W = int(x.size(0)), int(x.size(1)), int(x.size(2)), int(x.size(3))
    NCOUT = int(weight.size(0))

    if NCOUT % C != 0:
        raise RuntimeError(
            "_conv_depthwise2d: weight out channels "
            + str(NCOUT)
            + " not divisible by input channels "
            + str(C)
        )
    MULT = NCOUT // C

    if bias is not None and int(bias.numel()) != NCOUT:
        raise RuntimeError(
            "_conv_depthwise2d: bias length "
            + str(int(bias.numel()))
            + " does not match out channels "
            + str(NCOUT)
        )

    OH = (H + 2 * PH - DH * (KH - 1) - 1) // SH + 1
    OW = (W + 2 * PW - DW * (KW - 1) - 1) // SW + 1

    out = torch.empty((N, NCOUT, OH, OW), dtype=x.dtype, device=x.device)
    if out.numel() == 0:
        return out

    sx0, sx1, sx2, sx3 = (int(s) for s in x.stride())
    sw0, _, sw2, sw3 = (int(s) for s in weight.stride())
    so0, so1, so2, so3 = (int(s) for s in out.stride())

    if OH * OW <= 1024:
        bw = min(128, triton.next_power_of_2(max(OW, 1)))
        bh = min(8, triton.next_power_of_2(max(OH, 1)))
        while bh * bw > 256 and bw > 1:
            if bw > bh:
                bw //= 2
            else:
                bh //= 2
        num_warps = max(1, min(4, (bh * bw) // 64))
        grid = (
            triton.cdiv(OW, bw),
            triton.cdiv(OH, bh),
            N * NCOUT,
        )
        _conv_depthwise2d_kernel[grid](
            x,
            weight,
            bias if bias is not None else x,
            out,
            NCOUT,
            H,
            W,
            OH,
            OW,
            sx0,
            sx1,
            sx2,
            sx3,
            sw0,
            sw2,
            sw3,
            so0,
            so1,
            so2,
            so3,
            SH,
            SW,
            PH,
            PW,
            DH,
            DW,
            MULT=MULT,
            HAS_BIAS=bias is not None,
            KH=KH,
            KW=KW,
            BLOCK_H=bh,
            BLOCK_W=bw,
            num_warps=num_warps,
            num_stages=1,
        )
        return out

    grid = lambda meta: (
        triton.cdiv(OW, meta["BLOCK_W"]),
        triton.cdiv(OH, meta["ROWS"]),
        N * NCOUT,
    )
    _conv_depthwise2d_rows_kernel_auto[grid](
        x,
        weight,
        bias if bias is not None else x,
        out,
        NCOUT,
        H,
        W,
        OH,
        OW,
        sx0,
        sx1,
        sx2,
        sx3,
        sw0,
        sw2,
        sw3,
        so0,
        so1,
        so2,
        so3,
        PH,
        PW,
        SW,
        DW,
        MULT=MULT,
        HAS_BIAS=bias is not None,
        KH=KH,
        KW=KW,
        SH=SH,
        DH=DH,
    )
    return out

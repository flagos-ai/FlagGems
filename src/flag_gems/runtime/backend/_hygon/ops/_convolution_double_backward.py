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

"""Triton implementation of aten::_convolution_double_backward.

All three outputs are produced by device kernels:

  grad_ggO    = A(ggI, weight) [+ A(self, ggW)] [+ ggb]      (gather kernel, map = FWD)
  grad_self   = A^T_{ggW}(gO)                                (gather kernel, map = opposite)
  grad_weight = A^T_{ggI}(gO)                                (reduction kernel)

The gather kernel owns a (output-channel block, spatial block) tile of one
(n, group) pair and accumulates fp32 via tl.dot with ieee precision.  The
weight kernel owns a (d0, d1) tile of one kernel tap and reduces over
(n, y, x) of gO.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _dot_mm(a, b, IS_F16: tl.constexpr):
    """Matrix multiply keeping the operands in their native dtype.

    fp16/bf16 operands feed the matrix units directly with fp32 accumulation;
    fp32 operands use the ieee path.  Products are exact in both cases, so the
    result accuracy class is unchanged (fp32 accumulator throughout).
    """
    if IS_F16:
        return tl.dot(a, b, out_dtype=tl.float32)
    return tl.dot(a.to(tl.float32), b.to(tl.float32), input_precision="ieee")


@triton.jit
def _db_gather_kernel(
    out_ptr,
    A1_ptr,
    B1_ptr,
    A2_ptr,
    B2_ptr,
    bias_ptr,
    OCTOT,
    OC,
    S,
    Wq,
    Hb,
    Wb,
    R,
    sh,
    sw,
    ph,
    pw,
    dh,
    dw,
    gf1,
    sa1_0,
    sa1_1,
    sa1_h,
    sa1_w,
    gf2,
    sa2_0,
    sa2_1,
    sa2_h,
    sa2_w,
    sb1_n,
    sb1_c,
    sb1_h,
    sb1_w,
    sb2_n,
    sb2_c,
    sb2_h,
    sb2_w,
    sbias,
    ocg,
    cbg,
    G,
    KHR,
    KWR,
    HAS_A1: tl.constexpr,
    HAS_A2: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    MAP_FWD: tl.constexpr,
    S1: tl.constexpr,
    IS_F16: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BOC: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
):
    pid0 = tl.program_id(0)
    n = pid0 // G
    g = pid0 - n * G

    oc = tl.program_id(1) * BOC + tl.arange(0, BOC)
    moc = oc < OC
    s = tl.program_id(2) * BS + tl.arange(0, BS)
    ms = s < S
    y = s // Wq
    x = s - y * Wq

    acc = tl.zeros((BOC, BS), dtype=tl.float32)
    a1b = g * gf1 + oc[:, None] * sa1_0
    a2b = g * gf2 + oc[:, None] * sa2_0

    for kh in tl.range(0, KHR):
        for kw in tl.range(0, KWR):
            if MAP_FWD:
                ih = y * sh - ph + kh * dh
                iw = x * sw - pw + kw * dw
                m = ms
            else:
                nh = y + ph - kh * dh
                nw = x + pw - kw * dw
                if S1:
                    ih = nh
                    iw = nw
                    m = ms
                else:
                    m = ms & (nh % sh == 0) & (nw % sw == 0)
                    ih = nh // sh
                    iw = nw // sw
            m = m & (ih >= 0) & (ih < Hb) & (iw >= 0) & (iw < Wb)
            off1 = (n * sb1_n + ih * sb1_h + iw * sb1_w)[None, :]
            off2 = (n * sb2_n + ih * sb2_h + iw * sb2_w)[None, :]
            ka1 = a1b + kh * sa1_h + kw * sa1_w
            ka2 = a2b + kh * sa2_h + kw * sa2_w
            for k0 in tl.range(0, R, BK):
                rk = k0 + tl.arange(0, BK)
                mk = rk < R
                if HAS_A1:
                    a1 = tl.load(
                        A1_ptr + ka1 + rk[None, :] * sa1_1,
                        mask=moc[:, None] & mk[None, :],
                        other=0.0,
                    )
                    b1 = tl.load(
                        B1_ptr + off1 + (g * cbg + rk)[:, None] * sb1_c,
                        mask=mk[:, None] & m[None, :],
                        other=0.0,
                    )
                    acc += _dot_mm(a1, b1, IS_F16)
                if HAS_A2:
                    a2 = tl.load(
                        A2_ptr + ka2 + rk[None, :] * sa2_1,
                        mask=moc[:, None] & mk[None, :],
                        other=0.0,
                    )
                    b2 = tl.load(
                        B2_ptr + off2 + (g * cbg + rk)[:, None] * sb2_c,
                        mask=mk[:, None] & m[None, :],
                        other=0.0,
                    )
                    acc += _dot_mm(a2, b2, IS_F16)

    if HAS_BIAS:
        bv = tl.load(bias_ptr + (g * ocg + oc) * sbias, mask=moc, other=0.0)
        acc += bv[:, None].to(tl.float32)

    tl.store(
        out_ptr + (n * OCTOT + g * ocg + oc)[:, None] * S + s[None, :],
        acc,
        mask=moc[:, None] & ms[None, :],
    )


@triton.jit
def _db_zero_kernel(ptr, NEL, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(ptr + off, tl.zeros((BLOCK,), dtype=tl.float32), mask=off < NEL)


@triton.jit
def _db_dw_kernel(
    part_ptr,
    T0_ptr,
    T1_ptr,
    N,
    S,
    Wq,
    Hb,
    Wb,
    D0,
    D1,
    NBLK,
    SPLITN,
    NEL,
    sh,
    sw,
    ph,
    pw,
    dh,
    dw,
    st0_n,
    st0_c,
    st0_h,
    st0_w,
    st1_n,
    st1_c,
    st1_h,
    st1_w,
    G,
    MAP_FWD: tl.constexpr,
    ROW_MAPPED: tl.constexpr,
    S1: tl.constexpr,
    IS_F16: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BD0: tl.constexpr,
    BD1: tl.constexpr,
    BR: tl.constexpr,
):
    pid = tl.program_id(0)
    q = pid // SPLITN
    split = pid - q * SPLITN
    n = q // G
    g = q - n * G
    d0 = tl.program_id(1) * BD0 + tl.arange(0, BD0)
    m0 = d0 < D0
    d1 = tl.program_id(2) * BD1 + tl.arange(0, BD1)
    m1 = d1 < D1
    o0 = ((g * D0 + d0) * st0_c)[:, None]
    o1 = ((g * D1 + d1) * st1_c)[:, None]
    TAPS: tl.constexpr = KH * KW
    pbase = pid * NEL

    for kh in tl.static_range(KH):
        for kw in tl.static_range(KW):
            acc = tl.zeros((BD0, BD1), dtype=tl.float32)
            for sb in tl.range(split, NBLK, SPLITN):
                r = sb * BR + tl.arange(0, BR)
                mr = r < S
                y = r // Wq
                x = r - y * Wq
                if MAP_FWD:
                    ih = y * sh - ph + kh * dh
                    iw = x * sw - pw + kw * dw
                    mm = mr
                else:
                    nh = y + ph - kh * dh
                    nw = x + pw - kw * dw
                    if S1:
                        ih = nh
                        iw = nw
                        mm = mr
                    else:
                        mm = mr & (nh % sh == 0) & (nw % sw == 0)
                        ih = nh // sh
                        iw = nw // sw
                mm = mm & (ih >= 0) & (ih < Hb) & (iw >= 0) & (iw < Wb)
                if ROW_MAPPED:
                    ry = ih
                    rx = iw
                    rm = mm
                    cy = y
                    cx = x
                    cm = mr
                else:
                    ry = y
                    rx = x
                    rm = mr
                    cy = ih
                    cx = iw
                    cm = mm
                t0 = tl.load(
                    T0_ptr + n * st0_n + o0 + (ry * st0_h + rx * st0_w)[None, :],
                    mask=m0[:, None] & rm[None, :],
                    other=0.0,
                )
                t1 = tl.load(
                    T1_ptr + n * st1_n + o1 + (cy * st1_h + cx * st1_w)[None, :],
                    mask=m1[:, None] & cm[None, :],
                    other=0.0,
                )
                acc += _dot_mm(t0, tl.trans(t1), IS_F16)
            tl.store(
                part_ptr
                + pbase
                + (g * D0 + d0)[:, None] * (D1 * TAPS)
                + d1[None, :] * TAPS
                + kh * KW
                + kw,
                acc,
                mask=m0[:, None] & m1[None, :],
            )


@triton.jit
def _db_dw_reduce_kernel(part_ptr, dw_ptr, NPROG, NEL, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < NEL
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for p in tl.range(0, NPROG):
        acc += tl.load(part_ptr + p * NEL + off, mask=m, other=0.0)
    tl.store(dw_ptr + off, acc, mask=m)


def _blk(v, lo=16, hi=64):
    b = lo
    while b < v:
        b <<= 1
    return b if b < hi else hi


def _is16(x):
    d = x.dtype
    return d == torch.float16 or d == torch.bfloat16


def _convolution_double_backward(
    ggI,
    ggW,
    ggb,
    gO,
    weight,
    self,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
):
    logger.debug("GEMS_HYGON _CONVOLUTION_DOUBLE_BACKWARD")
    one_d = weight.dim() == 3
    if one_d:
        if ggI is not None:
            ggI = ggI.unsqueeze(2)
        if ggW is not None:
            ggW = ggW.unsqueeze(2)
        gO = gO.unsqueeze(2)
        weight = weight.unsqueeze(2)
        self = self.unsqueeze(2)
        sh_ = 1
        sw_ = stride[0]
        ph_ = 0
        pw_ = padding[0]
        dh_ = 1
        dw_ = dilation[0]
    else:
        sh_ = stride[0]
        sw_ = stride[1]
        ph_ = padding[0]
        pw_ = padding[1]
        dh_ = dilation[0]
        dw_ = dilation[1]

    N, Cin, Hin, Win = self.shape
    Cout = gO.shape[1]
    Hout = gO.shape[2]
    Wout = gO.shape[3]
    Cin_g = Cin // groups
    Cout_g = Cout // groups
    KH = weight.shape[2]
    KW = weight.shape[3]
    S1 = (sh_ == 1) and (sw_ == 1)
    s_out = Hout * Wout
    s_in = Hin * Win
    m0 = output_mask[0]
    m1 = output_mask[1]
    m2 = output_mask[2]
    # Measured behaviour of the target aten reference: output_mask does not
    # suppress an output; an entry is returned as None only when the operand
    # that output requires is absent (grad_self needs ggW, grad_weight needs
    # ggI).  Keep that convention.
    m0 = True
    m1 = ggW is not None
    m2 = ggI is not None
    m0 = m2 or m1 or (ggb is not None)
    o0 = o1 = o2 = None

    # ---------------- grad_ggO ----------------
    if m0:
        out = torch.empty(gO.shape, dtype=gO.dtype, device=gO.device)
        has1 = ggI is not None
        has2 = ggW is not None
        hasb = ggb is not None
        a1 = weight
        w0 = a1.stride(0)
        w1 = a1.stride(1)
        if transposed:
            sa1_0 = w1
            sa1_1 = w0
            gf1 = Cin_g * w0
        else:
            sa1_0 = w0
            sa1_1 = w1
            gf1 = Cout_g * w0
        sa1_h = a1.stride(2)
        sa1_w = a1.stride(3)
        a2 = ggW if has2 else weight
        w0 = a2.stride(0)
        w1 = a2.stride(1)
        if transposed:
            sa2_0 = w1
            sa2_1 = w0
            gf2 = Cin_g * w0
        else:
            sa2_0 = w0
            sa2_1 = w1
            gf2 = Cout_g * w0
        sa2_h = a2.stride(2)
        sa2_w = a2.stride(3)
        b1 = ggI if has1 else self
        b2 = self
        bias = ggb if hasb else self
        BOC = _blk(Cout_g)
        BS = _blk(s_out)
        BK = _blk(Cin_g)
        grid = (N * groups, triton.cdiv(Cout_g, BOC), triton.cdiv(s_out, BS))
        _db_gather_kernel[grid](
            out,
            a1,
            b1,
            a2,
            b2,
            bias,
            Cout,
            Cout_g,
            s_out,
            Wout,
            Hin,
            Win,
            Cin_g,
            sh_,
            sw_,
            ph_,
            pw_,
            dh_,
            dw_,
            gf1,
            sa1_0,
            sa1_1,
            sa1_h,
            sa1_w,
            gf2,
            sa2_0,
            sa2_1,
            sa2_h,
            sa2_w,
            b1.stride(0),
            b1.stride(1),
            b1.stride(2),
            b1.stride(3),
            b2.stride(0),
            b2.stride(1),
            b2.stride(2),
            b2.stride(3),
            bias.stride(0) if hasb else 1,
            Cout_g,
            Cin_g,
            groups,
            KH,
            KW,
            HAS_A1=has1,
            HAS_A2=has2,
            HAS_BIAS=hasb,
            MAP_FWD=(not transposed),
            S1=S1,
            IS_F16=_is16(a1),
            KH=KH,
            KW=KW,
            BOC=BOC,
            BS=BS,
            BK=BK,
            num_warps=4,
        )
        o0 = out

    # ---------------- grad_self ----------------
    if m1 and ggW is not None:
        out = torch.empty(self.shape, dtype=self.dtype, device=self.device)
        a1 = ggW
        w0 = a1.stride(0)
        w1 = a1.stride(1)
        if transposed:
            sa1_0 = w0
            sa1_1 = w1
            gf1 = Cin_g * w0
        else:
            sa1_0 = w1
            sa1_1 = w0
            gf1 = Cout_g * w0
        b1 = gO
        BOC = _blk(Cin_g)
        BS = _blk(s_in)
        BK = _blk(Cout_g)
        grid = (N * groups, triton.cdiv(Cin_g, BOC), triton.cdiv(s_in, BS))
        _db_gather_kernel[grid](
            out,
            a1,
            b1,
            a1,
            b1,
            b1,
            Cin,
            Cin_g,
            s_in,
            Win,
            Hout,
            Wout,
            Cout_g,
            sh_,
            sw_,
            ph_,
            pw_,
            dh_,
            dw_,
            gf1,
            sa1_0,
            sa1_1,
            a1.stride(2),
            a1.stride(3),
            0,
            0,
            0,
            0,
            0,
            b1.stride(0),
            b1.stride(1),
            b1.stride(2),
            b1.stride(3),
            b1.stride(0),
            b1.stride(1),
            b1.stride(2),
            b1.stride(3),
            1,
            Cin_g,
            Cout_g,
            groups,
            KH,
            KW,
            HAS_A1=True,
            HAS_A2=False,
            HAS_BIAS=False,
            MAP_FWD=transposed,
            S1=S1,
            IS_F16=_is16(a1),
            KH=KH,
            KW=KW,
            BOC=BOC,
            BS=BS,
            BK=BK,
            num_warps=4,
        )
        o1 = out

    # ---------------- grad_weight ----------------
    if m2 and ggI is not None:
        dw = torch.empty(weight.shape, dtype=weight.dtype, device=weight.device)
        if transposed:
            t0 = ggI
            t1 = gO
            d0ch = Cin_g
            d1ch = Cout_g
            row_mapped = True
            map_fwd = False
        else:
            t0 = gO
            t1 = ggI
            d0ch = Cout_g
            d1ch = Cin_g
            row_mapped = False
            map_fwd = True
        BD0 = _blk(d0ch)
        BD1 = _blk(d1ch)
        BR = _blk(s_out, 16, 128)
        nblk = triton.cdiv(s_out, BR)
        nbase = N * groups * triton.cdiv(d0ch, BD0) * triton.cdiv(d1ch, BD1)
        # A single program owning the whole reduction leaves the device idle,
        # so split the (n, spatial-block) reduction over several programs and
        # combine the fp32 partials in a second pass.
        splitn = -(-160 // nbase)
        if splitn < 1:
            splitn = 1
        elif splitn > nblk:
            splitn = nblk
        nprog = N * groups * splitn
        nel = dw.numel()
        part = torch.empty(nprog * nel, dtype=torch.float32, device=dw.device)
        _db_zero_kernel[(triton.cdiv(nprog * nel, 4096),)](
            part,
            nprog * nel,
            BLOCK=4096,
            num_warps=4,
        )
        grid = (N * groups * splitn, triton.cdiv(d0ch, BD0), triton.cdiv(d1ch, BD1))
        nw = 4 if BD0 * BR <= 4096 else 8
        _db_dw_kernel[grid](
            part,
            t0,
            t1,
            N,
            s_out,
            Wout,
            Hin,
            Win,
            d0ch,
            d1ch,
            nblk,
            splitn,
            nel,
            sh_,
            sw_,
            ph_,
            pw_,
            dh_,
            dw_,
            t0.stride(0),
            t0.stride(1),
            t0.stride(2),
            t0.stride(3),
            t1.stride(0),
            t1.stride(1),
            t1.stride(2),
            t1.stride(3),
            groups,
            MAP_FWD=map_fwd,
            ROW_MAPPED=row_mapped,
            S1=S1,
            IS_F16=_is16(t0),
            KH=KH,
            KW=KW,
            BD0=BD0,
            BD1=BD1,
            BR=BR,
            num_warps=nw,
        )
        _db_dw_reduce_kernel[(triton.cdiv(nel, 1024),)](
            part,
            dw,
            nprog,
            nel,
            BLOCK=1024,
            num_warps=4,
        )
        o2 = dw

    if one_d:
        if o0 is not None:
            o0 = o0.squeeze(2)
        if o1 is not None:
            o1 = o1.squeeze(2)
        if o2 is not None:
            o2 = o2.squeeze(2)
    return o0, o1, o2

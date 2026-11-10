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

"""Triton implementation of aten::_convolution_double_backward (implicit GEMM).

Math (validated against the target ATen reference):
  op  = conv  when transposed is False, convT when transposed is True
  op' = the opposite direction
  grad_ggO    = op(ggI, weight) + op(self, ggW) + broadcast(ggb)
  grad_self   = op'(gO, ggW)
  grad_weight[c0,c1,k] = sum_{n,a} A[n,c0,a] * B[n, (c0//(W0/groups))*W1+c1, a*s-p+k*d]
      with (A,B) = (gO, ggI) if not transposed else (ggI, gO)

All three outputs are reductions along a K axis and run as tl.dot tiles: the conv
outputs as (out-channel x K) * (K x spatial), the weight gradient as
(c0 x (n,a)) * ((n,a) x c1).  Group-aligned output-channel blocking keeps the
input-channel index uniform across a tile.  fp32 operands use
input_precision="ieee" (exact fp32; the reference itself runs cuDNN TF32 at
~3e-4 relative), fp16/bf16 use native mma with fp32 accumulation.

The ATen op ignores output_mask (verified on target) and returns None only when
the driving second-order input is absent; the wrapper mirrors that.
"""

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------- kernels


@triton.jit
def _conv_op_kernel(
    out_ptr,
    x1_ptr,
    w1_ptr,
    x2_ptr,
    w2_ptr,
    b_ptr,
    SP,
    OWA,
    XH,
    XW,
    COG,
    CIG,
    osn,
    osc,
    osh,
    osw,
    x1sn,
    x1sc,
    x1sh,
    x1sw,
    x2sn,
    x2sc,
    x2sh,
    x2sw,
    w10,
    w11,
    w12,
    w13,
    w20,
    w21,
    w22,
    w23,
    bs,
    sh,
    ph,
    dh,
    sw,
    pw,
    dw,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
    COLLECT: tl.constexpr,
    HAS1: tl.constexpr,
    HAS2: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_IEEE: tl.constexpr,
):
    n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_s = tl.program_id(2)

    cblocks = (COG + BLOCK_C - 1) // BLOCK_C
    g = pid_c // cblocks
    cb = pid_c - g * cblocks
    cw = cb * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = cw < COG
    oc = g * COG + cw

    o = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    smask = o < SP
    ho = o // OWA
    wo = o - ho * OWA

    acc = tl.zeros([BLOCK_C, BLOCK_S], dtype=tl.float32)

    if HAS1 or HAS2:
        nkb = (CIG + BLOCK_K - 1) // BLOCK_K
        for kh in range(0, KH):
            if COLLECT:
                th = ho + ph - kh * dh
                tp = tl.maximum(th, 0)
                ih = tp // sh
                vh = (th >= 0) & (tp - ih * sh == 0) & (ih < XH)
            else:
                ih = ho * sh - ph + kh * dh
                vh = (ih >= 0) & (ih < XH)
            ihc = tl.maximum(tl.minimum(ih, XH - 1), 0)
            for kw in range(0, KW):
                if COLLECT:
                    tw = wo + pw - kw * dw
                    tq = tl.maximum(tw, 0)
                    iw = tq // sw
                    vw = (tw >= 0) & (tq - iw * sw == 0) & (iw < XW)
                else:
                    iw = wo * sw - pw + kw * dw
                    vw = (iw >= 0) & (iw < XW)
                iwc = tl.maximum(tl.minimum(iw, XW - 1), 0)
                m = smask & vh & vw
                for kb in range(0, nkb):
                    ci = kb * BLOCK_K + tl.arange(0, BLOCK_K)
                    kmask = ci < CIG
                    ich = g * CIG + ci
                    xoff = (
                        n * x1sn
                        + ich[:, None] * x1sc
                        + ihc[None, :] * x1sh
                        + iwc[None, :] * x1sw
                    )
                    xoff2 = (
                        n * x2sn
                        + ich[:, None] * x2sc
                        + ihc[None, :] * x2sh
                        + iwc[None, :] * x2sw
                    )
                    xm = kmask[:, None] & m[None, :]
                    if COLLECT:
                        wo1 = (
                            ich[None, :] * w10 + cw[:, None] * w11 + kh * w12 + kw * w13
                        )
                        wo2 = (
                            ich[None, :] * w20 + cw[:, None] * w21 + kh * w22 + kw * w23
                        )
                    else:
                        wo1 = (
                            oc[:, None] * w10 + ci[None, :] * w11 + kh * w12 + kw * w13
                        )
                        wo2 = (
                            oc[:, None] * w20 + ci[None, :] * w21 + kh * w22 + kw * w23
                        )
                    wm = cmask[:, None] & kmask[None, :]
                    if HAS1:
                        xt1 = tl.load(x1_ptr + xoff, mask=xm, other=0.0)
                        wt1 = tl.load(w1_ptr + wo1, mask=wm, other=0.0)
                        if USE_IEEE:
                            acc = tl.dot(wt1, xt1, acc, input_precision="ieee")
                        else:
                            acc = tl.dot(wt1, xt1, acc)
                    if HAS2:
                        xt2 = tl.load(x2_ptr + xoff2, mask=xm, other=0.0)
                        wt2 = tl.load(w2_ptr + wo2, mask=wm, other=0.0)
                        if USE_IEEE:
                            acc = tl.dot(wt2, xt2, acc, input_precision="ieee")
                        else:
                            acc = tl.dot(wt2, xt2, acc)

    if HAS_BIAS:
        bv = tl.load(b_ptr + oc * bs, mask=cmask, other=0.0)
        acc += bv[:, None]

    ooff = n * osn + oc[:, None] * osc + ho[None, :] * osh + wo[None, :] * osw
    tl.store(out_ptr + ooff, acc, mask=cmask[:, None] & smask[None, :])


@triton.jit
def _weight_grad_kernel(
    gw_ptr,
    a_ptr,
    b_ptr,
    NA,
    SPa,
    OWA,
    BH,
    BW,
    C1,
    COG,
    TMPK,
    asn,
    asc,
    ash,
    asw,
    bsn,
    bsc,
    bsh,
    bsw,
    g0s,
    g1s,
    g2s,
    g3s,
    sh,
    ph,
    dh,
    sw,
    pw,
    dw,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    KSPLIT: tl.constexpr,
    USE_IEEE: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid_m = pid0 // KSPLIT
    pid_k = pid0 - pid_m * KSPLIT
    pid_n = tl.program_id(1)
    tap = tl.program_id(2)

    mblocks = (COG + BLOCK_M - 1) // BLOCK_M
    g = pid_m // mblocks
    mb = pid_m - g * mblocks
    cwo = mb * BLOCK_M + tl.arange(0, BLOCK_M)
    cmask = cwo < COG
    c0 = g * COG + cwo

    c1 = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = c1 < C1
    kh = tap // KW
    kw = tap - kh * KW
    chb = g * C1 + c1

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for kb in range(pid_k * BLOCK_K, NA, KSPLIT * BLOCK_K):
        idx = kb + tl.arange(0, BLOCK_K)
        m0 = idx < NA
        n = idx // SPa
        a = idx - n * SPa
        ah = a // OWA
        aw = a - ah * OWA
        ih = ah * sh - ph + kh * dh
        iw = aw * sw - pw + kw * dw
        vh = (ih >= 0) & (ih < BH)
        vw = (iw >= 0) & (iw < BW)
        ihc = tl.maximum(tl.minimum(ih, BH - 1), 0)
        iwc = tl.maximum(tl.minimum(iw, BW - 1), 0)
        at = tl.load(
            a_ptr
            + (
                n[None, :] * asn
                + c0[:, None] * asc
                + ah[None, :] * ash
                + aw[None, :] * asw
            ),
            mask=cmask[:, None] & m0[None, :],
            other=0.0,
        )
        bt = tl.load(
            b_ptr
            + (
                n[:, None] * bsn
                + chb[None, :] * bsc
                + ihc[:, None] * bsh
                + iwc[:, None] * bsw
            ),
            mask=(m0 & vh & vw)[:, None] & nmask[None, :],
            other=0.0,
        )
        if USE_IEEE:
            acc = tl.dot(at, bt, acc, input_precision="ieee")
        else:
            acc = tl.dot(at, bt, acc)

    off = c0[:, None] * g0s + c1[None, :] * g1s + kh * g2s + kw * g3s
    omask = cmask[:, None] & nmask[None, :]
    if KSPLIT == 1:
        tl.store(gw_ptr + off, acc, mask=omask)
    else:
        tl.store(gw_ptr + pid_k * TMPK + off, acc, mask=omask)


@triton.jit
def _reduce_split_kernel(out_ptr, tmp_ptr, W, TMPK, KSPLIT, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = idx < W
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for k in range(0, KSPLIT):
        acc += tl.load(tmp_ptr + k * TMPK + idx, mask=m, other=0.0)
    tl.store(out_ptr + idx, acc, mask=m)


# ---------------------------------------------------------------- helpers


def _d4(t):
    s = t.shape
    if t.dim() == 3:
        return int(s[0]), int(s[1]), 1, int(s[2])
    return int(s[0]), int(s[1]), int(s[2]), int(s[3])


def _s4(t):
    if t.dim() == 3:
        return int(t.stride(0)), int(t.stride(1)), 0, int(t.stride(2))
    return int(t.stride(0)), int(t.stride(1)), int(t.stride(2)), int(t.stride(3))


def _alloc(shape, dtype, device):
    t = torch.empty(tuple(int(v) for v in shape), dtype=dtype, device=device)
    st = t.stride()
    if t.dim() == 3:
        return t, (int(st[0]), int(st[1]), 0, int(st[2]))
    return t, (int(st[0]), int(st[1]), int(st[2]), int(st[3]))


def _p2(v):
    v = int(v)
    return 1 if v <= 1 else 1 << (v - 1).bit_length()


# ---------------------------------------------------------------- entry


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
    nd = len(stride)
    if nd == 1:
        st = (1, int(stride[0]))
        pd = (0, int(padding[0]))
        dl = (1, int(dilation[0]))
    else:
        st = (int(stride[0]), int(stride[1]))
        pd = (int(padding[0]), int(padding[1]))
        dl = (int(dilation[0]), int(dilation[1]))
    g = int(groups)
    transposed = bool(transposed)

    W = weight
    W0, W1 = int(W.shape[0]), int(W.shape[1])
    KH = int(W.shape[-2]) if W.dim() == 4 else 1
    KW = int(W.shape[-1])
    w4 = _s4(W)
    use_ieee = W.dtype == torch.float32

    res = [None, None, None]

    # ---------------- grad_ggO (shape of gO) ----------------
    if (ggI is not None) or (ggW is not None) or (ggb is not None):
        out, ost = _alloc(gO.shape, gO.dtype, gO.device)
        N, C, H, Wd = _d4(gO)
        SP = H * Wd
        collect = transposed
        COG = W1 if collect else W0 // g
        CIG = (W0 // g) if collect else W1
        XH, XW = _d4(self)[2], _d4(self)[3]

        terms = []
        if ggI is not None:
            terms.append((ggI, W, w4))
        if ggW is not None:
            terms.append((self, ggW, _s4(ggW)))
        while len(terms) < 2:
            terms.append((self, W, w4))
        x1t, w1t, w1st = terms[0]
        x2t, w2t, w2st = terms[1]
        x1 = _s4(x1t)
        x2 = _s4(x2t)
        has1 = (ggI is not None) or (ggW is not None)
        has2 = (ggI is not None) and (ggW is not None)

        bias = ggb if ggb is not None else gO
        bstride = int(ggb.stride(0)) if ggb is not None else 1

        BC = max(16, min(64, _p2(COG)))
        BS = max(16, min(64, _p2(min(Wd, SP))))
        BK = max(16, min(32, _p2(max(CIG, 1))))
        if N > 0 and C > 0 and SP > 0:
            grid = (N, g * triton.cdiv(COG, BC), triton.cdiv(SP, BS))
            nw = max(1, min(8, BC * BS // 512))
            _conv_op_kernel[grid](
                out,
                x1t,
                w1t,
                x2t,
                w2t,
                bias,
                SP,
                Wd,
                XH,
                XW,
                COG,
                CIG,
                ost[0],
                ost[1],
                ost[2],
                ost[3],
                x1[0],
                x1[1],
                x1[2],
                x1[3],
                x2[0],
                x2[1],
                x2[2],
                x2[3],
                w1st[0],
                w1st[1],
                w1st[2],
                w1st[3],
                w2st[0],
                w2st[1],
                w2st[2],
                w2st[3],
                bstride,
                st[0],
                pd[0],
                dl[0],
                st[1],
                pd[1],
                dl[1],
                KH=KH,
                KW=KW,
                BLOCK_S=BS,
                BLOCK_C=BC,
                BLOCK_K=BK,
                COLLECT=collect,
                HAS1=has1,
                HAS2=has2,
                HAS_BIAS=ggb is not None,
                USE_IEEE=use_ieee,
                num_warps=nw,
            )
        res[0] = out

    # ---------------- grad_self (shape of self) ----------------
    if ggW is not None:
        out, ost = _alloc(self.shape, self.dtype, self.device)
        N, C, H, Wd = _d4(self)
        SP = H * Wd
        collect = not transposed
        COG = W1 if collect else W0 // g
        CIG = (W0 // g) if collect else W1
        gs = _s4(gO)
        wst = _s4(ggW)
        XH, XW = _d4(gO)[2], _d4(gO)[3]
        BC = max(16, min(64, _p2(COG)))
        BS = max(16, min(64, _p2(min(Wd, SP))))
        BK = max(16, min(32, _p2(max(CIG, 1))))
        if N > 0 and C > 0 and SP > 0:
            grid = (N, g * triton.cdiv(COG, BC), triton.cdiv(SP, BS))
            nw = max(1, min(8, BC * BS // 512))
            _conv_op_kernel[grid](
                out,
                gO,
                ggW,
                gO,
                ggW,
                gO,
                SP,
                Wd,
                XH,
                XW,
                COG,
                CIG,
                ost[0],
                ost[1],
                ost[2],
                ost[3],
                gs[0],
                gs[1],
                gs[2],
                gs[3],
                gs[0],
                gs[1],
                gs[2],
                gs[3],
                wst[0],
                wst[1],
                wst[2],
                wst[3],
                wst[0],
                wst[1],
                wst[2],
                wst[3],
                1,
                st[0],
                pd[0],
                dl[0],
                st[1],
                pd[1],
                dl[1],
                KH=KH,
                KW=KW,
                BLOCK_S=BS,
                BLOCK_C=BC,
                BLOCK_K=BK,
                COLLECT=collect,
                HAS1=True,
                HAS2=False,
                HAS_BIAS=False,
                USE_IEEE=use_ieee,
                num_warps=nw,
            )
        res[1] = out

    # ---------------- grad_weight (shape of weight) ----------------
    if ggI is not None:
        gw, gst = _alloc(W.shape, W.dtype, W.device)
        if transposed:
            A, B = ggI, gO
        else:
            A, B = gO, ggI
        Na, Ca, Ha, Wa = _d4(A)
        Nb, Cb, Hb, Wb = _d4(B)
        SPa = Ha * Wa
        NA = Na * SPa
        ast = _s4(A)
        bst = _s4(B)
        COG = W0 // g
        W_elems = W0 * W1 * KH * KW
        if W_elems > 0 and NA > 0 and SPa > 0:
            BM = max(16, min(64, _p2(COG)))
            BN = max(16, min(32, _p2(W1)))
            BK = max(16, min(64, _p2(min(NA, 64))))
            mblk = triton.cdiv(COG, BM)
            nblk = triton.cdiv(W1, BN)
            base = g * mblk * nblk * KH * KW
            ksplit = 1
            while base * ksplit < 96 and ksplit < 16:
                ksplit *= 2
            ksplit = min(ksplit, max(1, triton.cdiv(NA, BK)))
            if ksplit > 1:
                tmp = torch.empty(
                    (ksplit, W_elems), dtype=torch.float32, device=W.device
                )
                outbuf = tmp
                tmpk = W_elems
            else:
                tmp = None
                outbuf = gw
                tmpk = 0
            grid = (g * mblk * ksplit, nblk, KH * KW)
            nw = max(2, min(8, BM * BN // 256))
            _weight_grad_kernel[grid](
                outbuf,
                A,
                B,
                NA,
                SPa,
                Wa,
                Hb,
                Wb,
                W1,
                COG,
                tmpk,
                ast[0],
                ast[1],
                ast[2],
                ast[3],
                bst[0],
                bst[1],
                bst[2],
                bst[3],
                gst[0],
                gst[1],
                gst[2],
                gst[3],
                st[0],
                pd[0],
                dl[0],
                st[1],
                pd[1],
                dl[1],
                KH=KH,
                KW=KW,
                BLOCK_M=BM,
                BLOCK_N=BN,
                BLOCK_K=BK,
                KSPLIT=ksplit,
                USE_IEEE=use_ieee,
                num_warps=nw,
            )
            if tmp is not None:
                rb = 256
                _reduce_split_kernel[(triton.cdiv(W_elems, rb),)](
                    gw, tmp, W_elems, W_elems, ksplit, BLOCK=rb, num_warps=4
                )
        res[2] = gw

    return tuple(res)

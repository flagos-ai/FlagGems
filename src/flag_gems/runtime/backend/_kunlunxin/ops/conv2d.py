# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Device-resident NCHW convolution for Kunlunxin XPUs."""

import logging
import threading

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def conv2d_output_size(in_size, kernel_size, stride, padding, dilation):
    return (in_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


@libentry()
@triton.jit
def _forward(
    x,
    w,
    b,
    y,
    n,
    hin,
    win,
    cout,
    hout,
    wout,
    cpg,
    opg,
    kh,
    kw,
    sh,
    sw,
    ph,
    pw,
    dh,
    dw,
    xsn,
    xsc,
    xsh,
    xsw,
    wso,
    wsi,
    wsh,
    wsw,
    ysn,
    ysc,
    ysh,
    ysw,
    HAS_BIAS: tl.constexpr,
    CPG: tl.constexpr,
    OPG: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    m = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ow = m % wout
    q = m // wout
    oh = q % hout
    q = q // hout
    oc = q % cout
    ni = q // cout
    plane = hout * wout
    group = oc // OPG
    acc = tl.zeros((BLOCK,), tl.float32)
    for r in range(KH):
        ih = oh * sh - ph + r * dh
        for s in range(KW):
            iw = ow * sw - pw + s * dw
            valid = (ih >= 0) & (ih < hin) & (iw >= 0) & (iw < win)
            safe_ih = tl.where(valid, ih, 0)
            safe_iw = tl.where(valid, iw, 0)
            for ci in range(CPG):
                xv = tl.load(
                    x
                    + ni * xsn
                    + (group * CPG + ci) * xsc
                    + safe_ih * xsh
                    + safe_iw * xsw,
                    mask=m < n * cout * plane,
                    other=0.0,
                )
                xv = tl.where(valid, xv, 0.0)
                wv = tl.load(w + oc * wso + ci * wsi + r * wsh + s * wsw)
                acc += xv.to(tl.float32) * wv.to(tl.float32)
    if HAS_BIAS:
        acc += tl.load(b + oc).to(tl.float32)
    tl.store(
        y + ni * ysn + oc * ysc + oh * ysh + ow * ysw,
        acc,
        mask=m < n * cout * plane,
    )


@libentry()
@triton.jit
def _forward_spatial_tile(
    x,
    w,
    b,
    y,
    n,
    hin,
    win,
    cout,
    hout,
    wout,
    cpg,
    opg,
    kh,
    kw,
    sh,
    sw,
    ph,
    pw,
    dh,
    dw,
    xsn,
    xsc,
    xsh,
    xsw,
    wso,
    wsi,
    wsh,
    wsw,
    ysn,
    ysc,
    ysh,
    ysw,
    HAS_BIAS: tl.constexpr,
    CPG: tl.constexpr,
    OPG: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    oc = tl.program_id(1)
    plane = hout * wout
    ni = p // plane
    q = p % plane
    oh, ow = q // wout, q % wout
    group = oc // OPG
    mask = p < n * plane
    acc = tl.zeros((BLOCK,), tl.float32)
    for r in range(KH):
        ih = oh * sh - ph + r * dh
        for s in range(KW):
            iw = ow * sw - pw + s * dw
            valid = mask & (ih >= 0) & (ih < hin) & (iw >= 0) & (iw < win)
            safe_ih = tl.where(valid, ih, 0)
            safe_iw = tl.where(valid, iw, 0)
            for ci in range(CPG):
                xv = tl.load(
                    x
                    + ni * xsn
                    + (group * CPG + ci) * xsc
                    + safe_ih * xsh
                    + safe_iw * xsw,
                    mask=mask,
                    other=0.0,
                )
                xv = tl.where(valid, xv, 0.0)
                wv = tl.load(w + oc * wso + ci * wsi + r * wsh + s * wsw)
                acc += xv.to(tl.float32) * wv.to(tl.float32)
    if HAS_BIAS:
        acc += tl.load(b + oc).to(tl.float32)
    tl.store(
        y + ni * ysn + oc * ysc + oh * ysh + ow * ysw,
        acc,
        mask=mask,
    )


@libentry()
@triton.jit
def _forward_spatial_channels4(
    x,
    w,
    b,
    y,
    n,
    hin,
    win,
    cout,
    hout,
    wout,
    cpg,
    kh,
    kw,
    sh,
    sw,
    ph,
    pw,
    dh,
    dw,
    xsn,
    xsc,
    xsh,
    xsw,
    wso,
    wsi,
    wsh,
    wsw,
    ysn,
    ysc,
    ysh,
    ysw,
    HAS_BIAS: tl.constexpr,
    CPG: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    oc = tl.program_id(1) * 4
    plane = hout * wout
    ni = p // plane
    q = p % plane
    oh, ow = q // wout, q % wout
    pmask = p < n * plane
    m0, m1, m2, m3 = oc < cout, oc + 1 < cout, oc + 2 < cout, oc + 3 < cout
    acc0 = tl.zeros((BLOCK,), tl.float32)
    acc1 = tl.zeros((BLOCK,), tl.float32)
    acc2 = tl.zeros((BLOCK,), tl.float32)
    acc3 = tl.zeros((BLOCK,), tl.float32)
    for r in range(KH):
        ih = oh * sh - ph + r * dh
        for s in range(KW):
            iw = ow * sw - pw + s * dw
            valid = pmask & (ih >= 0) & (ih < hin) & (iw >= 0) & (iw < win)
            safe_ih = tl.where(valid, ih, 0)
            safe_iw = tl.where(valid, iw, 0)
            for ci in range(CPG):
                xv = tl.load(
                    x + ni * xsn + ci * xsc + safe_ih * xsh + safe_iw * xsw,
                    mask=pmask,
                    other=0.0,
                )
                xv = tl.where(valid, xv, 0.0).to(tl.float32)
                acc0 += xv * tl.load(
                    w + oc * wso + ci * wsi + r * wsh + s * wsw, mask=m0, other=0.0
                ).to(tl.float32)
                acc1 += xv * tl.load(
                    w + (oc + 1) * wso + ci * wsi + r * wsh + s * wsw,
                    mask=m1,
                    other=0.0,
                ).to(tl.float32)
                acc2 += xv * tl.load(
                    w + (oc + 2) * wso + ci * wsi + r * wsh + s * wsw,
                    mask=m2,
                    other=0.0,
                ).to(tl.float32)
                acc3 += xv * tl.load(
                    w + (oc + 3) * wso + ci * wsi + r * wsh + s * wsw,
                    mask=m3,
                    other=0.0,
                ).to(tl.float32)
    if HAS_BIAS:
        acc0 += tl.load(b + oc, mask=m0, other=0.0).to(tl.float32)
        acc1 += tl.load(b + oc + 1, mask=m1, other=0.0).to(tl.float32)
        acc2 += tl.load(b + oc + 2, mask=m2, other=0.0).to(tl.float32)
        acc3 += tl.load(b + oc + 3, mask=m3, other=0.0).to(tl.float32)
    tl.store(y + ni * ysn + oc * ysc + oh * ysh + ow * ysw, acc0, mask=pmask & m0)
    tl.store(y + ni * ysn + (oc + 1) * ysc + oh * ysh + ow * ysw, acc1, mask=pmask & m1)
    tl.store(y + ni * ysn + (oc + 2) * ysc + oh * ysh + ow * ysw, acc2, mask=pmask & m2)
    tl.store(y + ni * ysn + (oc + 3) * ysc + oh * ysh + ow * ysw, acc3, mask=pmask & m3)


@libentry()
@triton.jit
def _forward_spatial_channels8(
    x,
    w,
    b,
    y,
    n,
    hin,
    win,
    cout,
    hout,
    wout,
    cpg,
    kh,
    kw,
    sh,
    sw,
    ph,
    pw,
    dh,
    dw,
    xsn,
    xsc,
    xsh,
    xsw,
    wso,
    wsi,
    wsh,
    wsw,
    ysn,
    ysc,
    ysh,
    ysw,
    HAS_BIAS: tl.constexpr,
    CPG: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    oc = tl.program_id(1) * 8
    plane = hout * wout
    ni = p // plane
    q = p % plane
    oh, ow = q // wout, q % wout
    pmask = p < n * plane
    acc0 = tl.zeros((BLOCK,), tl.float32)
    acc1 = tl.zeros((BLOCK,), tl.float32)
    acc2 = tl.zeros((BLOCK,), tl.float32)
    acc3 = tl.zeros((BLOCK,), tl.float32)
    acc4 = tl.zeros((BLOCK,), tl.float32)
    acc5 = tl.zeros((BLOCK,), tl.float32)
    acc6 = tl.zeros((BLOCK,), tl.float32)
    acc7 = tl.zeros((BLOCK,), tl.float32)
    for r in range(KH):
        ih = oh * sh - ph + r * dh
        for s in range(KW):
            iw = ow * sw - pw + s * dw
            valid = pmask & (ih >= 0) & (ih < hin) & (iw >= 0) & (iw < win)
            safe_ih = tl.where(valid, ih, 0)
            safe_iw = tl.where(valid, iw, 0)
            for ci in range(CPG):
                xv = tl.load(
                    x + ni * xsn + ci * xsc + safe_ih * xsh + safe_iw * xsw,
                    mask=pmask,
                    other=0.0,
                )
                xv = tl.where(valid, xv, 0.0).to(tl.float32)
                acc0 += xv * tl.load(w + oc * wso + ci * wsi + r * wsh + s * wsw).to(
                    tl.float32
                )
                acc1 += xv * tl.load(
                    w + (oc + 1) * wso + ci * wsi + r * wsh + s * wsw
                ).to(tl.float32)
                acc2 += xv * tl.load(
                    w + (oc + 2) * wso + ci * wsi + r * wsh + s * wsw
                ).to(tl.float32)
                acc3 += xv * tl.load(
                    w + (oc + 3) * wso + ci * wsi + r * wsh + s * wsw
                ).to(tl.float32)
                acc4 += xv * tl.load(
                    w + (oc + 4) * wso + ci * wsi + r * wsh + s * wsw
                ).to(tl.float32)
                acc5 += xv * tl.load(
                    w + (oc + 5) * wso + ci * wsi + r * wsh + s * wsw
                ).to(tl.float32)
                acc6 += xv * tl.load(
                    w + (oc + 6) * wso + ci * wsi + r * wsh + s * wsw
                ).to(tl.float32)
                acc7 += xv * tl.load(
                    w + (oc + 7) * wso + ci * wsi + r * wsh + s * wsw
                ).to(tl.float32)
    if HAS_BIAS:
        acc0 += tl.load(b + oc).to(tl.float32)
        acc1 += tl.load(b + oc + 1).to(tl.float32)
        acc2 += tl.load(b + oc + 2).to(tl.float32)
        acc3 += tl.load(b + oc + 3).to(tl.float32)
        acc4 += tl.load(b + oc + 4).to(tl.float32)
        acc5 += tl.load(b + oc + 5).to(tl.float32)
        acc6 += tl.load(b + oc + 6).to(tl.float32)
        acc7 += tl.load(b + oc + 7).to(tl.float32)
    tl.store(y + ni * ysn + oc * ysc + oh * ysh + ow * ysw, acc0, mask=pmask)
    tl.store(y + ni * ysn + (oc + 1) * ysc + oh * ysh + ow * ysw, acc1, mask=pmask)
    tl.store(y + ni * ysn + (oc + 2) * ysc + oh * ysh + ow * ysw, acc2, mask=pmask)
    tl.store(y + ni * ysn + (oc + 3) * ysc + oh * ysh + ow * ysw, acc3, mask=pmask)
    tl.store(y + ni * ysn + (oc + 4) * ysc + oh * ysh + ow * ysw, acc4, mask=pmask)
    tl.store(y + ni * ysn + (oc + 5) * ysc + oh * ysh + ow * ysw, acc5, mask=pmask)
    tl.store(y + ni * ysn + (oc + 6) * ysc + oh * ysh + ow * ysw, acc6, mask=pmask)
    tl.store(y + ni * ysn + (oc + 7) * ysc + oh * ysh + ow * ysw, acc7, mask=pmask)


@libentry()
@triton.jit
def _forward_spatial_channels16(
    x,
    w,
    b,
    y,
    n,
    hin,
    win,
    cout,
    hout,
    wout,
    cpg,
    kh,
    kw,
    sh,
    sw,
    ph,
    pw,
    dh,
    dw,
    xsn,
    xsc,
    xsh,
    xsw,
    wso,
    wsi,
    wsh,
    wsw,
    ysn,
    ysc,
    ysh,
    ysw,
    HAS_BIAS: tl.constexpr,
    CPG: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    oc = tl.program_id(1) * 16
    plane = hout * wout
    ni = p // plane
    q = p % plane
    oh, ow = q // wout, q % wout
    pmask = p < n * plane
    acc0 = tl.zeros((BLOCK,), tl.float32)
    acc1 = tl.zeros((BLOCK,), tl.float32)
    acc2 = tl.zeros((BLOCK,), tl.float32)
    acc3 = tl.zeros((BLOCK,), tl.float32)
    acc4 = tl.zeros((BLOCK,), tl.float32)
    acc5 = tl.zeros((BLOCK,), tl.float32)
    acc6 = tl.zeros((BLOCK,), tl.float32)
    acc7 = tl.zeros((BLOCK,), tl.float32)
    acc8 = tl.zeros((BLOCK,), tl.float32)
    acc9 = tl.zeros((BLOCK,), tl.float32)
    acc10 = tl.zeros((BLOCK,), tl.float32)
    acc11 = tl.zeros((BLOCK,), tl.float32)
    acc12 = tl.zeros((BLOCK,), tl.float32)
    acc13 = tl.zeros((BLOCK,), tl.float32)
    acc14 = tl.zeros((BLOCK,), tl.float32)
    acc15 = tl.zeros((BLOCK,), tl.float32)
    for r in range(KH):
        ih = oh * sh - ph + r * dh
        for s in range(KW):
            iw = ow * sw - pw + s * dw
            valid = pmask & (ih >= 0) & (ih < hin) & (iw >= 0) & (iw < win)
            safe_ih = tl.where(valid, ih, 0)
            safe_iw = tl.where(valid, iw, 0)
            for ci in range(CPG):
                xv = tl.load(
                    x + ni * xsn + ci * xsc + safe_ih * xsh + safe_iw * xsw,
                    mask=pmask,
                    other=0.0,
                )
                xv = tl.where(valid, xv, 0.0).to(tl.float32)
                acc0 += xv * tl.load(w + oc * wso + ci * wsi + r * wsh + s * wsw).to(
                    tl.float32
                )
                acc1 += xv * tl.load(
                    w + (oc + 1) * wso + ci * wsi + r * wsh + s * wsw
                ).to(tl.float32)
                acc2 += xv * tl.load(
                    w + (oc + 2) * wso + ci * wsi + r * wsh + s * wsw
                ).to(tl.float32)
                acc3 += xv * tl.load(
                    w + (oc + 3) * wso + ci * wsi + r * wsh + s * wsw
                ).to(tl.float32)
                acc4 += xv * tl.load(
                    w + (oc + 4) * wso + ci * wsi + r * wsh + s * wsw
                ).to(tl.float32)
                acc5 += xv * tl.load(
                    w + (oc + 5) * wso + ci * wsi + r * wsh + s * wsw
                ).to(tl.float32)
                acc6 += xv * tl.load(
                    w + (oc + 6) * wso + ci * wsi + r * wsh + s * wsw
                ).to(tl.float32)
                acc7 += xv * tl.load(
                    w + (oc + 7) * wso + ci * wsi + r * wsh + s * wsw
                ).to(tl.float32)
                acc8 += xv * tl.load(
                    w + (oc + 8) * wso + ci * wsi + r * wsh + s * wsw
                ).to(tl.float32)
                acc9 += xv * tl.load(
                    w + (oc + 9) * wso + ci * wsi + r * wsh + s * wsw
                ).to(tl.float32)
                acc10 += xv * tl.load(
                    w + (oc + 10) * wso + ci * wsi + r * wsh + s * wsw
                ).to(tl.float32)
                acc11 += xv * tl.load(
                    w + (oc + 11) * wso + ci * wsi + r * wsh + s * wsw
                ).to(tl.float32)
                acc12 += xv * tl.load(
                    w + (oc + 12) * wso + ci * wsi + r * wsh + s * wsw
                ).to(tl.float32)
                acc13 += xv * tl.load(
                    w + (oc + 13) * wso + ci * wsi + r * wsh + s * wsw
                ).to(tl.float32)
                acc14 += xv * tl.load(
                    w + (oc + 14) * wso + ci * wsi + r * wsh + s * wsw
                ).to(tl.float32)
                acc15 += xv * tl.load(
                    w + (oc + 15) * wso + ci * wsi + r * wsh + s * wsw
                ).to(tl.float32)
    if HAS_BIAS:
        acc0 += tl.load(b + oc).to(tl.float32)
        acc1 += tl.load(b + oc + 1).to(tl.float32)
        acc2 += tl.load(b + oc + 2).to(tl.float32)
        acc3 += tl.load(b + oc + 3).to(tl.float32)
        acc4 += tl.load(b + oc + 4).to(tl.float32)
        acc5 += tl.load(b + oc + 5).to(tl.float32)
        acc6 += tl.load(b + oc + 6).to(tl.float32)
        acc7 += tl.load(b + oc + 7).to(tl.float32)
        acc8 += tl.load(b + oc + 8).to(tl.float32)
        acc9 += tl.load(b + oc + 9).to(tl.float32)
        acc10 += tl.load(b + oc + 10).to(tl.float32)
        acc11 += tl.load(b + oc + 11).to(tl.float32)
        acc12 += tl.load(b + oc + 12).to(tl.float32)
        acc13 += tl.load(b + oc + 13).to(tl.float32)
        acc14 += tl.load(b + oc + 14).to(tl.float32)
        acc15 += tl.load(b + oc + 15).to(tl.float32)
    tl.store(y + ni * ysn + oc * ysc + oh * ysh + ow * ysw, acc0, mask=pmask)
    tl.store(y + ni * ysn + (oc + 1) * ysc + oh * ysh + ow * ysw, acc1, mask=pmask)
    tl.store(y + ni * ysn + (oc + 2) * ysc + oh * ysh + ow * ysw, acc2, mask=pmask)
    tl.store(y + ni * ysn + (oc + 3) * ysc + oh * ysh + ow * ysw, acc3, mask=pmask)
    tl.store(y + ni * ysn + (oc + 4) * ysc + oh * ysh + ow * ysw, acc4, mask=pmask)
    tl.store(y + ni * ysn + (oc + 5) * ysc + oh * ysh + ow * ysw, acc5, mask=pmask)
    tl.store(y + ni * ysn + (oc + 6) * ysc + oh * ysh + ow * ysw, acc6, mask=pmask)
    tl.store(y + ni * ysn + (oc + 7) * ysc + oh * ysh + ow * ysw, acc7, mask=pmask)
    tl.store(y + ni * ysn + (oc + 8) * ysc + oh * ysh + ow * ysw, acc8, mask=pmask)
    tl.store(y + ni * ysn + (oc + 9) * ysc + oh * ysh + ow * ysw, acc9, mask=pmask)
    tl.store(y + ni * ysn + (oc + 10) * ysc + oh * ysh + ow * ysw, acc10, mask=pmask)
    tl.store(y + ni * ysn + (oc + 11) * ysc + oh * ysh + ow * ysw, acc11, mask=pmask)
    tl.store(y + ni * ysn + (oc + 12) * ysc + oh * ysh + ow * ysw, acc12, mask=pmask)
    tl.store(y + ni * ysn + (oc + 13) * ysc + oh * ysh + ow * ysw, acc13, mask=pmask)
    tl.store(y + ni * ysn + (oc + 14) * ysc + oh * ysh + ow * ysw, acc14, mask=pmask)
    tl.store(y + ni * ysn + (oc + 15) * ysc + oh * ysh + ow * ysw, acc15, mask=pmask)


@libentry()
@triton.jit
def _input_grad(
    dy,
    w,
    dx,
    n,
    cin,
    hin,
    win,
    hout,
    wout,
    cpg,
    opg,
    kh,
    kw,
    sh,
    sw,
    ph,
    pw,
    dh,
    dw,
    dysn,
    dysc,
    dysh,
    dysw,
    wso,
    wsi,
    wsh,
    wsw,
    dxsn,
    dxsc,
    dxsh,
    dxsw,
    BM: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    OPG: tl.constexpr,
    CPG: tl.constexpr,
):
    # One program per (spatial block, input channel); rank-1 throughout.
    # Out-of-range indices are clamped to 0 and the contribution is zeroed
    # afterwards with tl.where, rather than feeding them to a masked load.
    pm = tl.program_id(0)
    c = tl.program_id(1)
    m = pm * BM + tl.arange(0, BM)
    plane = hin * win
    ni = m // plane
    q = m - ni * plane
    ih, iw = q // win, q % win
    g = c // CPG
    lc = c % CPG
    mmask = ni < n
    sih = tl.where(mmask, ih, 0)
    siw = tl.where(mmask, iw, 0)
    acc = tl.zeros((BM,), tl.float32)
    for r in range(0, KH):
        hnum = ih + ph - r * dh
        oh = hnum // sh
        hvalid = (hnum == oh * sh) & (oh >= 0) & (oh < hout)
        soh = tl.where(hvalid, oh, 0)
        for s in range(0, KW):
            wnum = iw + pw - s * dw
            ow = wnum // sw
            wvalid = (wnum == ow * sw) & (ow >= 0) & (ow < wout)
            sow = tl.where(wvalid, ow, 0)
            valid = mmask & hvalid & wvalid
            for o in range(0, OPG):
                gv = tl.load(
                    dy + ni * dysn + (g * OPG + o) * dysc + soh * dysh + sow * dysw,
                    mask=mmask,
                    other=0.0,
                )
                gv = tl.where(valid, gv, 0.0)
                wv = tl.load(w + (g * OPG + o) * wso + lc * wsi + r * wsh + s * wsw)
                acc += gv.to(tl.float32) * wv.to(tl.float32)
    tl.store(dx + ni * dxsn + c * dxsc + sih * dxsh + siw * dxsw, acc, mask=mmask)


@libentry()
@triton.jit
def _weight_grad(
    x,
    dy,
    grad_w,
    n,
    hin,
    win,
    cout,
    hout,
    wout,
    cpg,
    opg,
    kh,
    kw,
    sh,
    sw,
    ph,
    pw,
    dh,
    dw,
    xsn,
    xsc,
    xsh,
    xsw,
    dysn,
    dysc,
    dysh,
    dysw,
    gws0,
    gws1,
    gws2,
    gws3,
    BP: tl.constexpr,
    BK: tl.constexpr,
):
    # Rank-1 throughout, one program per weight element, with the spatial
    # reduction carried by a scalar loop over p and a rank-1 block reduction.
    # grid == weight.numel(), so oc < cout always holds.
    k = tl.program_id(0)
    area = cpg * kh * kw
    oc = k // area
    rem = k % area
    ci = rem // (kh * kw)
    rem = rem % (kh * kw)
    r = rem // kw
    s = rem % kw
    g = oc // opg
    total = n * hout * wout
    acc = tl.zeros((BP,), tl.float32)
    for pbase in range(0, total, BP):
        p = pbase + tl.arange(0, BP)
        ni = p // (hout * wout)
        q = p - ni * (hout * wout)
        oh = q // wout
        ow = q - oh * wout
        # Clamp every index that can leave the tensor, then zero the
        # contribution with tl.where, instead of feeding an out-of-range
        # index to a masked load.
        pv = p < total
        sni = tl.where(pv, ni, 0)
        soh = tl.where(pv, oh, 0)
        sow = tl.where(pv, ow, 0)
        ih = oh * sh - ph + r * dh
        iw = ow * sw - pw + s * dw
        valid = pv & (ih >= 0) & (ih < hin) & (iw >= 0) & (iw < win)
        sih = tl.where(valid, ih, 0)
        siw = tl.where(valid, iw, 0)
        xv = tl.load(
            x + sni * xsn + (g * cpg + ci) * xsc + sih * xsh + siw * xsw,
            mask=pv,
            other=0.0,
        )
        xv = tl.where(valid, xv, 0.0)
        gy = tl.load(
            dy + sni * dysn + oc * dysc + soh * dysh + sow * dysw,
            mask=pv,
            other=0.0,
        )
        acc += xv.to(tl.float32) * gy.to(tl.float32)
    tl.store(grad_w + oc * gws0 + ci * gws1 + r * gws2 + s * gws3, tl.sum(acc, axis=0))


@libentry()
@triton.jit
def _bias_grad(
    dy,
    grad_b,
    n,
    cout,
    hout,
    wout,
    dysn,
    dysc,
    dysh,
    dysw,
    BP: tl.constexpr,
    BO: tl.constexpr,
):
    # Rank-1, one program per output channel.
    o = tl.program_id(0)
    total = n * hout * wout
    acc = tl.zeros((BP,), tl.float32)
    for pbase in range(0, total, BP):
        p = pbase + tl.arange(0, BP)
        ni = p // (hout * wout)
        q = p - ni * (hout * wout)
        oh = q // wout
        ow = q - oh * wout
        pv = p < total
        sni = tl.where(pv, ni, 0)
        soh = tl.where(pv, oh, 0)
        sow = tl.where(pv, ow, 0)
        acc += tl.load(
            dy + sni * dysn + o * dysc + soh * dysh + sow * dysw,
            mask=pv,
            other=0.0,
        )
    tl.store(grad_b + o, tl.sum(acc, axis=0))


def _pair(value, name):
    if isinstance(value, int):
        return value, value
    if (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and all(isinstance(v, int) for v in value)
    ):
        return value
    raise RuntimeError(f"conv2d(): {name} must be an int or pair of ints")


def _output_shape(padding, kh, kw, dh, dw, sh, sw, hin, win):
    if isinstance(padding, str):
        if padding == "valid":
            return (
                0,
                0,
                (hin - dh * (kh - 1) - 1) // sh + 1,
                (win - dw * (kw - 1) - 1) // sw + 1,
            )
        if padding == "same" and sh == 1 and sw == 1:
            return (dh * (kh - 1)) // 2, (dw * (kw - 1)) // 2, hin, win
        raise RuntimeError("conv2d only supports padding='same' with stride 1")
    ph, pw = _pair(padding, "padding")
    return (
        ph,
        pw,
        (hin + 2 * ph - dh * (kh - 1) - 1) // sh + 1,
        (win + 2 * pw - dw * (kw - 1) - 1) // sw + 1,
    )


class Conv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups):
        if input.ndim != 4 or weight.ndim != 4:
            raise RuntimeError("conv2d expects NCHW input and OIHW weights")
        if (
            groups <= 0
            or input.shape[1] % groups
            or weight.shape[0] % groups
            or weight.shape[1] * groups != input.shape[1]
        ):
            raise RuntimeError(
                "conv2d input, weight, and groups have incompatible channels"
            )
        if bias is not None and (bias.ndim != 1 or bias.numel() != weight.shape[0]):
            raise RuntimeError("conv2d bias must contain one value per output channel")
        sh, sw = _pair(stride, "stride")
        dh, dw = _pair(dilation, "dilation")
        if min(sh, sw, dh, dw) <= 0:
            raise RuntimeError("conv2d stride and dilation must be positive")
        n, _, hin, win = input.shape
        cout, cpg, kh, kw = weight.shape
        ph, pw, hout, wout = _output_shape(padding, kh, kw, dh, dw, sh, sw, hin, win)
        if min(ph, pw) < 0 or min(hout, wout) <= 0:
            raise RuntimeError("conv2d calculated output size is too small")
        output = torch.empty(
            (n, cout, hout, wout), device=input.device, dtype=input.dtype
        )
        opg = cout // groups
        reduction = cpg * kh * kw
        output_elements = n * cout * hout * wout
        use_spatial_tile = reduction >= 64 and output_elements >= 65536
        use_channels16 = use_spatial_tile and groups == 1 and cout % 16 == 0
        use_channels8 = use_spatial_tile and groups == 1 and cout % 8 == 0
        use_channels4 = use_spatial_tile and groups == 1 and cout % 4 == 0
        if use_channels16:
            block = 128
            _forward_spatial_channels16[
                (triton.cdiv(n * hout * wout, block), cout // 16)
            ](
                input,
                weight,
                bias,
                output,
                n,
                hin,
                win,
                cout,
                hout,
                wout,
                cpg,
                kh,
                kw,
                sh,
                sw,
                ph,
                pw,
                dh,
                dw,
                *input.stride(),
                *weight.stride(),
                *output.stride(),
                HAS_BIAS=bias is not None,
                CPG=cpg,
                KH=kh,
                KW=kw,
                BLOCK=block,
                num_warps=4,
            )
        elif use_channels8:
            block = 128
            _forward_spatial_channels8[
                (triton.cdiv(n * hout * wout, block), cout // 8)
            ](
                input,
                weight,
                bias,
                output,
                n,
                hin,
                win,
                cout,
                hout,
                wout,
                cpg,
                kh,
                kw,
                sh,
                sw,
                ph,
                pw,
                dh,
                dw,
                *input.stride(),
                *weight.stride(),
                *output.stride(),
                HAS_BIAS=bias is not None,
                CPG=cpg,
                KH=kh,
                KW=kw,
                BLOCK=block,
                num_warps=4,
            )
        elif use_channels4:
            block = 128
            _forward_spatial_channels4[
                (triton.cdiv(n * hout * wout, block), triton.cdiv(cout, 4))
            ](
                input,
                weight,
                bias,
                output,
                n,
                hin,
                win,
                cout,
                hout,
                wout,
                cpg,
                kh,
                kw,
                sh,
                sw,
                ph,
                pw,
                dh,
                dw,
                *input.stride(),
                *weight.stride(),
                *output.stride(),
                HAS_BIAS=bias is not None,
                CPG=cpg,
                KH=kh,
                KW=kw,
                BLOCK=block,
                num_warps=4,
            )
        else:
            block = 128 if use_spatial_tile else 64
            kernel = _forward_spatial_tile if use_spatial_tile else _forward
            grid = (
                (triton.cdiv(n * hout * wout, block), cout)
                if use_spatial_tile
                else (triton.cdiv(output_elements, block),)
            )
            kernel[grid](
                input,
                weight,
                bias,
                output,
                n,
                hin,
                win,
                cout,
                hout,
                wout,
                cpg,
                opg,
                kh,
                kw,
                sh,
                sw,
                ph,
                pw,
                dh,
                dw,
                *input.stride(),
                *weight.stride(),
                *output.stride(),
                HAS_BIAS=bias is not None,
                CPG=cpg,
                OPG=opg,
                KH=kh,
                KW=kw,
                BLOCK=block,
                num_warps=4,
            )
        ctx.save_for_backward(input, weight)
        ctx.args = (
            stride,
            padding,
            dilation,
            groups,
            ph,
            pw,
            hout,
            wout,
            bias is not None,
        )
        return output

    @staticmethod
    def backward(ctx, out_grad):
        input, weight = ctx.saved_tensors
        stride, padding, dilation, groups, ph, pw, hout, wout, has_bias = ctx.args
        sh, sw = _pair(stride, "stride")
        dh, dw = _pair(dilation, "dilation")
        n, cin, hin, win = input.shape
        cout, cpg, kh, kw = weight.shape
        need_x, need_w, need_b = ctx.needs_input_grad[:3]
        grad_x = grad_w = grad_b = None
        if need_x:
            grad_x = torch.empty_like(input)
            _input_grad[(triton.cdiv(n * hin * win, 32), cin)](
                out_grad,
                weight,
                grad_x,
                n,
                cin,
                hin,
                win,
                hout,
                wout,
                cpg,
                cout // groups,
                kh,
                kw,
                sh,
                sw,
                ph,
                pw,
                dh,
                dw,
                *out_grad.stride(),
                *weight.stride(),
                *grad_x.stride(),
                BM=32,
                KH=kh,
                KW=kw,
                OPG=cout // groups,
                CPG=cpg,
            )
        if need_w:
            grad_w = torch.empty_like(weight)
            _weight_grad[(weight.numel(),)](
                input,
                out_grad,
                grad_w,
                n,
                hin,
                win,
                cout,
                hout,
                wout,
                cpg,
                cout // groups,
                kh,
                kw,
                sh,
                sw,
                ph,
                pw,
                dh,
                dw,
                *input.stride(),
                *out_grad.stride(),
                *grad_w.stride(),
                BP=64,
                BK=64,
            )
        if has_bias and need_b:
            grad_b = torch.empty((cout,), device=out_grad.device, dtype=out_grad.dtype)
            _bias_grad[(cout,)](
                out_grad, grad_b, n, cout, hout, wout, *out_grad.stride(), BP=64, BO=64
            )
        return grad_x, grad_w, grad_b, None, None, None, None


def _d2(v):
    return v if isinstance(v, (tuple, list)) else (v, v)


_VENDOR_CACHE = {}
_VENDOR_BLOCKED = set()
_VENDOR_LOCK = threading.Lock()
_VENDOR_MAX_KEYS = 64


def _vendor_launch(
    input,
    weight,
    bias,
    n,
    hin,
    win,
    cout,
    hout,
    wout,
    cpg,
    kh,
    kw,
    sh,
    sw,
    ph,
    pw,
    dh,
    dw,
    groups,
):
    """One plain launch of the forward kernel."""
    from .conv_depthwise2d import conv2d_forward_kernel

    out = torch.empty((n, cout, hout, wout), device=input.device, dtype=input.dtype)
    conv2d_forward_kernel[(triton.cdiv(n * cout * hout * wout, 64),)](
        input,
        weight,
        out,
        bias,
        n,
        hin,
        win,
        cout,
        hout,
        wout,
        *input.stride(),
        *weight.stride(),
        *out.stride(),
        cpg,
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        groups,
        cout // groups,
        HAS_BIAS=True,
        BLOCK=64,
        num_warps=4,
    )
    return out


def _raw_stream(dev_index):
    """Current-stream query without building a Stream object."""
    try:
        return torch._C._cuda_getCurrentRawStream(dev_index)
    except Exception:
        return torch.cuda.current_stream().cuda_stream


def _vendor_replay(rec, input, weight, bias):
    """Replay the captured launcher call with fresh tensors spliced in."""
    if _raw_stream(rec["dev"]) != rec["stream"]:
        raise RuntimeError("stream changed")
    a = list(rec["a2"])
    out = torch.empty(rec["out_shape"], device=input.device, dtype=rec["out_dtype"])
    a[rec["j_in"]] = input
    a[rec["j_w"]] = weight
    a[rec["j_out"]] = out
    if rec["jb"] is not None:
        a[rec["jb"]] = bias
    rec["launcher"].launch(*a)
    return out


def _vendor_conv2d_forward(input, weight, bias, stride, padding, dilation, groups):
    """Forward-only fast path; returns None when the call is not eligible.

    Only forward-only (no-grad) calls take this path: it is not part of the
    autograd graph, so a call that needs a backward stays on `Conv2d.apply`.
    """
    if input.ndim != 4 or weight.ndim != 4:
        return None
    if not (input.is_cuda and weight.is_cuda):
        return None
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return None
    if input.dtype != weight.dtype:
        return None
    if input.requires_grad or weight.requires_grad:
        return None
    if not (input.is_contiguous() and weight.is_contiguous()):
        return None
    # The bias slot is read as fp32; a None bias becomes a cached fp32 zero
    # tensor so the slot is never dropped from the signature.
    if bias is not None and (
        bias.dtype != torch.float32 or bias.ndim != 1 or not bias.is_contiguous()
    ):
        return None
    n, cin, hin, win = input.shape
    cout, cpg, kh, kw = weight.shape
    if groups <= 0 or cout % groups or cpg * groups != cin:
        return None
    if min(hin, win) <= 1 and input.dtype == torch.bfloat16:
        # bf16 with a unit spatial extent is not supported on this path
        return None
    sh, sw = _pair(stride, "stride")
    dh, dw = _pair(dilation, "dilation")
    if min(sh, sw, dh, dw) <= 0:
        return None
    if isinstance(padding, str):
        if padding == "valid":
            ph = pw = 0
        elif padding == "same":
            if sh != 1 or sw != 1:
                return None
            th, tw = dh * (kh - 1), dw * (kw - 1)
            if th % 2 or tw % 2:
                # an odd total pad needs an asymmetric pad, which is not
                # expressible here
                return None
            ph, pw = th // 2, tw // 2
        else:
            return None
    else:
        ph, pw = _pair(padding, "padding")
    if min(ph, pw) < 0:
        return None
    hout = (hin + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    wout = (win + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    if hout <= 0 or wout <= 0:
        return None

    # imported lazily: conv_depthwise2d imports conv2d at module scope
    from .conv_depthwise2d import _capture_and_delegate, _record, _verify, _zero_bias

    if bias is None:
        bias = _zero_bias(cout, input.device)
    launch = (
        input,
        weight,
        bias,
        n,
        hin,
        win,
        cout,
        hout,
        wout,
        cpg,
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        groups,
    )
    key = (
        tuple(input.shape),
        tuple(input.stride()),
        input.dtype,
        tuple(weight.shape),
        tuple(weight.stride()),
        input.device.index,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        groups,
    )

    rec = _VENDOR_CACHE.get(key)
    if rec is not None:
        try:
            out = _vendor_replay(rec, input, weight, bias)
            if rec["verified"] or _verify(rec, out):
                return out
        except Exception:
            pass
        with _VENDOR_LOCK:
            _VENDOR_CACHE.pop(key, None)
            if len(_VENDOR_BLOCKED) < 256:
                _VENDOR_BLOCKED.add(key)
        return _vendor_launch(*launch)

    if key in _VENDOR_BLOCKED:
        return _vendor_launch(*launch)

    with _VENDOR_LOCK:
        if len(_VENDOR_CACHE) >= _VENDOR_MAX_KEYS:
            return _vendor_launch(*launch)
        out, cap = _capture_and_delegate(lambda: _vendor_launch(*launch))
        rec2 = _record(out, cap, input, weight) if cap else None
        if rec2 is None:
            if len(_VENDOR_BLOCKED) < 256:
                _VENDOR_BLOCKED.add(key)
            return out
        rec2["dev"] = input.device.index
        rec2["stream"] = _raw_stream(input.device.index)
        _VENDOR_CACHE[key] = rec2
    return out


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    vendor_out = _vendor_conv2d_forward(
        input, weight, bias, stride, padding, dilation, groups
    )
    if vendor_out is not None:
        logger.debug("GEMS_KUNLUNXIN CONV2D (fast path)")
        return vendor_out
    logger.debug("GEMS_KUNLUNXIN CONV2D")
    return Conv2d.apply(input, weight, bias, stride, padding, dilation, groups)

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


# ---------------------------------------------------------------------------
# Forward (non-transposed) implicit-GEMM convolution, flat-k form.
#
#   out[n, co, oh, ow] = bias[co] + sum_{ci, kh, kw}
#       X[n, g*CIN_G + ci, oh*SH - PH + kh*DH, ow*SW - PW + kw*DW] * W[co, ci, kh, kw]
#
# The dot is oriented as  acc[BLOCK_C, BLOCK_W] = W[BLOCK_C, BLOCK_K] @ X[BLOCK_K, BLOCK_W]
# so that the fastest-varying axis of both operands is a unit-stride axis:
#   * W rows are contiguous along k=(ci,kh,kw) for a contiguous weight,
#   * X rows are contiguous along ow (output width has unit stride),
#   * the store is contiguous along ow.
# Grid: (cdiv(WOUT, BLOCK_W), groups * cdiv(COUT_G, BLOCK_C), N * HOUT)
# ---------------------------------------------------------------------------
@triton.jit
def _conv2d_fwd_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    o_ptr,
    K,
    CIN_G,
    COUT_G,
    HIN,
    WIN,
    HOUT,
    WOUT,
    SH,
    SW,
    PH,
    PW,
    DH,
    DW,
    XN,
    XC,
    XH,
    XW,
    WSC,
    WSI,
    WSH,
    WSW,
    BS,
    ON,
    OC,
    HAS_BIAS: tl.constexpr,
    WCONTIG: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)

    n = pid_h // HOUT
    oh = pid_h % HOUT

    ng = (COUT_G + BLOCK_C - 1) // BLOCK_C
    g = pid_c // ng
    offs_c = (pid_c - g * ng) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < COUT_G

    ow = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_w = ow < WOUT

    co = g * COUT_G + offs_c
    koff = tl.arange(0, BLOCK_K)

    wbase = w_ptr + co * WSC
    xrow = x_ptr + n * XN + (g * CIN_G) * XC

    acc = tl.zeros((BLOCK_C, BLOCK_W), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + koff
        mask_k = offs_k < K
        ci = offs_k // (KH * KW)
        rem_k = offs_k % (KH * KW)
        kh = rem_k // KW
        kw = rem_k % KW
        if WCONTIG:
            aptr = wbase[:, None] + offs_k[None, :]
        else:
            aptr = wbase[:, None] + (ci * WSI + kh * WSH + kw * WSW)[None, :]
        wt = tl.load(aptr, mask=mask_c[:, None] & mask_k[None, :], other=0.0)

        ih = oh * SH - PH + kh * DH
        iw = ow[None, :] * SW - PW + kw[:, None] * DW
        xmask = (
            mask_k[:, None]
            & mask_w[None, :]
            & (ih >= 0)[:, None]
            & (ih < HIN)[:, None]
            & (iw >= 0)
            & (iw < WIN)
        )
        xptr = xrow + (ci * XC + ih * XH)[:, None] + iw * XW
        xt = tl.load(xptr, mask=xmask, other=0.0)
        acc = tl.dot(wt, xt, acc, input_precision="ieee")

    if HAS_BIAS:
        bv = tl.load(b_ptr + co * BS, mask=mask_c, other=0.0).to(tl.float32)
        acc = acc + bv[:, None]

    optr = o_ptr + co[:, None] * OC + n * ON + oh * WOUT + ow[None, :]
    tl.store(
        optr, acc.to(o_ptr.dtype.element_ty), mask=mask_c[:, None] & mask_w[None, :]
    )


# ---------------------------------------------------------------------------
# Large-problem forward kernel: one flat pipelined loop over q = (tap, ci block)
# with the tap decomposed as uniform scalars, so the whole boundary test is a
# 1-D W mask plus a scalar row test.  It consumes a weight pre-permuted to
# Wp[co][kh][kw][ci] so the dot-K axis (ci) is unit-stride.
# Grid: (cdiv(WOUT, BLOCK_W), groups * cdiv(COUT_G, BLOCK_C), N * HOUT)
# ---------------------------------------------------------------------------
_TAP_CONFIGS = [
    triton.Config({"BLOCK_W": 128, "BLOCK_KC": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_W": 128, "BLOCK_KC": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_W": 128, "BLOCK_KC": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_W": 128, "BLOCK_KC": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_W": 128, "BLOCK_KC": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_W": 128, "BLOCK_KC": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_W": 128, "BLOCK_KC": 16}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_W": 64, "BLOCK_KC": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_W": 64, "BLOCK_KC": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_W": 64, "BLOCK_KC": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_W": 64, "BLOCK_KC": 32}, num_warps=2, num_stages=3),
    triton.Config({"BLOCK_W": 32, "BLOCK_KC": 64}, num_warps=2, num_stages=2),
]


@triton.autotune(
    configs=_TAP_CONFIGS,
    key=[
        "DTYPE_KEY",
        "CIN_G",
        "COUT_G",
        "HIN",
        "WIN",
        "HOUT",
        "WOUT",
        "SH",
        "SW",
        "KH",
        "KW",
    ],
)
@triton.jit
def _conv2d_tap_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    o_ptr,
    DTYPE_KEY,
    CIN_G,
    COUT_G,
    HIN,
    WIN,
    HOUT,
    WOUT,
    SH,
    SW,
    PH,
    PW,
    DH,
    DW,
    XN,
    XC,
    XH,
    XW,
    WSC,
    BS,
    ON,
    OC,
    HAS_BIAS: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_KC: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)

    n = pid_h // HOUT
    oh = pid_h % HOUT

    ng = (COUT_G + BLOCK_C - 1) // BLOCK_C
    g = pid_c // ng
    offs_c = (pid_c - g * ng) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < COUT_G

    ow = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    co = g * COUT_G + offs_c
    ci_off = tl.arange(0, BLOCK_KC)

    wbase = w_ptr + co * WSC
    xrow = x_ptr + n * XN + (g * CIN_G) * XC

    cblocks = (CIN_G + BLOCK_KC - 1) // BLOCK_KC
    nq = KH * KW * cblocks

    acc = tl.zeros((BLOCK_C, BLOCK_W), dtype=tl.float32)
    for q in range(0, nq):
        tap = q // cblocks
        c0 = (q - tap * cblocks) * BLOCK_KC
        kh = tap // KW
        kw = tap - kh * KW
        ci = c0 + ci_off
        cm = ci < CIN_G
        # keep the X address in range for padded tail channels; the weight row is
        # zeroed by cm so the extra lanes contribute nothing
        cis = tl.where(cm, ci, 0)
        # permuted weight: contiguous along ci
        aptr = wbase[:, None] + (tap * CIN_G + ci)[None, :]
        ih = oh * SH - PH + kh * DH
        iw = ow * SW + (kw * DW - PW)
        wmask = (ih >= 0) & (ih < HIN) & (iw >= 0) & (iw < WIN)
        xptr = xrow + (cis * XC)[:, None] + ih * XH + iw * XW
        wt = tl.load(aptr, mask=mask_c[:, None] & cm[None, :], other=0.0)
        xt = tl.load(xptr, mask=wmask[None, :], other=0.0)
        acc = tl.dot(wt, xt, acc, input_precision="ieee")

    if HAS_BIAS:
        bv = tl.load(b_ptr + co * BS, mask=mask_c, other=0.0).to(tl.float32)
        acc = acc + bv[:, None]

    optr = o_ptr + co[:, None] * OC + n * ON + oh * WOUT + ow[None, :]
    tl.store(
        optr,
        acc.to(o_ptr.dtype.element_ty),
        mask=mask_c[:, None] & (ow < WOUT)[None, :],
    )


# ---------------------------------------------------------------------------
# Large-problem forward kernel with output-row grouping.
#
# One flat pipelined loop over q = (input row offset j, kw, ci block), with all of
# j, kw and the ci block recovered as uniform scalars.  The X tile loaded for a
# given (j, kw) input row is reused by every output row r whose kernel tap
# kh = j - r is in range, so BLOCK_H output rows cost (BLOCK_H + KH - 1) * KW
# input-row loads instead of BLOCK_H * KH * KW.  Requires unit row stride and
# unit row dilation (the dispatcher guarantees that); the weight is consumed in
# the pre-permuted Wp[co][kh][kw][ci] layout so ci is unit-stride.
# Grid: (cdiv(WOUT, BLOCK_W), groups * cdiv(COUT_G, BLOCK_C), N * cdiv(HOUT, BLOCK_H))
# ---------------------------------------------------------------------------
_ROW_CONFIGS = [
    triton.Config(
        {"BLOCK_W": 128, "BLOCK_KC": 64, "BLOCK_H": 1}, num_warps=4, num_stages=2
    ),
    triton.Config(
        {"BLOCK_W": 128, "BLOCK_KC": 64, "BLOCK_H": 1}, num_warps=8, num_stages=2
    ),
    triton.Config(
        {"BLOCK_W": 128, "BLOCK_KC": 32, "BLOCK_H": 1}, num_warps=4, num_stages=3
    ),
    triton.Config(
        {"BLOCK_W": 64, "BLOCK_KC": 64, "BLOCK_H": 1}, num_warps=4, num_stages=2
    ),
    triton.Config(
        {"BLOCK_W": 128, "BLOCK_KC": 64, "BLOCK_H": 2}, num_warps=4, num_stages=2
    ),
    triton.Config(
        {"BLOCK_W": 128, "BLOCK_KC": 64, "BLOCK_H": 2}, num_warps=8, num_stages=2
    ),
    triton.Config(
        {"BLOCK_W": 64, "BLOCK_KC": 64, "BLOCK_H": 2}, num_warps=4, num_stages=2
    ),
    triton.Config(
        {"BLOCK_W": 128, "BLOCK_KC": 32, "BLOCK_H": 2}, num_warps=4, num_stages=3
    ),
    triton.Config(
        {"BLOCK_W": 64, "BLOCK_KC": 64, "BLOCK_H": 4}, num_warps=4, num_stages=2
    ),
    triton.Config(
        {"BLOCK_W": 64, "BLOCK_KC": 64, "BLOCK_H": 4}, num_warps=4, num_stages=3
    ),
    triton.Config(
        {"BLOCK_W": 32, "BLOCK_KC": 64, "BLOCK_H": 4}, num_warps=4, num_stages=2
    ),
    triton.Config(
        {"BLOCK_W": 64, "BLOCK_KC": 32, "BLOCK_H": 4}, num_warps=4, num_stages=3
    ),
]


@triton.autotune(
    configs=_ROW_CONFIGS,
    key=[
        "DTYPE_KEY",
        "CIN_G",
        "COUT_G",
        "HIN",
        "WIN",
        "HOUT",
        "WOUT",
        "SW",
        "PH",
        "PW",
        "DW",
        "KH",
        "KW",
    ],
)
@triton.jit
def _conv2d_row_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    o_ptr,
    DTYPE_KEY,
    CIN_G,
    COUT_G,
    HIN,
    WIN,
    HOUT,
    WOUT,
    SW,
    PH,
    PW,
    DW,
    XN,
    XC,
    XH,
    XW,
    WSC,
    BS,
    ON,
    OC,
    HAS_BIAS: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_KC: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)

    nh = (HOUT + BLOCK_H - 1) // BLOCK_H
    n = pid_h // nh
    oh0 = (pid_h - n * nh) * BLOCK_H

    ng = (COUT_G + BLOCK_C - 1) // BLOCK_C
    g = pid_c // ng
    offs_c = (pid_c - g * ng) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < COUT_G

    ow = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    co = g * COUT_G + offs_c
    ci_off = tl.arange(0, BLOCK_KC)

    wbase = w_ptr + co * WSC
    xrow = x_ptr + n * XN + (g * CIN_G) * XC

    cblocks = (CIN_G + BLOCK_KC - 1) // BLOCK_KC
    per_j = KW * cblocks
    nq = (BLOCK_H + KH - 1) * per_j
    ih0 = oh0 - PH

    acc0 = tl.zeros((BLOCK_C, BLOCK_W), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_C, BLOCK_W), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_C, BLOCK_W), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_C, BLOCK_W), dtype=tl.float32)
    for q in range(0, nq):
        j = q // per_j
        rem = q - j * per_j
        kw = rem // cblocks
        c0 = (rem - kw * cblocks) * BLOCK_KC
        ih = ih0 + j
        hok = (ih >= 0) & (ih < HIN)
        ci = c0 + ci_off
        cm = ci < CIN_G
        cis = tl.where(cm, ci, 0)
        iw = ow * SW + (kw * DW - PW)
        wmask = hok & (iw >= 0) & (iw < WIN)
        xt = tl.load(
            xrow + (cis * XC)[:, None] + ih * XH + iw * XW,
            mask=wmask[None, :],
            other=0.0,
        )
        if j < KH:
            wt = tl.load(
                wbase[:, None] + ((j * KW + kw) * CIN_G + ci)[None, :],
                mask=mask_c[:, None] & cm[None, :],
                other=0.0,
            )
            acc0 = tl.dot(wt, xt, acc0, input_precision="ieee")
        if BLOCK_H > 1:
            if (j >= 1) & (j < KH + 1):
                wt = tl.load(
                    wbase[:, None] + (((j - 1) * KW + kw) * CIN_G + ci)[None, :],
                    mask=mask_c[:, None] & cm[None, :],
                    other=0.0,
                )
                acc1 = tl.dot(wt, xt, acc1, input_precision="ieee")
        if BLOCK_H > 2:
            if (j >= 2) & (j < KH + 2):
                wt = tl.load(
                    wbase[:, None] + (((j - 2) * KW + kw) * CIN_G + ci)[None, :],
                    mask=mask_c[:, None] & cm[None, :],
                    other=0.0,
                )
                acc2 = tl.dot(wt, xt, acc2, input_precision="ieee")
            if (j >= 3) & (j < KH + 3):
                wt = tl.load(
                    wbase[:, None] + (((j - 3) * KW + kw) * CIN_G + ci)[None, :],
                    mask=mask_c[:, None] & cm[None, :],
                    other=0.0,
                )
                acc3 = tl.dot(wt, xt, acc3, input_precision="ieee")

    if HAS_BIAS:
        bv = tl.load(b_ptr + co * BS, mask=mask_c, other=0.0).to(tl.float32)
        acc0 = acc0 + bv[:, None]
        acc1 = acc1 + bv[:, None]
        acc2 = acc2 + bv[:, None]
        acc3 = acc3 + bv[:, None]
    base = o_ptr + co[:, None] * OC + n * ON + ow[None, :]
    smask = mask_c[:, None] & (ow < WOUT)[None, :]
    tl.store(base + oh0 * WOUT, acc0.to(o_ptr.dtype.element_ty), mask=smask)
    if BLOCK_H > 1:
        tl.store(
            base + (oh0 + 1) * WOUT,
            acc1.to(o_ptr.dtype.element_ty),
            mask=smask & (oh0 + 1 < HOUT),
        )
    if BLOCK_H > 2:
        tl.store(
            base + (oh0 + 2) * WOUT,
            acc2.to(o_ptr.dtype.element_ty),
            mask=smask & (oh0 + 2 < HOUT),
        )
        tl.store(
            base + (oh0 + 3) * WOUT,
            acc3.to(o_ptr.dtype.element_ty),
            mask=smask & (oh0 + 3 < HOUT),
        )


@triton.jit
def _weight_permute_kernel(
    w_ptr,
    wp_ptr,
    NUM,
    CIN_G,
    WSC,
    WSI,
    WSH,
    WSW,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < NUM
    ci = idx % CIN_G
    t = idx // CIN_G
    kw = t % KW
    t2 = t // KW
    kh = t2 % KH
    co = t2 // KH
    src = co * WSC + ci * WSI + kh * WSH + kw * WSW
    tl.store(wp_ptr + idx, tl.load(w_ptr + src, mask=mask, other=0.0), mask=mask)


# ---------------------------------------------------------------------------
# Transposed convolution (gather form).
#   out[n, co, oh, ow] = bias[co] + sum_{ci, kh, kw}
#       [nh % SH == 0 and nw % SW == 0 and in range]
#       * X[n, g*CIN_G + ci, nh/SH, nw/SW] * W[g*CIN_G + ci, co - g*COUT_G, kh, kw]
#   nh = oh + PH - kh*DH,  nw = ow + PW - kw*DW
# Grid: (cdiv(N*HOUT*WOUT, BLOCK_M), cdiv(COUT_G, BLOCK_N), groups)
# ---------------------------------------------------------------------------
@triton.jit
def _conv2d_transposed_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    o_ptr,
    M,
    K,
    CIN_G,
    COUT_G,
    HIN,
    WIN,
    HOUT,
    WOUT,
    SH,
    SW,
    PH,
    PW,
    DH,
    DW,
    XN,
    XC,
    XH,
    XW,
    WSI,
    WSC,
    WSH,
    WSW,
    BS,
    ON,
    OC,
    HAS_BIAS: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_g = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    hw = HOUT * WOUT
    n_idx = offs_m // hw
    rem = offs_m % hw
    oh = rem // WOUT
    ow = rem % WOUT
    mask_m = offs_m < M

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < COUT_G

    cin_base = pid_g * CIN_G
    xbase = x_ptr + n_idx * XN + cin_base * XC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        ci = offs_k // (KH * KW)
        rem_k = offs_k % (KH * KW)
        kh = rem_k // KW
        kw = rem_k % KW
        nh = oh[:, None] + PH - kh[None, :] * DH
        nw = ow[:, None] + PW - kw[None, :] * DW
        vh = (nh >= 0) & (nh < HIN * SH) & (nh % SH == 0)
        vw = (nw >= 0) & (nw < WIN * SW) & (nw % SW == 0)
        ih = nh // SH
        iw = nw // SW
        amask = mask_m[:, None] & mask_k[None, :] & vh & vw
        aptr = xbase[:, None] + ci[None, :] * XC + ih * XH + iw * XW
        a = tl.load(aptr, mask=amask, other=0.0)
        bptr = (
            w_ptr
            + (cin_base + ci)[:, None] * WSI
            + offs_n[None, :] * WSC
            + kh[:, None] * WSH
            + kw[:, None] * WSW
        )
        b = tl.load(bptr, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc = tl.dot(a, b, acc, input_precision="ieee")

    if HAS_BIAS:
        bv = tl.load(b_ptr + (pid_g * COUT_G + offs_n) * BS, mask=mask_n, other=0.0)
        acc = acc + bv.to(tl.float32)[None, :]

    op = (
        o_ptr
        + n_idx[:, None] * ON
        + (pid_g * COUT_G + offs_n)[None, :] * OC
        + (oh * WOUT + ow)[:, None]
    )
    tl.store(op, acc.to(o_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


_NUM_WARPS = 4
_NUM_STAGES = 2
# Above this many multiply-accumulates the extra weight-permute launch is
# amortised by the cheaper per-tap kernel; below it the flat-k kernel wins.
_PERMUTE_MIN_MACS = 1 << 27


def _pair(v):
    if isinstance(v, (list, tuple)):
        return int(v[0]), int(v[1]) if len(v) > 1 else int(v[0])
    return int(v), int(v)


def convolution_overrideable(
    input, weight, bias, stride, padding, dilation, transposed, output_padding, groups
):
    logger.debug("GEMS_HYGON CONVOLUTION_OVERRIDEABLE")
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    dh, dw = _pair(dilation)

    n, cin, hin, win = input.shape
    kh = weight.shape[2]
    kw = weight.shape[3]

    if transposed:
        oph, opw = _pair(output_padding)
        cin_g = weight.shape[0] // groups
        cout_g = weight.shape[1]
        cout = cout_g * groups
        hout = (hin - 1) * sh - 2 * ph + dh * (kh - 1) + oph + 1
        wout = (win - 1) * sw - 2 * pw + dw * (kw - 1) + opw + 1
    else:
        cin_g = weight.shape[1]
        cout_g = weight.shape[0] // groups
        cout = cout_g * groups
        hout = (hin + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        wout = (win + 2 * pw - dw * (kw - 1) - 1) // sw + 1

    if groups < 1 or cin % groups or cout % groups:
        raise RuntimeError(
            "convolution_overrideable: expected input channels and output channels "
            "to be divisible by groups, got groups=%d" % groups
        )

    bptr = bias if bias is not None else input
    bs = bias.stride(0) if bias is not None else 1
    out = torch.empty((n, cout, hout, wout), device=input.device, dtype=input.dtype)

    if transposed:
        m = n * hout * wout
        k = cin_g * kh * kw
        bm = 64
        grid = ((m + bm - 1) // bm, (cout_g + 31) // 32, groups)
        _conv2d_transposed_kernel[grid](
            input,
            weight,
            bptr,
            out,
            m,
            k,
            cin_g,
            cout_g,
            hin,
            win,
            hout,
            wout,
            sh,
            sw,
            ph,
            pw,
            dh,
            dw,
            *input.stride(),
            *weight.stride(),
            bs,
            cout * hout * wout,
            hout * wout,
            HAS_BIAS=bias is not None,
            KH=kh,
            KW=kw,
            BLOCK_M=bm,
            BLOCK_N=32,
            BLOCK_K=64,
            num_warps=_NUM_WARPS,
            num_stages=_NUM_STAGES,
        )
        return out

    bw = 16
    while bw < wout and bw < 128:
        bw = bw << 1
    bc = 32 if cout_g <= 32 else 64
    grid_c = groups * ((cout_g + bc - 1) // bc)

    if n * hout * wout * cout_g * cin_g * kh * kw >= _PERMUTE_MIN_MACS:
        num = cout * kh * kw * cin_g
        wp = torch.empty((cout, kh, kw, cin_g), device=input.device, dtype=weight.dtype)
        _weight_permute_kernel[((num + 255) // 256,)](
            weight,
            wp,
            num,
            cin_g,
            *weight.stride(),
            KH=kh,
            KW=kw,
            BLOCK=256,
            num_warps=4,
        )
        if sh == 1 and dh == 1:
            _conv2d_row_kernel[
                lambda META: (
                    (wout + META["BLOCK_W"] - 1) // META["BLOCK_W"],
                    grid_c,
                    n * ((hout + META["BLOCK_H"] - 1) // META["BLOCK_H"]),
                )
            ](
                input,
                wp,
                bptr,
                out,
                input.element_size(),
                cin_g,
                cout_g,
                hin,
                win,
                hout,
                wout,
                sw,
                ph,
                pw,
                dw,
                *input.stride(),
                kh * kw * cin_g,
                bs,
                cout * hout * wout,
                hout * wout,
                HAS_BIAS=bias is not None,
                KH=kh,
                KW=kw,
                BLOCK_C=bc,
            )
            return out
        _conv2d_tap_kernel[
            lambda META: (
                (wout + META["BLOCK_W"] - 1) // META["BLOCK_W"],
                grid_c,
                n * hout,
            )
        ](
            input,
            wp,
            bptr,
            out,
            input.element_size(),
            cin_g,
            cout_g,
            hin,
            win,
            hout,
            wout,
            sh,
            sw,
            ph,
            pw,
            dh,
            dw,
            *input.stride(),
            kh * kw * cin_g,
            bs,
            cout * hout * wout,
            hout * wout,
            HAS_BIAS=bias is not None,
            KH=kh,
            KW=kw,
            BLOCK_C=bc,
        )
        return out

    k = cin_g * kh * kw
    bk = 16 if k <= 16 else (32 if k <= 32 else 64)
    _conv2d_fwd_kernel[((wout + bw - 1) // bw, grid_c, n * hout)](
        input,
        weight,
        bptr,
        out,
        k,
        cin_g,
        cout_g,
        hin,
        win,
        hout,
        wout,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        *input.stride(),
        *weight.stride(),
        bs,
        cout * hout * wout,
        hout * wout,
        HAS_BIAS=bias is not None,
        WCONTIG=weight.is_contiguous(),
        KH=kh,
        KW=kw,
        BLOCK_C=bc,
        BLOCK_W=bw,
        BLOCK_K=bk,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    return out

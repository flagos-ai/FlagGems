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

"""Triton implementation of ``aten::convolution_overrideable.out``.

Architecture
------------
* ``_conv2d_fwd_kernel``  implicit-GEMM (im2col) forward convolution.
  grid = (ceil(OH*OW / BLOCK_M), ceil(Cout/groups / BLOCK_N), N*groups).
  Each program owns a (BLOCK_M, BLOCK_N) output tile, loops over the KH*KW
  taps and the BLOCK_K channel blocks and accumulates with ``tl.dot``.
* ``_conv2d_t_kernel``    gather (no atomics) transposed convolution with the
  same tiling; a tap contributes only when its source numerator is
  non-negative, exactly divisible by the stride and inside the input.
* 1D convolutions are promoted to 2D through zero-copy ``unsqueeze`` views.
* Accumulation is IEEE float32 (``input_precision="ieee"`` for fp32 operands,
  fp32 accumulate for fp16/bf16); the cast to the output dtype happens once, at
  the store.

The host wrapper validates dtype/device, resizes an empty ``out``, writes
through ``out.stride()`` (non-contiguous / channels-last outputs are written in
place) and returns the very same ``out`` object.
"""

import logging
import warnings

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


@triton.jit
def _conv2d_fwd_kernel(
    in_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,
    NSTRIDE,
    CIN_STRIDE,
    HSTRIDE,
    WSTRIDE,
    WN,
    WC,
    WKH,
    WKW,
    ON,
    OC,
    OHS,
    OWS,
    N_POS,
    IN_H,
    IN_W,
    OUT_CG,
    OUT_W,
    SH,
    SW,
    PH,
    PW,
    DH,
    DW,
    CIG: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    GROUPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IEEE: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    g = pid_b % GROUPS
    n = pid_b // GROUPS

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < N_POS
    oh = offs_m // OUT_W
    ow = offs_m - oh * OUT_W

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < OUT_CG

    in_base = in_ptr + n.to(tl.int64) * NSTRIDE + (g * CIG).to(tl.int64) * CIN_STRIDE
    w_base = w_ptr + (g * OUT_CG + offs_n) * WN

    ih0 = oh * SH - PH
    iw0 = ow * SW - PW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for t in range(KH * KW):
        kh = t // KW
        kw = t - kh * KW
        ih = ih0 + kh * DH
        iw = iw0 + kw * DW
        am = m_mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
        in_tile = in_base + ih * HSTRIDE + iw * WSTRIDE
        w_tile = w_base + (kh * WKH + kw * WKW)
        for k0 in tl.static_range(0, CIG, BLOCK_K):
            ks = k0 + tl.arange(0, BLOCK_K)
            k_mask = ks < CIG
            a = tl.load(
                in_tile[:, None] + (ks * CIN_STRIDE)[None, :],
                mask=am[:, None] & k_mask[None, :],
                other=0.0,
            )
            b = tl.load(
                w_tile[None, :] + (ks * WC)[:, None],
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            if SPLIT:
                # 3-term bf16 split: hi*hi + hi*lo + lo*hi (fp32 accumulate,
                # lo*lo dropped).  Moves the fp32 contraction onto matrix cores.
                a_hi = a.to(tl.bfloat16)
                b_hi = b.to(tl.bfloat16)
                a_lo = (a - a_hi.to(tl.float32)).to(tl.bfloat16)
                b_lo = (b - b_hi.to(tl.float32)).to(tl.bfloat16)
                acc = tl.dot(a_hi, b_hi, acc)
                acc = tl.dot(a_hi, b_lo, acc)
                acc = tl.dot(a_lo, b_hi, acc)
            elif IEEE:
                acc = tl.dot(a, b, acc, input_precision="ieee")
            else:
                acc = tl.dot(a, b, acc)

    oc = g * OUT_CG + offs_n
    if HAS_BIAS:
        bias_v = tl.load(bias_ptr + oc, mask=n_mask, other=0.0).to(tl.float32)
        acc = acc + bias_v[None, :]

    o_ptr = (
        out_ptr
        + n.to(tl.int64) * ON
        + (oc * OC)[None, :]
        + (oh * OHS + ow * OWS)[:, None]
    )
    tl.store(
        o_ptr,
        acc.to(out_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


@triton.jit
def _conv2d_t_kernel(
    in_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,
    NSTRIDE,
    CIN_STRIDE,
    HSTRIDE,
    WSTRIDE,
    WCI,
    WCO,
    WKH,
    WKW,
    ON,
    OC,
    OHS,
    OWS,
    N_POS,
    IN_H,
    IN_W,
    OUT_CG,
    OUT_W,
    SH,
    SW,
    PH,
    PW,
    DH,
    DW,
    CIG: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    GROUPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IEEE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    g = pid_b % GROUPS
    n = pid_b // GROUPS

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < N_POS
    oh = offs_m // OUT_W
    ow = offs_m - oh * OUT_W

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < OUT_CG

    in_base = in_ptr + n.to(tl.int64) * NSTRIDE + (g * CIG).to(tl.int64) * CIN_STRIDE
    w_base = w_ptr + (g * CIG).to(tl.int64) * WCI + (offs_n * WCO)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for t in range(KH * KW):
        kh = t // KW
        kw = t - kh * KW
        nh = oh + PH - kh * DH
        nw = ow + PW - kw * DW
        pos = (nh >= 0) & (nw >= 0)
        ih = tl.where(pos, nh // SH, 0)
        iw = tl.where(pos, nw // SW, 0)
        am = (
            m_mask
            & pos
            & (nh - ih * SH == 0)
            & (nw - iw * SW == 0)
            & (ih < IN_H)
            & (iw < IN_W)
        )
        a_ptr = in_base + ih * HSTRIDE + iw * WSTRIDE
        w_tile = w_base + (kh * WKH + kw * WKW)
        for k0 in tl.static_range(0, CIG, BLOCK_K):
            ks = k0 + tl.arange(0, BLOCK_K)
            k_mask = ks < CIG
            a = tl.load(
                a_ptr[:, None] + (ks * CIN_STRIDE)[None, :],
                mask=am[:, None] & k_mask[None, :],
                other=0.0,
            )
            b = tl.load(
                w_tile[None, :] + (ks * WCI)[:, None],
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            if IEEE:
                acc = tl.dot(a, b, acc, input_precision="ieee")
            else:
                acc = tl.dot(a, b, acc)

    oc = g * OUT_CG + offs_n
    if HAS_BIAS:
        bias_v = tl.load(bias_ptr + oc, mask=n_mask, other=0.0).to(tl.float32)
        acc = acc + bias_v[None, :]

    o_ptr = (
        out_ptr
        + n.to(tl.int64) * ON
        + (oc * OC)[None, :]
        + (oh * OHS + ow * OWS)[:, None]
    )
    tl.store(
        o_ptr,
        acc.to(out_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


@triton.jit
def _nchw_to_nhwc_kernel(
    in_ptr,
    out_ptr,
    N,
    C,
    H,
    W,
    sN,
    sC,
    sH,
    sW,
    C_CONST: tl.constexpr,
    CONTIG: tl.constexpr,
    BW: tl.constexpr,
    CB: tl.constexpr,
):
    """Per-call NCHW -> NHWC copy so the conv gathers land on a contiguous
    channel axis (vectorized 16-byte loads instead of scalar 2-byte loads)."""
    pid_w = tl.program_id(0)
    h = tl.program_id(1)
    n = tl.program_id(2)
    ws = pid_w * BW + tl.arange(0, BW)
    wm = ws < W
    for c0 in range(0, C, CB):
        cs = c0 + tl.arange(0, CB)
        cm = cs < C
        m = cm[:, None] & wm[None, :]
        base = in_ptr + n * sN + h * sH
        if CONTIG:
            src = base + cs[:, None] * sC + ws[None, :]
        else:
            src = base + cs[:, None] * sC + ws[None, :] * sW
        v = tl.load(src, mask=m, other=0.0)
        o_base = out_ptr + ((n * H + h) * W) * C_CONST
        dst = o_base + ws[None, :] * C_CONST + cs[:, None]
        tl.store(dst, v, mask=m)


@triton.jit
def _weight_to_tap_kernel(
    w_ptr,
    out_ptr,
    NEL,
    CIG,
    COUT,
    COG,
    KW,
    WN,
    WC,
    WKH,
    WKW,
    BLOCK: tl.constexpr,
):
    """weight [Cout, Cig, KH, KW] -> [tap, Cout, Cig] (cin contiguous)."""
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = idx < NEL
    cin = idx % CIG
    t1 = idx // CIG
    oc = t1 % COUT
    tap = t1 // COUT
    kh = tap // KW
    kw = tap - kh * KW
    g = oc // COG
    src = w_ptr + oc * WN + (g * CIG + cin) * WC + kh * WKH + kw * WKW
    v = tl.load(src, mask=m, other=0.0)
    tl.store(out_ptr + idx, v, mask=m)


_NHWC_CONFIGS = [
    # All configurations are shared-memory safe for the fp32 operands as well:
    # (BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N) * 4 bytes stays under 64 KiB.
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64}, num_warps=4, num_stages=2
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64}, num_warps=4, num_stages=2
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64}, num_warps=8, num_stages=2
    ),
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=8, num_stages=2
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=8, num_stages=3
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=3
    ),
]


@triton.autotune(
    configs=_NHWC_CONFIGS,
    key=[
        "N_POS",
        "IN_H",
        "IN_W",
        "OUT_CG",
        "CIG",
        "KH",
        "KW",
        "SH",
        "SW",
        "PH",
        "PW",
        "DH",
        "DW",
        "GROUPS",
        "OUT_W",
        "DTYPE_KEY",
    ],
)
@triton.jit
def _conv2d_fwd_nhwc_kernel(
    in_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,
    NSTRIDE,
    WSTRIDE,
    W_TAP,
    W_OC,
    ON,
    OC,
    OHS,
    OWS,
    N_POS,
    IN_H,
    IN_W,
    OUT_CG,
    OUT_W,
    SH,
    SW,
    PH,
    PW,
    DH,
    DW,
    CIG: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    GROUPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SPLIT: tl.constexpr,
    IEEE: tl.constexpr,
    DTYPE_KEY: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    g = pid_b % GROUPS
    n = pid_b // GROUPS

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < N_POS
    oh = offs_m // OUT_W
    ow = offs_m - oh * OUT_W

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < OUT_CG

    in_base = in_ptr + n.to(tl.int64) * NSTRIDE + g * CIG
    w_base = w_ptr + (g * OUT_CG + offs_n) * W_OC

    ih0 = oh * SH - PH
    iw0 = ow * SW - PW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    ks_full = tl.arange(0, BLOCK_K)

    for t in range(KH * KW):
        kh = t // KW
        kw = t - kh * KW
        ih = ih0 + kh * DH
        iw = iw0 + kw * DW
        am = m_mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
        in_tile = in_base + (ih * IN_W + iw) * WSTRIDE
        w_tile = w_base + t * W_TAP
        for k0 in tl.static_range(0, CIG, BLOCK_K):
            ks = k0 + ks_full
            k_mask = ks < CIG
            a = tl.load(
                in_tile[:, None] + ks[None, :],
                mask=am[:, None] & k_mask[None, :],
                other=0.0,
            )
            b = tl.load(
                w_tile[None, :] + ks[:, None],
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            if SPLIT:
                a_hi = a.to(tl.bfloat16)
                b_hi = b.to(tl.bfloat16)
                a_lo = (a - a_hi.to(tl.float32)).to(tl.bfloat16)
                b_lo = (b - b_hi.to(tl.float32)).to(tl.bfloat16)
                acc = tl.dot(a_hi, b_hi, acc)
                acc = tl.dot(a_hi, b_lo, acc)
                acc = tl.dot(a_lo, b_hi, acc)
            elif IEEE:
                acc = tl.dot(a, b, acc, input_precision="ieee")
            else:
                acc = tl.dot(a, b, acc)

    oc = g * OUT_CG + offs_n
    if HAS_BIAS:
        bias_v = tl.load(bias_ptr + oc, mask=n_mask, other=0.0).to(tl.float32)
        acc = acc + bias_v[None, :]

    o_ptr = (
        out_ptr
        + n.to(tl.int64) * ON
        + (oc * OC)[None, :]
        + (oh * OHS + ow * OWS)[:, None]
    )
    tl.store(
        o_ptr,
        acc.to(out_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


# ---------------------------------------------------------------------------
# host helpers
# ---------------------------------------------------------------------------


def _int_list(value, n):
    if isinstance(value, (list, tuple)):
        if len(value) == n:
            return [int(v) for v in value]
        if len(value) == 1:
            return [int(value[0])] * n
        return [int(v) for v in list(value)[:n]]
    return [int(value)] * n


def _resize_out(out, shape):
    if tuple(out.shape) == tuple(shape):
        return
    if out.numel() != 0:
        warnings.warn(
            "An output with one or more elements was resized since it had "
            f"shape {tuple(out.shape)}, which does not match the required "
            f"output shape {tuple(shape)}."
        )
    out.resize_(shape)


def _launch(
    inp,
    wt,
    bias,
    transposed,
    groups,
    sh,
    sw,
    ph,
    pw,
    dh,
    dw,
    oph,
    opw,
    kh,
    kw,
    out4,
):
    n_batch, cin, ih, iw = inp.shape
    if transposed:
        cin = wt.shape[0]
        cog = wt.shape[1]
        cig = cin // groups
        cout = cog * groups
        oh = (ih - 1) * sh - 2 * ph + dh * (kh - 1) + oph + 1
        ow = (iw - 1) * sw - 2 * pw + dw * (kw - 1) + opw + 1
    else:
        cout = wt.shape[0]
        cig = wt.shape[1]
        cog = cout // groups
        oh = (ih + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        ow = (iw + 2 * pw - dw * (kw - 1) - 1) // sw + 1

    n_pos = oh * ow
    if n_pos <= 0 or n_batch <= 0 or cog <= 0:
        return

    bn = max(16, min(64, triton.next_power_of_2(cog)))
    if n_pos >= 512:
        bm = 128
    elif n_pos >= 256:
        bm = 64
    else:
        bm = 32
    bm = max(16, min(bm, triton.next_power_of_2(n_pos)))
    bk = max(16, min(64, triton.next_power_of_2(cig)))
    num_warps = 8 if bm >= 128 else 4
    num_stages = 3 if inp.dtype.itemsize <= 2 else 2

    grid = (triton.cdiv(n_pos, bm), triton.cdiv(cog, bn), n_batch * groups)
    in_strides = inp.stride()
    w_strides = wt.stride()
    o_strides = out4.stride()
    bias_ptr = bias if bias is not None else wt
    ieee = inp.dtype == torch.float32
    # With vectorized NHWC operands and a tuned tile the exact IEEE fp32 dot
    # reaches ~8 TMAC/s, while the 3-term bf16 emulation only delivers ~4.7
    # TMAC/s effective, so fp32 keeps the exact IEEE dot (SPLIT disabled) and
    # gets its own, lower NHWC gate where the transform still pays for 3x3.
    split = False
    gather_work = inp.numel() * kh * kw
    use_nhwc = (not transposed) and (
        gather_work >= 150000000 or (ieee and gather_work >= 50000000)
    )

    if use_nhwc:
        # Channel-contiguous layout: the A-tile's k axis becomes a plain arange
        # over contiguous channels, which lowers to 16-byte vectorized loads
        # instead of scalar 2-byte gathers.
        nhwc = torch.empty((n_batch, ih, iw, cin), dtype=inp.dtype, device=inp.device)
        bw = 64 if iw >= 64 else max(16, triton.next_power_of_2(iw))
        cb = min(128, triton.next_power_of_2(cin))
        _nchw_to_nhwc_kernel[(triton.cdiv(iw, bw), ih, n_batch)](
            inp,
            nhwc,
            n_batch,
            cin,
            ih,
            iw,
            in_strides[0],
            in_strides[1],
            in_strides[2],
            in_strides[3],
            C_CONST=cin,
            CONTIG=bool(in_strides[3] == 1),
            BW=bw,
            CB=cb,
            num_warps=4,
        )
        nel = kh * kw * cout * cig
        w2 = torch.empty((nel,), dtype=inp.dtype, device=inp.device)
        _weight_to_tap_kernel[(triton.cdiv(nel, 1024),)](
            wt,
            w2,
            nel,
            cig,
            cout,
            cog,
            kw,
            w_strides[0],
            w_strides[1],
            w_strides[2],
            w_strides[3],
            BLOCK=1024,
            num_warps=4,
        )
        _conv2d_fwd_nhwc_kernel[
            lambda META: (
                triton.cdiv(n_pos, META["BLOCK_M"]),
                triton.cdiv(cog, META["BLOCK_N"]),
                n_batch * groups,
            )
        ](
            nhwc,
            w2,
            bias_ptr,
            out4,
            nhwc.stride(0),
            nhwc.stride(2),
            cout * cig,
            cig,
            o_strides[0],
            o_strides[1],
            o_strides[2],
            o_strides[3],
            n_pos,
            ih,
            iw,
            cog,
            ow,
            sh,
            sw,
            ph,
            pw,
            dh,
            dw,
            CIG=cig,
            KH=kh,
            KW=kw,
            GROUPS=groups,
            HAS_BIAS=bias is not None,
            SPLIT=split,
            IEEE=ieee,
            DTYPE_KEY=inp.dtype.itemsize,
        )
        return None

    if transposed:
        _conv2d_t_kernel[grid](
            inp,
            wt,
            bias_ptr,
            out4,
            in_strides[0],
            in_strides[1],
            in_strides[2],
            in_strides[3],
            w_strides[0],
            w_strides[1],
            w_strides[2],
            w_strides[3],
            o_strides[0],
            o_strides[1],
            o_strides[2],
            o_strides[3],
            n_pos,
            ih,
            iw,
            cog,
            ow,
            sh,
            sw,
            ph,
            pw,
            dh,
            dw,
            CIG=cig,
            KH=kh,
            KW=kw,
            GROUPS=groups,
            HAS_BIAS=bias is not None,
            IEEE=ieee,
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        _conv2d_fwd_kernel[grid](
            inp,
            wt,
            bias_ptr,
            out4,
            in_strides[0],
            in_strides[1],
            in_strides[2],
            in_strides[3],
            w_strides[0],
            w_strides[1],
            w_strides[2],
            w_strides[3],
            o_strides[0],
            o_strides[1],
            o_strides[2],
            o_strides[3],
            n_pos,
            ih,
            iw,
            cog,
            ow,
            sh,
            sw,
            ph,
            pw,
            dh,
            dw,
            CIG=cig,
            KH=kh,
            KW=kw,
            GROUPS=groups,
            HAS_BIAS=bias is not None,
            IEEE=ieee,
            SPLIT=split,
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return None


def convolution_overrideable_out(
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    out=None,
):
    logger.debug("GEMS_HYGON CONVOLUTION_OVERRIDEABLE_OUT")
    """DPS ``aten::convolution_overrideable.out`` driven by Triton kernels."""
    in_dtype = input.dtype
    in_device = input.device

    if weight.dtype != in_dtype or weight.device != in_device:
        raise RuntimeError(
            "convolution_overrideable: weight must match the input dtype and device"
        )
    if bias is not None and (bias.dtype != in_dtype or bias.device != in_device):
        raise RuntimeError(
            "convolution_overrideable: bias must match the input dtype and device"
        )
    if out is not None:
        if not isinstance(out, torch.Tensor):
            raise RuntimeError("convolution_overrideable: out must be a Tensor")
        if out.dtype != in_dtype:
            raise RuntimeError(
                "convolution_overrideable: expected out to have dtype "
                f"{in_dtype} but got {out.dtype}"
            )
        if out.device != in_device:
            raise RuntimeError(
                "convolution_overrideable: expected out to be on device "
                f"{in_device} but got {out.device}"
            )

    spatial = weight.dim() - 2
    if spatial == 1:
        sh = _int_list(stride, 1)[0]
        ph = _int_list(padding, 1)[0]
        dh = _int_list(dilation, 1)[0]
        oph = _int_list(output_padding, 1)[0]
        sw, pw, dw, opw = 1, 0, 1, 0
        inp = input.unsqueeze(-1)
        wt = weight.unsqueeze(-1)
        kh, kw = wt.shape[2], 1
    elif spatial == 2:
        sh, sw = _int_list(stride, 2)
        ph, pw = _int_list(padding, 2)
        dh, dw = _int_list(dilation, 2)
        oph, opw = _int_list(output_padding, 2)
        inp, wt = input, weight
        kh, kw = wt.shape[2], wt.shape[3]
    else:
        raise NotImplementedError(
            "convolution_overrideable: only 1D and 2D convolutions are supported"
        )

    groups = int(groups)
    if groups < 1:
        raise RuntimeError("convolution_overrideable: groups must be >= 1")
    n_batch, cin = inp.shape[0], inp.shape[1]
    if transposed:
        cout = wt.shape[1] * groups
    else:
        cout = wt.shape[0]
    if cin % groups or cout % groups:
        raise RuntimeError(
            "convolution_overrideable: channels must be divisible by groups"
        )

    ih, iw = inp.shape[2], inp.shape[3]
    if transposed:
        oh = (ih - 1) * sh - 2 * ph + dh * (kh - 1) + oph + 1
        ow = (iw - 1) * sw - 2 * pw + dw * (kw - 1) + opw + 1
    else:
        oh = (ih + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        ow = (iw + 2 * pw - dw * (kw - 1) - 1) // sw + 1

    if spatial == 1:
        true_shape = (n_batch, cout, oh)
        if out is None:
            out = torch.empty(true_shape, dtype=in_dtype, device=in_device)
        else:
            _resize_out(out, true_shape)
        out4 = out.unsqueeze(-1)
    else:
        true_shape = (n_batch, cout, oh, ow)
        if out is None:
            out = torch.empty(true_shape, dtype=in_dtype, device=in_device)
        else:
            _resize_out(out, true_shape)
        out4 = out

    if out4.numel() > 0:
        _launch(
            inp,
            wt,
            bias,
            bool(transposed),
            groups,
            sh,
            sw,
            ph,
            pw,
            dh,
            dw,
            oph,
            opw,
            kh,
            kw,
            out4,
        )
    return out

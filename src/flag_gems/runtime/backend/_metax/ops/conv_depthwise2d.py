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


def _dwconv2d_kernel(
    in_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    C,
    OUTC,
    H,
    W,
    OH,
    OW,
    in_hs,
    in_ws,
    M,
    HAS_BIAS: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    oc = pid_nc % OUTC
    ic = oc // M
    n = pid_nc // OUTC

    oh = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    ow = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    ow = tl.max_contiguous(tl.multiple_of(ow, BLOCK_W), BLOCK_W)
    ow_ok = ow < OW

    x_base = in_ptr + (n * C + ic) * (H * W)
    o_base = out_ptr + pid_nc * (OH * OW)
    w_base = w_ptr + oc * (KH * KW)

    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Hoisted per-tap column vectors (loop-invariant across kh).
    for kw in tl.static_range(KW):
        iw = ow * SW - PW + kw * DW
        col_ok = ow_ok & (iw >= 0) & (iw < W)
        for kh in tl.static_range(KH):
            ih = oh * SH - PH + kh * DH
            row_ok = (ih >= 0) & (ih < H)
            m = row_ok[:, None] & col_ok[None, :]
            wv = tl.load(w_base + kh * KW + kw).to(tl.float32)
            xv = tl.load(
                x_base + ih[:, None] * in_hs + iw[None, :] * in_ws,
                mask=m,
                other=0.0,
            ).to(tl.float32)
            acc += xv * wv

    if HAS_BIAS:
        acc += tl.load(b_ptr + oc).to(tl.float32)

    out_mask = (oh[:, None] < OH) & ow_ok[None, :]
    tl.store(
        o_base + oh[:, None] * OW + ow[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=out_mask,
    )


@triton.jit
def _dwconv2d_s2_kernel(
    in_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    C,
    OUTC,
    H,
    W,
    OH,
    OW,
    in_hs,
    in_ws,
    M,
    HAS_BIAS: tl.constexpr,
    KH: tl.constexpr,
    SH: tl.constexpr,
    PH: tl.constexpr,
    DH: tl.constexpr,
    PW: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Specialized for SW == 2, DW == 1, KW == 3.
    # iw = 2*ow - PW + kw  (kw in 0..2).  With c0 = -PW the three taps are:
    #   kw=0 -> input[2*ow + c0]        = seg0 even lane
    #   kw=1 -> input[2*ow + c0 + 1]    = seg1 even lane
    #   kw=2 -> input[2*ow + c0 + 2]    = seg1 odd lane
    # where seg0 = contiguous row at column offset 2*col0 + c0,
    #       seg1 = contiguous row at column offset 2*col0 + c0 + 1.
    pid_nc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    oc = pid_nc % OUTC
    ic = oc // M
    n = pid_nc // OUTC

    oh = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    ow = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    ow = tl.max_contiguous(tl.multiple_of(ow, BLOCK_W), BLOCK_W)
    ow_ok = ow < OW

    col0 = pid_w * BLOCK_W
    c0: tl.constexpr = -PW
    j = tl.arange(0, 2 * BLOCK_W)
    cols0 = 2 * col0 + c0 + j
    cols1 = 2 * col0 + c0 + 1 + j
    col0_ok = (cols0 >= 0) & (cols0 < W)
    col1_ok = (cols1 >= 0) & (cols1 < W)

    x_base = in_ptr + (n * C + ic) * (H * W)
    o_base = out_ptr + pid_nc * (OH * OW)
    w_base = w_ptr + oc * (KH * 3)

    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    for kh in tl.static_range(KH):
        ih = oh * SH - PH + kh * DH
        row_ok = (ih >= 0) & (ih < H)
        m0 = row_ok[:, None] & col0_ok[None, :]
        m1 = row_ok[:, None] & col1_ok[None, :]
        seg0 = tl.load(
            x_base + ih[:, None] * in_hs + cols0[None, :] * in_ws,
            mask=m0,
            other=0.0,
        )
        seg1 = tl.load(
            x_base + ih[:, None] * in_hs + cols1[None, :] * in_ws,
            mask=m1,
            other=0.0,
        )
        r0 = tl.reshape(seg0, (BLOCK_H, BLOCK_W, 2))
        r1 = tl.reshape(seg1, (BLOCK_H, BLOCK_W, 2))
        e0, _ = tl.split(r0)
        e1, o1 = tl.split(r1)
        wv0 = tl.load(w_base + kh * 3 + 0).to(tl.float32)
        wv1 = tl.load(w_base + kh * 3 + 1).to(tl.float32)
        wv2 = tl.load(w_base + kh * 3 + 2).to(tl.float32)
        acc += e0.to(tl.float32) * wv0
        acc += e1.to(tl.float32) * wv1
        acc += o1.to(tl.float32) * wv2

    if HAS_BIAS:
        acc += tl.load(b_ptr + oc).to(tl.float32)

    out_mask = (oh[:, None] < OH) & ow_ok[None, :]
    tl.store(
        o_base + oh[:, None] * OW + ow[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=out_mask,
    )


def _pair(v):
    if isinstance(v, torch.Tensor):
        v = v.tolist()
    if isinstance(v, (tuple, list)):
        return int(v[0]), int(v[1])
    return int(v), int(v)


def _conv_depthwise2d(input, weight, kernel_size, bias, stride, padding, dilation):
    N, C, H, W = input.shape
    OUTC = weight.shape[0]
    if weight.dim() == 4:
        KH, KW = int(weight.shape[2]), int(weight.shape[3])
    elif weight.dim() == 3:
        KH, KW = int(weight.shape[1]), int(weight.shape[2])
    else:
        KH, KW = _pair(kernel_size)

    SH, SW = _pair(stride)
    PH, PW = _pair(padding)
    DH, DW = _pair(dilation)

    OH = (H + 2 * PH - DH * (KH - 1) - 1) // SH + 1
    OW = (W + 2 * PW - DW * (KW - 1) - 1) // SW + 1

    out = torch.empty((N, OUTC, OH, OW), device=input.device, dtype=input.dtype)

    M = OUTC // C
    has_bias = bias is not None
    b_ptr = bias if has_bias else weight

    # Tile dispatch: small planes (both dims <= 32) use narrow tiles to keep
    # lane utilization high.  The stride-2 specialized kernel keeps the proven
    # 8x64/4-warp big tile; the generic big path uses 4-row tiles to remove
    # the half-wasted tail tile on odd output heights (e.g. OH=52, 5x5 case).
    if SW == 2 and DW == 1 and KW == 3:
        if OW <= 32 and OH <= 32:
            BLOCK_H = 16
            BLOCK_W = 16 if OW <= 16 else 32
            num_warps = 2
        else:
            BLOCK_H = 8
            BLOCK_W = 64
            num_warps = 4
    else:
        if OW <= 32 and OH <= 32:
            BLOCK_H = 16
            BLOCK_W = 16 if OW <= 16 else 32
            num_warps = 2
        else:
            BLOCK_H = 2
            BLOCK_W = 64
            num_warps = 1
    grid = (N * OUTC, triton.cdiv(OH, BLOCK_H), triton.cdiv(OW, BLOCK_W))
    if SW == 2 and DW == 1 and KW == 3:
        _dwconv2d_s2_kernel[grid](
            input,
            weight,
            b_ptr,
            out,
            C,
            OUTC,
            H,
            W,
            OH,
            OW,
            input.stride(2),
            input.stride(3),
            M,
            HAS_BIAS=has_bias,
            KH=KH,
            SH=SH,
            PH=PH,
            DH=DH,
            PW=PW,
            BLOCK_H=BLOCK_H,
            BLOCK_W=BLOCK_W,
            num_warps=num_warps,
        )
    else:
        _dwconv2d_kernel[grid](
            input,
            weight,
            b_ptr,
            out,
            C,
            OUTC,
            H,
            W,
            OH,
            OW,
            input.stride(2),
            input.stride(3),
            M,
            HAS_BIAS=has_bias,
            KH=KH,
            KW=KW,
            SH=SH,
            SW=SW,
            PH=PH,
            PW=PW,
            DH=DH,
            DW=DW,
            BLOCK_H=BLOCK_H,
            BLOCK_W=BLOCK_W,
            num_warps=num_warps,
        )
    return out

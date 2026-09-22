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

from flag_gems.ops.upsample_bilinear2d import (
    upsample_bilinear2d as default_upsample_bilinear2d,
)

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


# ---------------------------------------------------------------------------
# Bilinear 2D upsampling (PyTorch upsample_bilinear2d semantics).
#
# For every output pixel (oh, ow):
#   align_corners=True : h1r = rheight * oh,          rheight = (H-1)/(OH-1)
#   align_corners=False: h1r = rheight * (oh + 0.5) - 0.5, clamped to >= 0,
#                        rheight = 1/scales_h if scales_h > 0 else H/OH
#   h1 = floor(h1r), lambda_h = h1r - h1
#   rows: h1 (weight 1-lambda_h) and min(h1+1, H-1) (weight lambda_h)
# (same for columns)
#
# A specialized kernel handles the exact 2x / align_corners=False case with
# fully affine (vectorizable) loads over a (RP+2)x3 input neighborhood per
# (2*RP) x (2*BLOCK_M) output tile, and 2D pair stores. Everything else uses
# the general gather kernel.
# ---------------------------------------------------------------------------


@triton.jit
def _upsample_bilinear2d_kernel(
    in_ptr,
    out_ptr,
    H,
    W,
    OH,
    OW,
    rh,
    rw,
    align_corners: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Compute in the input precision except fp16/bf16 (use fp32 accumulator).
    elem = in_ptr.dtype.element_ty
    if elem == tl.float64:
        comp = tl.float64
    else:
        comp = tl.float32

    pid_row = tl.program_id(0)
    pid_w = tl.program_id(1)

    oh = pid_row % OH
    nc = pid_row // OH

    # --- row interpolation coefficients (scalar) ---
    oh_f = oh.to(comp)
    if align_corners:
        h1r = rh * oh_f
    else:
        h1r = rh * (oh_f + 0.5) - 0.5
        h1r = tl.maximum(h1r, 0.0)
    h1 = h1r.to(tl.int32)
    h1l = h1r - h1.to(comp)
    h1c = tl.minimum(h1 + 1, H - 1)

    # --- column interpolation coefficients (vector) ---
    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = offs_w < OW
    w_f = offs_w.to(comp)
    if align_corners:
        w1r = rw * w_f
    else:
        w1r = rw * (w_f + 0.5) - 0.5
        w1r = tl.maximum(w1r, 0.0)
    w1 = w1r.to(tl.int32)
    w1l = w1r - w1.to(comp)
    w1c = tl.minimum(w1 + 1, W - 1)

    # --- gather and interpolate ---
    in_base = nc.to(tl.int64) * H * W
    row0 = in_base + h1.to(tl.int64) * W
    row1 = in_base + h1c.to(tl.int64) * W
    w1_64 = w1.to(tl.int64)
    w1c_64 = w1c.to(tl.int64)

    a = tl.load(in_ptr + row0 + w1_64, mask=mask, other=0.0)
    b = tl.load(in_ptr + row0 + w1c_64, mask=mask, other=0.0)
    c = tl.load(in_ptr + row1 + w1_64, mask=mask, other=0.0)
    d = tl.load(in_ptr + row1 + w1c_64, mask=mask, other=0.0)

    top = a * (1.0 - w1l) + b * w1l
    bot = c * (1.0 - w1l) + d * w1l
    out_val = top * (1.0 - h1l) + bot * h1l

    out_base = nc.to(tl.int64) * OH * OW
    out_off = out_base + oh.to(tl.int64) * OW + offs_w.to(tl.int64)
    tl.store(out_ptr + out_off, out_val, mask=mask)


@triton.jit
def _upsample_bilinear2d_2x_kernel(
    in_ptr,
    out_ptr,
    H,
    W,
    OH,
    OW,
    BLOCK_M: tl.constexpr,
):
    # Exact 2x upsample, align_corners=False, rh = rw = 0.5.
    # One row-pair per program: 2 output rows x (2*BLOCK_M) columns from a
    # 3x3 input neighborhood.
    elem = in_ptr.dtype.element_ty
    if elem == tl.float64:
        comp = tl.float64
    else:
        comp = tl.float32

    pid_r = tl.program_id(0)  # over N*C*(OH//2)
    pid_m = tl.program_id(1)  # over column-pair blocks

    npairs = OW // 2
    k = pid_r % (OH // 2)
    nc = pid_r // (OH // 2)

    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mvalid = m < npairs

    lh0 = tl.where(k > 0, 0.75, 0.0)
    mu0 = tl.where(m > 0, 0.75, 0.0).to(comp)

    r0 = tl.maximum(k - 1, 0)
    r1 = k
    r2 = tl.minimum(k + 1, H - 1)
    c0 = m - 1
    c1 = m
    c2 = m + 1

    in_base = nc * H * W
    base0 = in_base + r0 * W
    base1 = in_base + r1 * W
    base2 = in_base + r2 * W

    I11 = tl.load(in_ptr + base1 + c1, mask=mvalid, other=0.0)
    I01 = tl.load(in_ptr + base0 + c1, mask=mvalid, other=I11)
    I21 = tl.load(in_ptr + base2 + c1, mask=mvalid, other=I11)
    I10 = tl.load(in_ptr + base1 + c0, mask=(m > 0) & mvalid, other=I11)
    I12 = tl.load(in_ptr + base1 + c2, mask=(m < W - 1) & mvalid, other=I11)
    I00 = tl.load(in_ptr + base0 + c0, mask=(m > 0) & mvalid, other=I01)
    I02 = tl.load(in_ptr + base0 + c2, mask=(m < W - 1) & mvalid, other=I01)
    I20 = tl.load(in_ptr + base2 + c0, mask=(m > 0) & mvalid, other=I21)
    I22 = tl.load(in_ptr + base2 + c2, mask=(m < W - 1) & mvalid, other=I21)

    I00 = I00.to(comp)
    I01 = I01.to(comp)
    I02 = I02.to(comp)
    I10 = I10.to(comp)
    I11 = I11.to(comp)
    I12 = I12.to(comp)
    I20 = I20.to(comp)
    I21 = I21.to(comp)
    I22 = I22.to(comp)

    w00 = (1.0 - mu0) * I00 + mu0 * I01
    w10 = (1.0 - mu0) * I10 + mu0 * I11
    w20 = (1.0 - mu0) * I20 + mu0 * I21
    w01 = 0.75 * I01 + 0.25 * I02
    w11 = 0.75 * I11 + 0.25 * I12
    w21 = 0.75 * I21 + 0.25 * I22

    vA_e = (1.0 - lh0) * w00 + lh0 * w10
    vA_o = (1.0 - lh0) * w01 + lh0 * w11
    vB_e = 0.75 * w10 + 0.25 * w20
    vB_o = 0.75 * w11 + 0.25 * w21

    j2d = tl.arange(0, 2)
    valA = tl.where(j2d[None, :] == 0, vA_e[:, None], vA_o[:, None])
    valB = tl.where(j2d[None, :] == 0, vB_e[:, None], vB_o[:, None])

    out_base = nc * OH * OW
    off2d = 2 * m[:, None] + j2d[None, :]
    mask2d = off2d < OW
    tl.store(out_ptr + out_base + (2 * k) * OW + off2d, valA, mask=mask2d)
    tl.store(out_ptr + out_base + (2 * k + 1) * OW + off2d, valB, mask=mask2d)


@triton.jit
def _upsample_bilinear2d_2x_rp4_kernel(
    in_ptr,
    out_ptr,
    H,
    W,
    OH,
    OW,
    BLOCK_M: tl.constexpr,
):
    # Exact 2x upsample, align_corners=False, rh = rw = 0.5.
    # Four row-pairs per program: 8 output rows x (2*BLOCK_M) columns from a
    # 6x3 input neighborhood (18 loads for 16 output rows x BLOCK_M vs 36 for
    # four separate row-pair programs). Rows shared between adjacent pairs.
    elem = in_ptr.dtype.element_ty
    if elem == tl.float64:
        comp = tl.float64
    else:
        comp = tl.float32

    pid_r = tl.program_id(0)  # over N*C*(OH//8)
    pid_m = tl.program_id(1)  # over column-pair blocks

    npairs = OW // 2
    k0 = pid_r % (OH // 8)
    k0 = k0 * 4  # first row-pair index
    nc = pid_r // (OH // 8)

    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mvalid = m < npairs
    mu0 = tl.where(m > 0, 0.75, 0.0).to(comp)

    c0 = m - 1
    c1 = m
    c2 = m + 1
    mc0 = (m > 0) & mvalid
    mc2 = (m < W - 1) & mvalid

    # 6 input rows (scalar-clamped), rows k0-1 .. k0+4
    r0 = tl.maximum(k0 - 1, 0)
    r1 = k0
    r2 = k0 + 1
    r3 = k0 + 2
    r4 = k0 + 3
    r5 = tl.minimum(k0 + 4, H - 1)

    in_base = nc * H * W
    b0 = in_base + r0 * W
    b1 = in_base + r1 * W
    b2 = in_base + r2 * W
    b3 = in_base + r3 * W
    b4 = in_base + r4 * W
    b5 = in_base + r5 * W

    # 6 rows x 3 column-shifted loads
    A0 = tl.load(in_ptr + b0 + c1, mask=mvalid, other=0.0)
    A1 = tl.load(in_ptr + b1 + c1, mask=mvalid, other=0.0)
    A2 = tl.load(in_ptr + b2 + c1, mask=mvalid, other=0.0)
    A3 = tl.load(in_ptr + b3 + c1, mask=mvalid, other=0.0)
    A4 = tl.load(in_ptr + b4 + c1, mask=mvalid, other=0.0)
    A5 = tl.load(in_ptr + b5 + c1, mask=mvalid, other=0.0)
    A0l = tl.load(in_ptr + b0 + c0, mask=mc0, other=A0)
    A0r = tl.load(in_ptr + b0 + c2, mask=mc2, other=A0)
    A1l = tl.load(in_ptr + b1 + c0, mask=mc0, other=A1)
    A1r = tl.load(in_ptr + b1 + c2, mask=mc2, other=A1)
    A2l = tl.load(in_ptr + b2 + c0, mask=mc0, other=A2)
    A2r = tl.load(in_ptr + b2 + c2, mask=mc2, other=A2)
    A3l = tl.load(in_ptr + b3 + c0, mask=mc0, other=A3)
    A3r = tl.load(in_ptr + b3 + c2, mask=mc2, other=A3)
    A4l = tl.load(in_ptr + b4 + c0, mask=mc0, other=A4)
    A4r = tl.load(in_ptr + b4 + c2, mask=mc2, other=A4)
    A5l = tl.load(in_ptr + b5 + c0, mask=mc0, other=A5)
    A5r = tl.load(in_ptr + b5 + c2, mask=mc2, other=A5)

    A0 = A0.to(comp)
    A0l = A0l.to(comp)
    A0r = A0r.to(comp)
    A1 = A1.to(comp)
    A1l = A1l.to(comp)
    A1r = A1r.to(comp)
    A2 = A2.to(comp)
    A2l = A2l.to(comp)
    A2r = A2r.to(comp)
    A3 = A3.to(comp)
    A3l = A3l.to(comp)
    A3r = A3r.to(comp)
    A4 = A4.to(comp)
    A4l = A4l.to(comp)
    A4r = A4r.to(comp)
    A5 = A5.to(comp)
    A5l = A5l.to(comp)
    A5r = A5r.to(comp)

    # column interpolation per input row
    e0 = (1.0 - mu0) * A0l + mu0 * A0
    o0 = 0.75 * A0 + 0.25 * A0r
    e1 = (1.0 - mu0) * A1l + mu0 * A1
    o1 = 0.75 * A1 + 0.25 * A1r
    e2 = (1.0 - mu0) * A2l + mu0 * A2
    o2 = 0.75 * A2 + 0.25 * A2r
    e3 = (1.0 - mu0) * A3l + mu0 * A3
    o3 = 0.75 * A3 + 0.25 * A3r
    e4 = (1.0 - mu0) * A4l + mu0 * A4
    o4 = 0.75 * A4 + 0.25 * A4r
    e5 = (1.0 - mu0) * A5l + mu0 * A5
    o5 = 0.75 * A5 + 0.25 * A5r

    lh0 = tl.where(k0 > 0, 0.75, 0.0).to(comp)

    # pair 0 (output rows 2*k0, 2*k0+1): rows 0,1,2
    vE = (1.0 - lh0) * e0 + lh0 * e1
    vO = (1.0 - lh0) * o0 + lh0 * o1
    vE2 = 0.75 * e1 + 0.25 * e2
    vO2 = 0.75 * o1 + 0.25 * o2
    # pair 1 (rows 2*k0+2, 2*k0+3): rows 1,2,3
    vE3 = 0.25 * e1 + 0.75 * e2
    vO3 = 0.25 * o1 + 0.75 * o2
    vE4 = 0.75 * e2 + 0.25 * e3
    vO4 = 0.75 * o2 + 0.25 * o3
    # pair 2 (rows 2*k0+4, 2*k0+5): rows 2,3,4
    vE5 = 0.25 * e2 + 0.75 * e3
    vO5 = 0.25 * o2 + 0.75 * o3
    vE6 = 0.75 * e3 + 0.25 * e4
    vO6 = 0.75 * o3 + 0.25 * o4
    # pair 3 (rows 2*k0+6, 2*k0+7): rows 3,4,5
    vE7 = 0.25 * e3 + 0.75 * e4
    vO7 = 0.25 * o3 + 0.75 * o4
    vE8 = 0.75 * e4 + 0.25 * e5
    vO8 = 0.75 * o4 + 0.25 * o5

    j2d = tl.arange(0, 2)
    off2d = 2 * m[:, None] + j2d[None, :]
    mask2d = off2d < OW
    out_base = nc * OH * OW
    rowoff = (2 * k0) * OW
    tl.store(
        out_ptr + out_base + rowoff + off2d,
        tl.where(j2d[None, :] == 0, vE[:, None], vO[:, None]),
        mask=mask2d,
    )
    tl.store(
        out_ptr + out_base + rowoff + OW + off2d,
        tl.where(j2d[None, :] == 0, vE2[:, None], vO2[:, None]),
        mask=mask2d,
    )
    tl.store(
        out_ptr + out_base + rowoff + 2 * OW + off2d,
        tl.where(j2d[None, :] == 0, vE3[:, None], vO3[:, None]),
        mask=mask2d,
    )
    tl.store(
        out_ptr + out_base + rowoff + 3 * OW + off2d,
        tl.where(j2d[None, :] == 0, vE4[:, None], vO4[:, None]),
        mask=mask2d,
    )
    tl.store(
        out_ptr + out_base + rowoff + 4 * OW + off2d,
        tl.where(j2d[None, :] == 0, vE5[:, None], vO5[:, None]),
        mask=mask2d,
    )
    tl.store(
        out_ptr + out_base + rowoff + 5 * OW + off2d,
        tl.where(j2d[None, :] == 0, vE6[:, None], vO6[:, None]),
        mask=mask2d,
    )
    tl.store(
        out_ptr + out_base + rowoff + 6 * OW + off2d,
        tl.where(j2d[None, :] == 0, vE7[:, None], vO7[:, None]),
        mask=mask2d,
    )
    tl.store(
        out_ptr + out_base + rowoff + 7 * OW + off2d,
        tl.where(j2d[None, :] == 0, vE8[:, None], vO8[:, None]),
        mask=mask2d,
    )


@triton.jit
def _upsample_bilinear2d_2x_rp4_w4_kernel(
    in_ptr,
    out_ptr,
    H,
    W,
    OH,
    OW,
    BLOCK_Q: tl.constexpr,
):
    # Exact 2x upsample, align_corners=False, rh = rw = 0.5.
    # Wide-column variant: each program produces 8 output rows x (4*BLOCK_Q)
    # output columns. Each lane covers TWO column-pairs (output cols 4q..4q+3)
    # fed by 4 aligned input columns (2q-1..2q+2) per row: 6 rows x 4 loads
    # = 24 loads for 32*BLOCK_Q outputs (0.75 loads/output vs 1.125 for the
    # 3-column stencil), and (BLOCK_Q,4) aligned stores.
    elem = in_ptr.dtype.element_ty
    if elem == tl.float64:
        comp = tl.float64
    else:
        comp = tl.float32

    pid_r = tl.program_id(0)  # over N*C*(OH//8)
    pid_q = tl.program_id(1)  # over 4-column blocks

    npairs4 = OW // 4
    k0 = pid_r % (OH // 8)
    k0 = k0 * 4
    nc = pid_r // (OH // 8)

    q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    qvalid = q < npairs4
    mq = (q > 0) & qvalid
    mq2 = (q < npairs4 - 1) & qvalid
    muq = tl.where(q > 0, 0.75, 0.0).to(comp)

    r0 = tl.maximum(k0 - 1, 0)
    r1 = k0
    r2 = k0 + 1
    r3 = k0 + 2
    r4 = k0 + 3
    r5 = tl.minimum(k0 + 4, H - 1)

    in_base = nc * H * W
    b0 = in_base + r0 * W
    b1 = in_base + r1 * W
    b2 = in_base + r2 * W
    b3 = in_base + r3 * W
    b4 = in_base + r4 * W
    b5 = in_base + r5 * W

    # --- column interpolation, row by row (4 aligned loads each) ---
    # row 0
    L1 = tl.load(in_ptr + b0 + 2 * q, mask=qvalid, other=0.0)
    L2 = tl.load(in_ptr + b0 + 2 * q + 1, mask=qvalid, other=0.0)
    L0 = tl.load(in_ptr + b0 + 2 * q - 1, mask=mq, other=L1)
    L3 = tl.load(in_ptr + b0 + 2 * q + 2, mask=mq2, other=L2)
    L1 = L1.to(comp)
    L2 = L2.to(comp)
    L0 = L0.to(comp)
    L3 = L3.to(comp)
    e0_0 = (1.0 - muq) * L0 + muq * L1
    e0_1 = 0.75 * L1 + 0.25 * L2
    e0_2 = 0.25 * L1 + 0.75 * L2
    e0_3 = 0.75 * L2 + 0.25 * L3
    # row 1
    L1 = tl.load(in_ptr + b1 + 2 * q, mask=qvalid, other=0.0)
    L2 = tl.load(in_ptr + b1 + 2 * q + 1, mask=qvalid, other=0.0)
    L0 = tl.load(in_ptr + b1 + 2 * q - 1, mask=mq, other=L1)
    L3 = tl.load(in_ptr + b1 + 2 * q + 2, mask=mq2, other=L2)
    L1 = L1.to(comp)
    L2 = L2.to(comp)
    L0 = L0.to(comp)
    L3 = L3.to(comp)
    e1_0 = (1.0 - muq) * L0 + muq * L1
    e1_1 = 0.75 * L1 + 0.25 * L2
    e1_2 = 0.25 * L1 + 0.75 * L2
    e1_3 = 0.75 * L2 + 0.25 * L3
    # row 2
    L1 = tl.load(in_ptr + b2 + 2 * q, mask=qvalid, other=0.0)
    L2 = tl.load(in_ptr + b2 + 2 * q + 1, mask=qvalid, other=0.0)
    L0 = tl.load(in_ptr + b2 + 2 * q - 1, mask=mq, other=L1)
    L3 = tl.load(in_ptr + b2 + 2 * q + 2, mask=mq2, other=L2)
    L1 = L1.to(comp)
    L2 = L2.to(comp)
    L0 = L0.to(comp)
    L3 = L3.to(comp)
    e2_0 = (1.0 - muq) * L0 + muq * L1
    e2_1 = 0.75 * L1 + 0.25 * L2
    e2_2 = 0.25 * L1 + 0.75 * L2
    e2_3 = 0.75 * L2 + 0.25 * L3
    # row 3
    L1 = tl.load(in_ptr + b3 + 2 * q, mask=qvalid, other=0.0)
    L2 = tl.load(in_ptr + b3 + 2 * q + 1, mask=qvalid, other=0.0)
    L0 = tl.load(in_ptr + b3 + 2 * q - 1, mask=mq, other=L1)
    L3 = tl.load(in_ptr + b3 + 2 * q + 2, mask=mq2, other=L2)
    L1 = L1.to(comp)
    L2 = L2.to(comp)
    L0 = L0.to(comp)
    L3 = L3.to(comp)
    e3_0 = (1.0 - muq) * L0 + muq * L1
    e3_1 = 0.75 * L1 + 0.25 * L2
    e3_2 = 0.25 * L1 + 0.75 * L2
    e3_3 = 0.75 * L2 + 0.25 * L3
    # row 4
    L1 = tl.load(in_ptr + b4 + 2 * q, mask=qvalid, other=0.0)
    L2 = tl.load(in_ptr + b4 + 2 * q + 1, mask=qvalid, other=0.0)
    L0 = tl.load(in_ptr + b4 + 2 * q - 1, mask=mq, other=L1)
    L3 = tl.load(in_ptr + b4 + 2 * q + 2, mask=mq2, other=L2)
    L1 = L1.to(comp)
    L2 = L2.to(comp)
    L0 = L0.to(comp)
    L3 = L3.to(comp)
    e4_0 = (1.0 - muq) * L0 + muq * L1
    e4_1 = 0.75 * L1 + 0.25 * L2
    e4_2 = 0.25 * L1 + 0.75 * L2
    e4_3 = 0.75 * L2 + 0.25 * L3
    # row 5
    L1 = tl.load(in_ptr + b5 + 2 * q, mask=qvalid, other=0.0)
    L2 = tl.load(in_ptr + b5 + 2 * q + 1, mask=qvalid, other=0.0)
    L0 = tl.load(in_ptr + b5 + 2 * q - 1, mask=mq, other=L1)
    L3 = tl.load(in_ptr + b5 + 2 * q + 2, mask=mq2, other=L2)
    L1 = L1.to(comp)
    L2 = L2.to(comp)
    L0 = L0.to(comp)
    L3 = L3.to(comp)
    e5_0 = (1.0 - muq) * L0 + muq * L1
    e5_1 = 0.75 * L1 + 0.25 * L2
    e5_2 = 0.25 * L1 + 0.75 * L2
    e5_3 = 0.75 * L2 + 0.25 * L3

    lh0 = tl.where(k0 > 0, 0.75, 0.0).to(comp)

    j4 = tl.arange(0, 4)
    off4 = 4 * q[:, None] + j4[None, :]
    mask4 = off4 < OW
    out_base = nc * OH * OW
    rowoff = (2 * k0) * OW

    # pair 0 -> output rows 2*k0, 2*k0+1 (rows 0,1,2)
    vE = (1.0 - lh0) * e0_0 + lh0 * e1_0
    vO = (1.0 - lh0) * e0_1 + lh0 * e1_1
    vP = (1.0 - lh0) * e0_2 + lh0 * e1_2
    vQ = (1.0 - lh0) * e0_3 + lh0 * e1_3
    vE2 = 0.75 * e1_0 + 0.25 * e2_0
    vO2 = 0.75 * e1_1 + 0.25 * e2_1
    vP2 = 0.75 * e1_2 + 0.25 * e2_2
    vQ2 = 0.75 * e1_3 + 0.25 * e2_3
    valA = tl.where(
        j4[None, :] == 0,
        vE[:, None],
        tl.where(
            j4[None, :] == 1,
            vO[:, None],
            tl.where(
                j4[None, :] == 2,
                vP[:, None],
                vQ[:, None],
            ),
        ),
    )
    valB = tl.where(
        j4[None, :] == 0,
        vE2[:, None],
        tl.where(
            j4[None, :] == 1,
            vO2[:, None],
            tl.where(
                j4[None, :] == 2,
                vP2[:, None],
                vQ2[:, None],
            ),
        ),
    )
    tl.store(out_ptr + out_base + rowoff + off4, valA, mask=mask4)
    tl.store(out_ptr + out_base + rowoff + OW + off4, valB, mask=mask4)

    # pair 1 -> output rows 2*k0+2, 2*k0+3 (rows 1,2,3)
    vE = 0.25 * e1_0 + 0.75 * e2_0
    vO = 0.25 * e1_1 + 0.75 * e2_1
    vP = 0.25 * e1_2 + 0.75 * e2_2
    vQ = 0.25 * e1_3 + 0.75 * e2_3
    vE2 = 0.75 * e2_0 + 0.25 * e3_0
    vO2 = 0.75 * e2_1 + 0.25 * e3_1
    vP2 = 0.75 * e2_2 + 0.25 * e3_2
    vQ2 = 0.75 * e2_3 + 0.25 * e3_3
    valC = tl.where(
        j4[None, :] == 0,
        vE[:, None],
        tl.where(
            j4[None, :] == 1,
            vO[:, None],
            tl.where(
                j4[None, :] == 2,
                vP[:, None],
                vQ[:, None],
            ),
        ),
    )
    valD = tl.where(
        j4[None, :] == 0,
        vE2[:, None],
        tl.where(
            j4[None, :] == 1,
            vO2[:, None],
            tl.where(
                j4[None, :] == 2,
                vP2[:, None],
                vQ2[:, None],
            ),
        ),
    )
    tl.store(out_ptr + out_base + rowoff + 2 * OW + off4, valC, mask=mask4)
    tl.store(out_ptr + out_base + rowoff + 3 * OW + off4, valD, mask=mask4)

    # pair 2 -> output rows 2*k0+4, 2*k0+5 (rows 2,3,4)
    vE = 0.25 * e2_0 + 0.75 * e3_0
    vO = 0.25 * e2_1 + 0.75 * e3_1
    vP = 0.25 * e2_2 + 0.75 * e3_2
    vQ = 0.25 * e2_3 + 0.75 * e3_3
    vE2 = 0.75 * e3_0 + 0.25 * e4_0
    vO2 = 0.75 * e3_1 + 0.25 * e4_1
    vP2 = 0.75 * e3_2 + 0.25 * e4_2
    vQ2 = 0.75 * e3_3 + 0.25 * e4_3
    valE = tl.where(
        j4[None, :] == 0,
        vE[:, None],
        tl.where(
            j4[None, :] == 1,
            vO[:, None],
            tl.where(
                j4[None, :] == 2,
                vP[:, None],
                vQ[:, None],
            ),
        ),
    )
    valF = tl.where(
        j4[None, :] == 0,
        vE2[:, None],
        tl.where(
            j4[None, :] == 1,
            vO2[:, None],
            tl.where(
                j4[None, :] == 2,
                vP2[:, None],
                vQ2[:, None],
            ),
        ),
    )
    tl.store(out_ptr + out_base + rowoff + 4 * OW + off4, valE, mask=mask4)
    tl.store(out_ptr + out_base + rowoff + 5 * OW + off4, valF, mask=mask4)

    # pair 3 -> output rows 2*k0+6, 2*k0+7 (rows 3,4,5)
    vE = 0.25 * e3_0 + 0.75 * e4_0
    vO = 0.25 * e3_1 + 0.75 * e4_1
    vP = 0.25 * e3_2 + 0.75 * e4_2
    vQ = 0.25 * e3_3 + 0.75 * e4_3
    vE2 = 0.75 * e4_0 + 0.25 * e5_0
    vO2 = 0.75 * e4_1 + 0.25 * e5_1
    vP2 = 0.75 * e4_2 + 0.25 * e5_2
    vQ2 = 0.75 * e4_3 + 0.25 * e5_3
    valG = tl.where(
        j4[None, :] == 0,
        vE[:, None],
        tl.where(
            j4[None, :] == 1,
            vO[:, None],
            tl.where(
                j4[None, :] == 2,
                vP[:, None],
                vQ[:, None],
            ),
        ),
    )
    valH = tl.where(
        j4[None, :] == 0,
        vE2[:, None],
        tl.where(
            j4[None, :] == 1,
            vO2[:, None],
            tl.where(
                j4[None, :] == 2,
                vP2[:, None],
                vQ2[:, None],
            ),
        ),
    )
    tl.store(out_ptr + out_base + rowoff + 6 * OW + off4, valG, mask=mask4)
    tl.store(out_ptr + out_base + rowoff + 7 * OW + off4, valH, mask=mask4)


_BLOCK_W = 256
_NUM_WARPS = 4
_BLOCK_M = 128
_NUM_WARPS_2X = 4


def _specialized_upsample_bilinear2d(
    input, output_size, align_corners=False, scales_h=None, scales_w=None
):
    N, C, H, W = input.shape
    OH, OW = int(output_size[0]), int(output_size[1])

    input = input.contiguous()
    out = torch.empty((N, C, OH, OW), dtype=input.dtype, device=input.device)

    if align_corners:
        rh = (H - 1) / (OH - 1) if OH > 1 else 0.0
        rw = (W - 1) / (OW - 1) if OW > 1 else 0.0
    else:
        if scales_h is not None and scales_h > 0:
            rh = 1.0 / scales_h
        else:
            rh = H / OH
        if scales_w is not None and scales_w > 0:
            rw = 1.0 / scales_w
        else:
            rw = W / OW

    if (not align_corners) and rh == 0.5 and rw == 0.5 and OH % 2 == 0 and OW % 2 == 0:
        npairs = OW // 2
        if OH % 8 == 0:
            dim0 = N * C * (OH // 8)
            use_rp4 = True
        else:
            dim0 = N * C * (OH // 2)
            use_rp4 = False
        if (
            (input.dtype != torch.float32)
            and OH % 8 == 0
            and OW % 4 == 0
            and dim0 >= 1024
        ):
            # wide 4-column stencil: 2 column-pairs per lane, 24 loads / 32 outputs
            # (best for 2-byte dtypes; fp32 keeps the 16B/thread 3-column path)
            qq = OW // 4
            if qq <= 64:
                bq, nq = 64, 2
            else:
                bq, nq = 256, 4
            grid = (dim0, triton.cdiv(qq, bq))
            _upsample_bilinear2d_2x_rp4_w4_kernel[grid](
                input,
                out,
                H,
                W,
                OH,
                OW,
                BLOCK_Q=bq,
                num_warps=nq,
            )
        else:
            if npairs <= 128:
                bm = 128
            elif npairs < 512 or dim0 < 16384:
                bm = 256
            else:
                bm = 512
            if use_rp4:
                grid = (dim0, triton.cdiv(npairs, bm))
                _upsample_bilinear2d_2x_rp4_kernel[grid](
                    input,
                    out,
                    H,
                    W,
                    OH,
                    OW,
                    BLOCK_M=bm,
                    num_warps=2 if bm <= 128 else 4,
                )
            else:
                grid = (N * C * (OH // 2), triton.cdiv(npairs, bm))
                _upsample_bilinear2d_2x_kernel[grid](
                    input,
                    out,
                    H,
                    W,
                    OH,
                    OW,
                    BLOCK_M=bm,
                    num_warps=2 if bm <= 128 else 4,
                )
    else:
        grid = (N * C * OH, triton.cdiv(OW, _BLOCK_W))
        _upsample_bilinear2d_kernel[grid](
            input,
            out,
            H,
            W,
            OH,
            OW,
            rh,
            rw,
            align_corners=align_corners,
            BLOCK_W=_BLOCK_W,
            num_warps=_NUM_WARPS,
        )
    return out


def upsample_bilinear2d(
    input, output_size, align_corners=False, scales_h=None, scales_w=None
):
    logger.debug("GEMS_MTHREADS UPSAMPLE_BILINEAR2D")
    if (
        isinstance(input, torch.Tensor)
        and input.device.type == "musa"
        and input.dtype in _SUPPORTED_DTYPES
    ):
        return _specialized_upsample_bilinear2d(
            input, output_size, align_corners, scales_h, scales_w
        )
    return default_upsample_bilinear2d(
        input, output_size, align_corners, scales_h, scales_w
    )

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
def _mhc_post_bwd_kernel(
    x_ptr,
    res_ptr,
    post_ptr,
    comb_ptr,
    do_ptr,
    dx_ptr,
    dres_ptr,
    dpost_ptr,
    dcomb_ptr,
    H,
    BLOCK_H: tl.constexpr,
    NS: tl.constexpr,
    BATCH_ST: tl.constexpr,
    BATCH_LD: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    hst = H
    fh = 4 * H
    nchunk = tl.cdiv(H, BLOCK_H)

    if BATCH_LD:
        i4p = tl.arange(0, 4)
        i16 = tl.arange(0, 16)
        pv = tl.load(post_ptr + pid * 4 + i4p)
        cvt = tl.load(comb_ptr + pid * 16 + i16)
        p0 = tl.sum(tl.where(i4p == 0, pv, 0.0), axis=0)
        p1 = tl.sum(tl.where(i4p == 1, pv, 0.0), axis=0)
        p2 = tl.sum(tl.where(i4p == 2, pv, 0.0), axis=0)
        p3 = tl.sum(tl.where(i4p == 3, pv, 0.0), axis=0)
        c00 = tl.sum(tl.where(i16 == 0, cvt, 0.0), axis=0)
        c01 = tl.sum(tl.where(i16 == 1, cvt, 0.0), axis=0)
        c02 = tl.sum(tl.where(i16 == 2, cvt, 0.0), axis=0)
        c03 = tl.sum(tl.where(i16 == 3, cvt, 0.0), axis=0)
        c10 = tl.sum(tl.where(i16 == 4, cvt, 0.0), axis=0)
        c11 = tl.sum(tl.where(i16 == 5, cvt, 0.0), axis=0)
        c12 = tl.sum(tl.where(i16 == 6, cvt, 0.0), axis=0)
        c13 = tl.sum(tl.where(i16 == 7, cvt, 0.0), axis=0)
        c20 = tl.sum(tl.where(i16 == 8, cvt, 0.0), axis=0)
        c21 = tl.sum(tl.where(i16 == 9, cvt, 0.0), axis=0)
        c22 = tl.sum(tl.where(i16 == 10, cvt, 0.0), axis=0)
        c23 = tl.sum(tl.where(i16 == 11, cvt, 0.0), axis=0)
        c30 = tl.sum(tl.where(i16 == 12, cvt, 0.0), axis=0)
        c31 = tl.sum(tl.where(i16 == 13, cvt, 0.0), axis=0)
        c32 = tl.sum(tl.where(i16 == 14, cvt, 0.0), axis=0)
        c33 = tl.sum(tl.where(i16 == 15, cvt, 0.0), axis=0)
    else:
        pb = post_ptr + pid * 4
        p0 = tl.load(pb + 0)
        p1 = tl.load(pb + 1)
        p2 = tl.load(pb + 2)
        p3 = tl.load(pb + 3)
        cb = comb_ptr + pid * 16
        c00 = tl.load(cb + 0)
        c01 = tl.load(cb + 1)
        c02 = tl.load(cb + 2)
        c03 = tl.load(cb + 3)
        c10 = tl.load(cb + 4)
        c11 = tl.load(cb + 5)
        c12 = tl.load(cb + 6)
        c13 = tl.load(cb + 7)
        c20 = tl.load(cb + 8)
        c21 = tl.load(cb + 9)
        c22 = tl.load(cb + 10)
        c23 = tl.load(cb + 11)
        c30 = tl.load(cb + 12)
        c31 = tl.load(cb + 13)
        c32 = tl.load(cb + 14)
        c33 = tl.load(cb + 15)

    do_b = do_ptr + pid * fh
    r_b = res_ptr + pid * fh
    x_b = x_ptr + pid * hst
    dx_b = dx_ptr + pid * hst
    dr_b = dres_ptr + pid * fh

    # ---- per-lane fp32 partial accumulators for the two reduction outputs ----
    ap0 = tl.zeros([BLOCK_H], dtype=tl.float32)
    ap1 = tl.zeros([BLOCK_H], dtype=tl.float32)
    ap2 = tl.zeros([BLOCK_H], dtype=tl.float32)
    ap3 = tl.zeros([BLOCK_H], dtype=tl.float32)

    a00 = tl.zeros([BLOCK_H], dtype=tl.float32)
    a01 = tl.zeros([BLOCK_H], dtype=tl.float32)
    a02 = tl.zeros([BLOCK_H], dtype=tl.float32)
    a03 = tl.zeros([BLOCK_H], dtype=tl.float32)
    a10 = tl.zeros([BLOCK_H], dtype=tl.float32)
    a11 = tl.zeros([BLOCK_H], dtype=tl.float32)
    a12 = tl.zeros([BLOCK_H], dtype=tl.float32)
    a13 = tl.zeros([BLOCK_H], dtype=tl.float32)
    a20 = tl.zeros([BLOCK_H], dtype=tl.float32)
    a21 = tl.zeros([BLOCK_H], dtype=tl.float32)
    a22 = tl.zeros([BLOCK_H], dtype=tl.float32)
    a23 = tl.zeros([BLOCK_H], dtype=tl.float32)
    a30 = tl.zeros([BLOCK_H], dtype=tl.float32)
    a31 = tl.zeros([BLOCK_H], dtype=tl.float32)
    a32 = tl.zeros([BLOCK_H], dtype=tl.float32)
    a33 = tl.zeros([BLOCK_H], dtype=tl.float32)

    for c in tl.range(0, nchunk, num_stages=NS):
        o = c * BLOCK_H + offs

        xv = tl.load(x_b + o).to(tl.float32)
        d0 = tl.load(do_b + o).to(tl.float32)
        d1 = tl.load(do_b + hst + o).to(tl.float32)
        d2 = tl.load(do_b + 2 * hst + o).to(tl.float32)
        d3 = tl.load(do_b + 3 * hst + o).to(tl.float32)

        # d_x = sum_mo d_o[mo] * post[mo]
        dv = d0 * p0 + d1 * p1 + d2 * p2 + d3 * p3
        tl.store(dx_b + o, dv.to(tl.bfloat16))

        r0 = tl.load(r_b + o).to(tl.float32)
        r1 = tl.load(r_b + hst + o).to(tl.float32)
        r2 = tl.load(r_b + 2 * hst + o).to(tl.float32)
        r3 = tl.load(r_b + 3 * hst + o).to(tl.float32)

        # d_residual[mi] = sum_mo d_o[mo] * comb[mi, mo]
        v0 = d0 * c00 + d1 * c01 + d2 * c02 + d3 * c03
        v1 = d0 * c10 + d1 * c11 + d2 * c12 + d3 * c13
        v2 = d0 * c20 + d1 * c21 + d2 * c22 + d3 * c23
        v3 = d0 * c30 + d1 * c31 + d2 * c32 + d3 * c33
        tl.store(dr_b + o, v0.to(tl.bfloat16))
        tl.store(dr_b + hst + o, v1.to(tl.bfloat16))
        tl.store(dr_b + 2 * hst + o, v2.to(tl.bfloat16))
        tl.store(dr_b + 3 * hst + o, v3.to(tl.bfloat16))

        # d_post_layer_mix[mo] = sum_h d_o[mo] * x
        ap0 += d0 * xv
        ap1 += d1 * xv
        ap2 += d2 * xv
        ap3 += d3 * xv

        # d_comb_res_mix[mi, mo] = sum_h d_o[mo] * residual[mi]
        a00 += d0 * r0
        a01 += d1 * r0
        a02 += d2 * r0
        a03 += d3 * r0
        a10 += d0 * r1
        a11 += d1 * r1
        a12 += d2 * r1
        a13 += d3 * r1
        a20 += d0 * r2
        a21 += d1 * r2
        a22 += d2 * r2
        a23 += d3 * r2
        a30 += d0 * r3
        a31 += d1 * r3
        a32 += d2 * r3
        a33 += d3 * r3

    s_p0 = tl.sum(ap0, axis=0)
    s_p1 = tl.sum(ap1, axis=0)
    s_p2 = tl.sum(ap2, axis=0)
    s_p3 = tl.sum(ap3, axis=0)
    s00 = tl.sum(a00, axis=0)
    s01 = tl.sum(a01, axis=0)
    s02 = tl.sum(a02, axis=0)
    s03 = tl.sum(a03, axis=0)
    s10 = tl.sum(a10, axis=0)
    s11 = tl.sum(a11, axis=0)
    s12 = tl.sum(a12, axis=0)
    s13 = tl.sum(a13, axis=0)
    s20 = tl.sum(a20, axis=0)
    s21 = tl.sum(a21, axis=0)
    s22 = tl.sum(a22, axis=0)
    s23 = tl.sum(a23, axis=0)
    s30 = tl.sum(a30, axis=0)
    s31 = tl.sum(a31, axis=0)
    s32 = tl.sum(a32, axis=0)
    s33 = tl.sum(a33, axis=0)

    if BATCH_ST:

        i16s = tl.arange(0, 16)
        dcv = (
            s00 * (i16s == 0).to(tl.float32)
            + s01 * (i16s == 1).to(tl.float32)
            + s02 * (i16s == 2).to(tl.float32)
            + s03 * (i16s == 3).to(tl.float32)
            + s10 * (i16s == 4).to(tl.float32)
            + s11 * (i16s == 5).to(tl.float32)
            + s12 * (i16s == 6).to(tl.float32)
            + s13 * (i16s == 7).to(tl.float32)
            + s20 * (i16s == 8).to(tl.float32)
            + s21 * (i16s == 9).to(tl.float32)
            + s22 * (i16s == 10).to(tl.float32)
            + s23 * (i16s == 11).to(tl.float32)
            + s30 * (i16s == 12).to(tl.float32)
            + s31 * (i16s == 13).to(tl.float32)
            + s32 * (i16s == 14).to(tl.float32)
            + s33 * (i16s == 15).to(tl.float32)
        )
        tl.store(dcomb_ptr + pid * 16 + i16s, dcv)
        i4s = tl.arange(0, 4)
        dpv = (
            s_p0 * (i4s == 0).to(tl.float32)
            + s_p1 * (i4s == 1).to(tl.float32)
            + s_p2 * (i4s == 2).to(tl.float32)
            + s_p3 * (i4s == 3).to(tl.float32)
        )
        tl.store(dpost_ptr + pid * 4 + i4s, dpv)
    else:
        dp_b = dpost_ptr + pid * 4
        tl.store(dp_b + 0, s_p0)
        tl.store(dp_b + 1, s_p1)
        tl.store(dp_b + 2, s_p2)
        tl.store(dp_b + 3, s_p3)
        dc_b = dcomb_ptr + pid * 16
        tl.store(dc_b + 0, s00)
        tl.store(dc_b + 1, s01)
        tl.store(dc_b + 2, s02)
        tl.store(dc_b + 3, s03)
        tl.store(dc_b + 4, s10)
        tl.store(dc_b + 5, s11)
        tl.store(dc_b + 6, s12)
        tl.store(dc_b + 7, s13)
        tl.store(dc_b + 8, s20)
        tl.store(dc_b + 9, s21)
        tl.store(dc_b + 10, s22)
        tl.store(dc_b + 11, s23)
        tl.store(dc_b + 12, s30)
        tl.store(dc_b + 13, s31)
        tl.store(dc_b + 14, s32)
        tl.store(dc_b + 15, s33)


_CONFIGS = [
    triton.Config(
        {"BLOCK_H": 1024, "NS": 2, "BATCH_ST": 1, "BATCH_LD": 0},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_H": 1024, "NS": 2, "BATCH_ST": 1, "BATCH_LD": 1},
        num_warps=4,
        num_stages=2,
    ),
]

_kernel_auto = triton.autotune(configs=_CONFIGS, key=["H"])(_mhc_post_bwd_kernel)

_DIVISORS = (1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1)


def _pick_block(h: int, limit: int) -> int:
    for b in _DIVISORS:
        if b <= limit and h % b == 0:
            return b
    return 1


def mhc_post_bwd(x, residual, post_layer_mix, comb_res_mix, d_o):
    """Compute input, residual and mixing gradients for four-stream mHC."""
    if (
        residual.shape != (*x.shape[:2], 4, x.shape[-1])
        or post_layer_mix.shape != (*x.shape[:2], 4, 1)
        or comb_res_mix.shape != (*x.shape[:2], 4, 4)
    ):
        raise ValueError("expected four mHC streams per token")

    if d_o.shape != residual.shape:
        raise ValueError("output gradient must match residual shape")
    n0, n1, H = x.shape
    n = n0 * n1

    x = x.contiguous()
    residual = residual.contiguous()
    post_layer_mix = post_layer_mix.contiguous()
    comb_res_mix = comb_res_mix.contiguous()
    d_o = d_o.contiguous()

    d_x = torch.empty_like(x)
    d_residual = torch.empty_like(residual)
    d_post_layer_mix = torch.empty(
        post_layer_mix.shape, dtype=torch.float32, device=x.device
    )
    d_comb_res_mix = torch.empty(
        comb_res_mix.shape, dtype=torch.float32, device=x.device
    )

    args = (
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        d_o,
        d_x,
        d_residual,
        d_post_layer_mix,
        d_comb_res_mix,
    )
    if H % 1024 == 0:
        _kernel_auto[(n,)](*args, H)
    else:
        # H not divisible by 1024 (the 1280/2560 correctness shapes) needs an
        # exact divisor instead of a masked tail.
        _mhc_post_bwd_kernel[(n,)](
            *args, H, BLOCK_H=_pick_block(H, 256), NS=3, BATCH_ST=1, BATCH_LD=1
        )
    return d_x, d_residual, d_post_layer_mix, d_comb_res_mix

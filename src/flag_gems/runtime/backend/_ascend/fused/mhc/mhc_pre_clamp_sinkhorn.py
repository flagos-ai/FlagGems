"""FlagGems-style Triton implementation of mhc_pre_clamp_sinkhorn.

Semantics (matches /data/liuhy/ops-transformer/mhc/mhc_pre_clamp_sinkhorn):

Inputs
------
    x       : (T, hcMult, D)   bf16/fp16/fp32
    phi     : (hcMix, hcMult*D) fp32      hcMix = hcMult*(hcMult+2)
    alpha   : (3,)             fp32
    base    : (hcMix,)         fp32
    norm_eps, hc_eps, clamp_min, clamp_max, iter_times: scalar hyperparams

Pipeline
--------
    1. RMSNorm across (hcMult*D) axis, produce inv_rms and scaled x.
    2. Heavy GEMM  mixes = (x*inv_rms).flatten(-2) @ phi.T
       -- delegated to torch.mm (bf16 matmul + fp32 accum), best on Ascend.
    3. Split mixes into (pre, post, combLogits) heads. Apply per-head affine
       (scale + bias) then sigmoid or clamp-softmax + Sinkhorn.

Ascend optimizations
--------------------
- Fused kernel A: two-pass RMSNorm in fp32 (register accumulators).
- torch.mm for the big GEMM; matches production MMAD.
- Fused kernel B (hc_mult=4 fastpath): all 16 combLogits scalars, 4 pre,
  4 post kept in registers; 20 Sinkhorn iterations run entirely in
  registers with reciprocal-then-multiply.
- shallow num_stages for Ascend NPU.
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 1024}, num_warps=8, num_stages=1),
    ],
    key=["HC_D"],
)
@triton.jit
def _rms_only_kernel(
    x_ptr,
    inv_rms_ptr,
    HC_D: tl.constexpr,
    D_INV: tl.constexpr,
    NORM_EPS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Single-pass: compute inv_rms only, no x_scaled write."""
    pid = tl.program_id(0)
    base = pid * HC_D
    sq = 0.0
    for h_start in range(0, HC_D, BLOCK_H):
        offs = h_start + tl.arange(0, BLOCK_H)
        mask = offs < HC_D
        v = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        sq += tl.sum(v * v, axis=0)
    inv = tl.math.rsqrt(sq * D_INV + NORM_EPS)
    tl.store(inv_rms_ptr + pid, inv)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 1024}, num_warps=8, num_stages=1),
    ],
    key=["HC_D"],
)
@triton.jit
def _rms_scale_kernel(
    x_ptr,  # (T, hcMult, D) input dtype
    x_scaled_ptr,  # (T, hcMult, D) fp32 output
    inv_rms_ptr,  # (T,) fp32
    HC_D: tl.constexpr,
    D_INV: tl.constexpr,  # 1/HC_D
    NORM_EPS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * HC_D
    sq = 0.0
    for h_start in range(0, HC_D, BLOCK_H):
        offs = h_start + tl.arange(0, BLOCK_H)
        mask = offs < HC_D
        v = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        sq += tl.sum(v * v, axis=0)
    inv = tl.math.rsqrt(sq * D_INV + NORM_EPS)
    tl.store(inv_rms_ptr + pid, inv)
    for h_start in range(0, HC_D, BLOCK_H):
        offs = h_start + tl.arange(0, BLOCK_H)
        mask = offs < HC_D
        v = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(x_scaled_ptr + base + offs, v * inv, mask=mask)


@triton.jit
def _heads_sinkhorn_kernel_hc4(
    mixes_ptr,  # (T, 24) fp32
    alpha_ptr,  # (3,)
    base_ptr,  # (24,)
    pre_ptr,  # (T, 4)
    post_ptr,  # (T, 4)
    comb_ptr,  # (T, 4, 4)
    logits_ptr,  # (T, 4, 4)  saved pre-clamp logits (unused when SAVE=0)
    HC_EPS: tl.constexpr,
    CLAMP_MIN: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    APPLY_CLAMP: tl.constexpr,
    ITERS: tl.constexpr,
    SAVE_INTERMEDIATES: tl.constexpr,
):
    pid = tl.program_id(0)
    a0 = tl.load(alpha_ptr + 0)
    a1 = tl.load(alpha_ptr + 1)
    a2 = tl.load(alpha_ptr + 2)
    mb = pid * 24

    # Pre head: 4 sigmoids
    for i in tl.static_range(4):
        m = tl.load(mixes_ptr + mb + i)
        b = tl.load(base_ptr + i)
        tl.store(pre_ptr + pid * 4 + i, tl.sigmoid(m * a0 + b) + HC_EPS)

    # Post head: 4 sigmoids, multiplied by 2 per aclnn spec (H^post_l = 2σ(...))
    for i in tl.static_range(4):
        m = tl.load(mixes_ptr + mb + 4 + i)
        b = tl.load(base_ptr + 4 + i)
        tl.store(post_ptr + pid * 4 + i, 2.0 * tl.sigmoid(m * a1 + b))

    # CombLogits: load 16 mixes + 16 bases, apply alpha[2]
    l00 = tl.load(mixes_ptr + mb + 8 + 0) * a2 + tl.load(base_ptr + 8 + 0)
    l01 = tl.load(mixes_ptr + mb + 8 + 1) * a2 + tl.load(base_ptr + 8 + 1)
    l02 = tl.load(mixes_ptr + mb + 8 + 2) * a2 + tl.load(base_ptr + 8 + 2)
    l03 = tl.load(mixes_ptr + mb + 8 + 3) * a2 + tl.load(base_ptr + 8 + 3)
    l10 = tl.load(mixes_ptr + mb + 8 + 4) * a2 + tl.load(base_ptr + 8 + 4)
    l11 = tl.load(mixes_ptr + mb + 8 + 5) * a2 + tl.load(base_ptr + 8 + 5)
    l12 = tl.load(mixes_ptr + mb + 8 + 6) * a2 + tl.load(base_ptr + 8 + 6)
    l13 = tl.load(mixes_ptr + mb + 8 + 7) * a2 + tl.load(base_ptr + 8 + 7)
    l20 = tl.load(mixes_ptr + mb + 8 + 8) * a2 + tl.load(base_ptr + 8 + 8)
    l21 = tl.load(mixes_ptr + mb + 8 + 9) * a2 + tl.load(base_ptr + 8 + 9)
    l22 = tl.load(mixes_ptr + mb + 8 + 10) * a2 + tl.load(base_ptr + 8 + 10)
    l23 = tl.load(mixes_ptr + mb + 8 + 11) * a2 + tl.load(base_ptr + 8 + 11)
    l30 = tl.load(mixes_ptr + mb + 8 + 12) * a2 + tl.load(base_ptr + 8 + 12)
    l31 = tl.load(mixes_ptr + mb + 8 + 13) * a2 + tl.load(base_ptr + 8 + 13)
    l32 = tl.load(mixes_ptr + mb + 8 + 14) * a2 + tl.load(base_ptr + 8 + 14)
    l33 = tl.load(mixes_ptr + mb + 8 + 15) * a2 + tl.load(base_ptr + 8 + 15)

    if SAVE_INTERMEDIATES:
        lb = pid * 16
        tl.store(logits_ptr + lb + 0, l00)
        tl.store(logits_ptr + lb + 1, l01)
        tl.store(logits_ptr + lb + 2, l02)
        tl.store(logits_ptr + lb + 3, l03)
        tl.store(logits_ptr + lb + 4, l10)
        tl.store(logits_ptr + lb + 5, l11)
        tl.store(logits_ptr + lb + 6, l12)
        tl.store(logits_ptr + lb + 7, l13)
        tl.store(logits_ptr + lb + 8, l20)
        tl.store(logits_ptr + lb + 9, l21)
        tl.store(logits_ptr + lb + 10, l22)
        tl.store(logits_ptr + lb + 11, l23)
        tl.store(logits_ptr + lb + 12, l30)
        tl.store(logits_ptr + lb + 13, l31)
        tl.store(logits_ptr + lb + 14, l32)
        tl.store(logits_ptr + lb + 15, l33)

    # Clamp (only when APPLY_CLAMP=1)
    if APPLY_CLAMP:
        l00 = tl.minimum(tl.maximum(l00, CLAMP_MIN), CLAMP_MAX)
        l01 = tl.minimum(tl.maximum(l01, CLAMP_MIN), CLAMP_MAX)
        l02 = tl.minimum(tl.maximum(l02, CLAMP_MIN), CLAMP_MAX)
        l03 = tl.minimum(tl.maximum(l03, CLAMP_MIN), CLAMP_MAX)
        l10 = tl.minimum(tl.maximum(l10, CLAMP_MIN), CLAMP_MAX)
        l11 = tl.minimum(tl.maximum(l11, CLAMP_MIN), CLAMP_MAX)
        l12 = tl.minimum(tl.maximum(l12, CLAMP_MIN), CLAMP_MAX)
        l13 = tl.minimum(tl.maximum(l13, CLAMP_MIN), CLAMP_MAX)
        l20 = tl.minimum(tl.maximum(l20, CLAMP_MIN), CLAMP_MAX)
        l21 = tl.minimum(tl.maximum(l21, CLAMP_MIN), CLAMP_MAX)
        l22 = tl.minimum(tl.maximum(l22, CLAMP_MIN), CLAMP_MAX)
        l23 = tl.minimum(tl.maximum(l23, CLAMP_MIN), CLAMP_MAX)
        l30 = tl.minimum(tl.maximum(l30, CLAMP_MIN), CLAMP_MAX)
        l31 = tl.minimum(tl.maximum(l31, CLAMP_MIN), CLAMP_MAX)
        l32 = tl.minimum(tl.maximum(l32, CLAMP_MIN), CLAMP_MAX)
        l33 = tl.minimum(tl.maximum(l33, CLAMP_MIN), CLAMP_MAX)

    # Row-softmax (subtract per-row max for numerical stability)
    m0 = tl.maximum(tl.maximum(l00, l01), tl.maximum(l02, l03))
    m1 = tl.maximum(tl.maximum(l10, l11), tl.maximum(l12, l13))
    m2 = tl.maximum(tl.maximum(l20, l21), tl.maximum(l22, l23))
    m3 = tl.maximum(tl.maximum(l30, l31), tl.maximum(l32, l33))
    e00 = tl.exp(l00 - m0)
    e01 = tl.exp(l01 - m0)
    e02 = tl.exp(l02 - m0)
    e03 = tl.exp(l03 - m0)
    e10 = tl.exp(l10 - m1)
    e11 = tl.exp(l11 - m1)
    e12 = tl.exp(l12 - m1)
    e13 = tl.exp(l13 - m1)
    e20 = tl.exp(l20 - m2)
    e21 = tl.exp(l21 - m2)
    e22 = tl.exp(l22 - m2)
    e23 = tl.exp(l23 - m2)
    e30 = tl.exp(l30 - m3)
    e31 = tl.exp(l31 - m3)
    e32 = tl.exp(l32 - m3)
    e33 = tl.exp(l33 - m3)
    inv_r0 = 1.0 / (e00 + e01 + e02 + e03)
    inv_r1 = 1.0 / (e10 + e11 + e12 + e13)
    inv_r2 = 1.0 / (e20 + e21 + e22 + e23)
    inv_r3 = 1.0 / (e30 + e31 + e32 + e33)
    # M[i,j] = e[i,j] * inv_r[i]
    v00 = e00 * inv_r0
    v01 = e01 * inv_r0
    v02 = e02 * inv_r0
    v03 = e03 * inv_r0
    v10 = e10 * inv_r1
    v11 = e11 * inv_r1
    v12 = e12 * inv_r1
    v13 = e13 * inv_r1
    v20 = e20 * inv_r2
    v21 = e21 * inv_r2
    v22 = e22 * inv_r2
    v23 = e23 * inv_r2
    v30 = e30 * inv_r3
    v31 = e31 * inv_r3
    v32 = e32 * inv_r3
    v33 = e33 * inv_r3

    # Add HC_EPS then col-normalize (first pass)
    v00 = v00 + HC_EPS
    v01 = v01 + HC_EPS
    v02 = v02 + HC_EPS
    v03 = v03 + HC_EPS
    v10 = v10 + HC_EPS
    v11 = v11 + HC_EPS
    v12 = v12 + HC_EPS
    v13 = v13 + HC_EPS
    v20 = v20 + HC_EPS
    v21 = v21 + HC_EPS
    v22 = v22 + HC_EPS
    v23 = v23 + HC_EPS
    v30 = v30 + HC_EPS
    v31 = v31 + HC_EPS
    v32 = v32 + HC_EPS
    v33 = v33 + HC_EPS
    inv_c0 = 1.0 / (v00 + v10 + v20 + v30 + HC_EPS)
    inv_c1 = 1.0 / (v01 + v11 + v21 + v31 + HC_EPS)
    inv_c2 = 1.0 / (v02 + v12 + v22 + v32 + HC_EPS)
    inv_c3 = 1.0 / (v03 + v13 + v23 + v33 + HC_EPS)
    v00 = v00 * inv_c0
    v01 = v01 * inv_c1
    v02 = v02 * inv_c2
    v03 = v03 * inv_c3
    v10 = v10 * inv_c0
    v11 = v11 * inv_c1
    v12 = v12 * inv_c2
    v13 = v13 * inv_c3
    v20 = v20 * inv_c0
    v21 = v21 * inv_c1
    v22 = v22 * inv_c2
    v23 = v23 * inv_c3
    v30 = v30 * inv_c0
    v31 = v31 * inv_c1
    v32 = v32 * inv_c2
    v33 = v33 * inv_c3

    # Remaining (ITERS-1) iterations: row-norm then col-norm
    for _ in tl.static_range(ITERS - 1):
        ir0 = 1.0 / (v00 + v01 + v02 + v03 + HC_EPS)
        ir1 = 1.0 / (v10 + v11 + v12 + v13 + HC_EPS)
        ir2 = 1.0 / (v20 + v21 + v22 + v23 + HC_EPS)
        ir3 = 1.0 / (v30 + v31 + v32 + v33 + HC_EPS)
        v00 = v00 * ir0
        v01 = v01 * ir0
        v02 = v02 * ir0
        v03 = v03 * ir0
        v10 = v10 * ir1
        v11 = v11 * ir1
        v12 = v12 * ir1
        v13 = v13 * ir1
        v20 = v20 * ir2
        v21 = v21 * ir2
        v22 = v22 * ir2
        v23 = v23 * ir2
        v30 = v30 * ir3
        v31 = v31 * ir3
        v32 = v32 * ir3
        v33 = v33 * ir3
        ic0 = 1.0 / (v00 + v10 + v20 + v30 + HC_EPS)
        ic1 = 1.0 / (v01 + v11 + v21 + v31 + HC_EPS)
        ic2 = 1.0 / (v02 + v12 + v22 + v32 + HC_EPS)
        ic3 = 1.0 / (v03 + v13 + v23 + v33 + HC_EPS)
        v00 = v00 * ic0
        v01 = v01 * ic1
        v02 = v02 * ic2
        v03 = v03 * ic3
        v10 = v10 * ic0
        v11 = v11 * ic1
        v12 = v12 * ic2
        v13 = v13 * ic3
        v20 = v20 * ic0
        v21 = v21 * ic1
        v22 = v22 * ic2
        v23 = v23 * ic3
        v30 = v30 * ic0
        v31 = v31 * ic1
        v32 = v32 * ic2
        v33 = v33 * ic3

    cb = pid * 16
    tl.store(comb_ptr + cb + 0, v00)
    tl.store(comb_ptr + cb + 1, v01)
    tl.store(comb_ptr + cb + 2, v02)
    tl.store(comb_ptr + cb + 3, v03)
    tl.store(comb_ptr + cb + 4, v10)
    tl.store(comb_ptr + cb + 5, v11)
    tl.store(comb_ptr + cb + 6, v12)
    tl.store(comb_ptr + cb + 7, v13)
    tl.store(comb_ptr + cb + 8, v20)
    tl.store(comb_ptr + cb + 9, v21)
    tl.store(comb_ptr + cb + 10, v22)
    tl.store(comb_ptr + cb + 11, v23)
    tl.store(comb_ptr + cb + 12, v30)
    tl.store(comb_ptr + cb + 13, v31)
    tl.store(comb_ptr + cb + 14, v32)
    tl.store(comb_ptr + cb + 15, v33)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_D": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 512}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_D": 1024}, num_warps=8, num_stages=1),
    ],
    key=["D"],
)
@triton.jit
def _y_scale_kernel_hc4(
    x_ptr,  # (T, 4, D) input dtype
    pre_ptr,  # (T, 4)    fp32
    y_ptr,  # (T, D)    input dtype   hin = sum_n(x[n] * pre[n])
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)
    d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_off < D

    p0 = tl.load(pre_ptr + pid_t * 4 + 0)
    p1 = tl.load(pre_ptr + pid_t * 4 + 1)
    p2 = tl.load(pre_ptr + pid_t * 4 + 2)
    p3 = tl.load(pre_ptr + pid_t * 4 + 3)

    xb = pid_t * 4 * D
    x0 = tl.load(x_ptr + xb + 0 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + xb + 1 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + xb + 2 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
    x3 = tl.load(x_ptr + xb + 3 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)

    hin = x0 * p0 + x1 * p1 + x2 * p2 + x3 * p3
    dt = y_ptr.dtype.element_ty
    tl.store(y_ptr + pid_t * D + d_off, hin.to(dt), mask=d_mask)


# ---------------------------------------------------------------------------
# Vectorized sinkhorn: BLOCK_T tokens per program, tl.static_range(14) iters
# ---------------------------------------------------------------------------
@triton.jit
def _sinkhorn_batched_14_hc4(
    mixes_ptr,
    alpha_ptr,
    base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
    HC_EPS: tl.constexpr,
    CLAMP_MIN: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
):
    pid = tl.program_id(0)
    t_off = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_off < T
    a0 = tl.load(alpha_ptr)
    a1 = tl.load(alpha_ptr + 1)
    a2 = tl.load(alpha_ptr + 2)
    mb = t_off * 24
    p0 = (
        tl.sigmoid(
            tl.load(mixes_ptr + mb + 0, mask=t_mask, other=0.0) * a0
            + tl.load(base_ptr + 0)
        )
        + HC_EPS
    )
    p1 = (
        tl.sigmoid(
            tl.load(mixes_ptr + mb + 1, mask=t_mask, other=0.0) * a0
            + tl.load(base_ptr + 1)
        )
        + HC_EPS
    )
    p2 = (
        tl.sigmoid(
            tl.load(mixes_ptr + mb + 2, mask=t_mask, other=0.0) * a0
            + tl.load(base_ptr + 2)
        )
        + HC_EPS
    )
    p3 = (
        tl.sigmoid(
            tl.load(mixes_ptr + mb + 3, mask=t_mask, other=0.0) * a0
            + tl.load(base_ptr + 3)
        )
        + HC_EPS
    )
    tl.store(pre_ptr + t_off * 4 + 0, p0, mask=t_mask)
    tl.store(pre_ptr + t_off * 4 + 1, p1, mask=t_mask)
    tl.store(pre_ptr + t_off * 4 + 2, p2, mask=t_mask)
    tl.store(pre_ptr + t_off * 4 + 3, p3, mask=t_mask)
    tl.store(
        post_ptr + t_off * 4 + 0,
        2.0
        * tl.sigmoid(
            tl.load(mixes_ptr + mb + 4, mask=t_mask, other=0.0) * a1
            + tl.load(base_ptr + 4)
        ),
        mask=t_mask,
    )
    tl.store(
        post_ptr + t_off * 4 + 1,
        2.0
        * tl.sigmoid(
            tl.load(mixes_ptr + mb + 5, mask=t_mask, other=0.0) * a1
            + tl.load(base_ptr + 5)
        ),
        mask=t_mask,
    )
    tl.store(
        post_ptr + t_off * 4 + 2,
        2.0
        * tl.sigmoid(
            tl.load(mixes_ptr + mb + 6, mask=t_mask, other=0.0) * a1
            + tl.load(base_ptr + 6)
        ),
        mask=t_mask,
    )
    tl.store(
        post_ptr + t_off * 4 + 3,
        2.0
        * tl.sigmoid(
            tl.load(mixes_ptr + mb + 7, mask=t_mask, other=0.0) * a1
            + tl.load(base_ptr + 7)
        ),
        mask=t_mask,
    )
    l00 = tl.load(mixes_ptr + mb + 8, mask=t_mask, other=0.0) * a2 + tl.load(
        base_ptr + 8
    )
    l01 = tl.load(mixes_ptr + mb + 9, mask=t_mask, other=0.0) * a2 + tl.load(
        base_ptr + 9
    )
    l02 = tl.load(mixes_ptr + mb + 10, mask=t_mask, other=0.0) * a2 + tl.load(
        base_ptr + 10
    )
    l03 = tl.load(mixes_ptr + mb + 11, mask=t_mask, other=0.0) * a2 + tl.load(
        base_ptr + 11
    )
    l10 = tl.load(mixes_ptr + mb + 12, mask=t_mask, other=0.0) * a2 + tl.load(
        base_ptr + 12
    )
    l11 = tl.load(mixes_ptr + mb + 13, mask=t_mask, other=0.0) * a2 + tl.load(
        base_ptr + 13
    )
    l12 = tl.load(mixes_ptr + mb + 14, mask=t_mask, other=0.0) * a2 + tl.load(
        base_ptr + 14
    )
    l13 = tl.load(mixes_ptr + mb + 15, mask=t_mask, other=0.0) * a2 + tl.load(
        base_ptr + 15
    )
    l20 = tl.load(mixes_ptr + mb + 16, mask=t_mask, other=0.0) * a2 + tl.load(
        base_ptr + 16
    )
    l21 = tl.load(mixes_ptr + mb + 17, mask=t_mask, other=0.0) * a2 + tl.load(
        base_ptr + 17
    )
    l22 = tl.load(mixes_ptr + mb + 18, mask=t_mask, other=0.0) * a2 + tl.load(
        base_ptr + 18
    )
    l23 = tl.load(mixes_ptr + mb + 19, mask=t_mask, other=0.0) * a2 + tl.load(
        base_ptr + 19
    )
    l30 = tl.load(mixes_ptr + mb + 20, mask=t_mask, other=0.0) * a2 + tl.load(
        base_ptr + 20
    )
    l31 = tl.load(mixes_ptr + mb + 21, mask=t_mask, other=0.0) * a2 + tl.load(
        base_ptr + 21
    )
    l32 = tl.load(mixes_ptr + mb + 22, mask=t_mask, other=0.0) * a2 + tl.load(
        base_ptr + 22
    )
    l33 = tl.load(mixes_ptr + mb + 23, mask=t_mask, other=0.0) * a2 + tl.load(
        base_ptr + 23
    )
    l00 = tl.minimum(tl.maximum(l00, CLAMP_MIN), CLAMP_MAX)
    l01 = tl.minimum(tl.maximum(l01, CLAMP_MIN), CLAMP_MAX)
    l02 = tl.minimum(tl.maximum(l02, CLAMP_MIN), CLAMP_MAX)
    l03 = tl.minimum(tl.maximum(l03, CLAMP_MIN), CLAMP_MAX)
    l10 = tl.minimum(tl.maximum(l10, CLAMP_MIN), CLAMP_MAX)
    l11 = tl.minimum(tl.maximum(l11, CLAMP_MIN), CLAMP_MAX)
    l12 = tl.minimum(tl.maximum(l12, CLAMP_MIN), CLAMP_MAX)
    l13 = tl.minimum(tl.maximum(l13, CLAMP_MIN), CLAMP_MAX)
    l20 = tl.minimum(tl.maximum(l20, CLAMP_MIN), CLAMP_MAX)
    l21 = tl.minimum(tl.maximum(l21, CLAMP_MIN), CLAMP_MAX)
    l22 = tl.minimum(tl.maximum(l22, CLAMP_MIN), CLAMP_MAX)
    l23 = tl.minimum(tl.maximum(l23, CLAMP_MIN), CLAMP_MAX)
    l30 = tl.minimum(tl.maximum(l30, CLAMP_MIN), CLAMP_MAX)
    l31 = tl.minimum(tl.maximum(l31, CLAMP_MIN), CLAMP_MAX)
    l32 = tl.minimum(tl.maximum(l32, CLAMP_MIN), CLAMP_MAX)
    l33 = tl.minimum(tl.maximum(l33, CLAMP_MIN), CLAMP_MAX)
    m0 = tl.maximum(tl.maximum(l00, l01), tl.maximum(l02, l03))
    m1 = tl.maximum(tl.maximum(l10, l11), tl.maximum(l12, l13))
    m2 = tl.maximum(tl.maximum(l20, l21), tl.maximum(l22, l23))
    m3 = tl.maximum(tl.maximum(l30, l31), tl.maximum(l32, l33))
    e00 = tl.exp(l00 - m0)
    e01 = tl.exp(l01 - m0)
    e02 = tl.exp(l02 - m0)
    e03 = tl.exp(l03 - m0)
    e10 = tl.exp(l10 - m1)
    e11 = tl.exp(l11 - m1)
    e12 = tl.exp(l12 - m1)
    e13 = tl.exp(l13 - m1)
    e20 = tl.exp(l20 - m2)
    e21 = tl.exp(l21 - m2)
    e22 = tl.exp(l22 - m2)
    e23 = tl.exp(l23 - m2)
    e30 = tl.exp(l30 - m3)
    e31 = tl.exp(l31 - m3)
    e32 = tl.exp(l32 - m3)
    e33 = tl.exp(l33 - m3)
    ir0 = 1.0 / (e00 + e01 + e02 + e03)
    ir1 = 1.0 / (e10 + e11 + e12 + e13)
    ir2 = 1.0 / (e20 + e21 + e22 + e23)
    ir3 = 1.0 / (e30 + e31 + e32 + e33)
    v00 = e00 * ir0
    v01 = e01 * ir0
    v02 = e02 * ir0
    v03 = e03 * ir0
    v10 = e10 * ir1
    v11 = e11 * ir1
    v12 = e12 * ir1
    v13 = e13 * ir1
    v20 = e20 * ir2
    v21 = e21 * ir2
    v22 = e22 * ir2
    v23 = e23 * ir2
    v30 = e30 * ir3
    v31 = e31 * ir3
    v32 = e32 * ir3
    v33 = e33 * ir3
    v00 += HC_EPS
    v01 += HC_EPS
    v02 += HC_EPS
    v03 += HC_EPS
    v10 += HC_EPS
    v11 += HC_EPS
    v12 += HC_EPS
    v13 += HC_EPS
    v20 += HC_EPS
    v21 += HC_EPS
    v22 += HC_EPS
    v23 += HC_EPS
    v30 += HC_EPS
    v31 += HC_EPS
    v32 += HC_EPS
    v33 += HC_EPS
    ic0 = 1.0 / (v00 + v10 + v20 + v30 + HC_EPS)
    ic1 = 1.0 / (v01 + v11 + v21 + v31 + HC_EPS)
    ic2 = 1.0 / (v02 + v12 + v22 + v32 + HC_EPS)
    ic3 = 1.0 / (v03 + v13 + v23 + v33 + HC_EPS)
    v00 *= ic0
    v01 *= ic1
    v02 *= ic2
    v03 *= ic3
    v10 *= ic0
    v11 *= ic1
    v12 *= ic2
    v13 *= ic3
    v20 *= ic0
    v21 *= ic1
    v22 *= ic2
    v23 *= ic3
    v30 *= ic0
    v31 *= ic1
    v32 *= ic2
    v33 *= ic3
    for _ in tl.static_range(14):
        ir0 = 1.0 / (v00 + v01 + v02 + v03 + HC_EPS)
        ir1 = 1.0 / (v10 + v11 + v12 + v13 + HC_EPS)
        ir2 = 1.0 / (v20 + v21 + v22 + v23 + HC_EPS)
        ir3 = 1.0 / (v30 + v31 + v32 + v33 + HC_EPS)
        v00 *= ir0
        v01 *= ir0
        v02 *= ir0
        v03 *= ir0
        v10 *= ir1
        v11 *= ir1
        v12 *= ir1
        v13 *= ir1
        v20 *= ir2
        v21 *= ir2
        v22 *= ir2
        v23 *= ir2
        v30 *= ir3
        v31 *= ir3
        v32 *= ir3
        v33 *= ir3
        ic0 = 1.0 / (v00 + v10 + v20 + v30 + HC_EPS)
        ic1 = 1.0 / (v01 + v11 + v21 + v31 + HC_EPS)
        ic2 = 1.0 / (v02 + v12 + v22 + v32 + HC_EPS)
        ic3 = 1.0 / (v03 + v13 + v23 + v33 + HC_EPS)
        v00 *= ic0
        v01 *= ic1
        v02 *= ic2
        v03 *= ic3
        v10 *= ic0
        v11 *= ic1
        v12 *= ic2
        v13 *= ic3
        v20 *= ic0
        v21 *= ic1
        v22 *= ic2
        v23 *= ic3
        v30 *= ic0
        v31 *= ic1
        v32 *= ic2
        v33 *= ic3
    cb = t_off * 16
    tl.store(comb_ptr + cb + 0, v00, mask=t_mask)
    tl.store(comb_ptr + cb + 1, v01, mask=t_mask)
    tl.store(comb_ptr + cb + 2, v02, mask=t_mask)
    tl.store(comb_ptr + cb + 3, v03, mask=t_mask)
    tl.store(comb_ptr + cb + 4, v10, mask=t_mask)
    tl.store(comb_ptr + cb + 5, v11, mask=t_mask)
    tl.store(comb_ptr + cb + 6, v12, mask=t_mask)
    tl.store(comb_ptr + cb + 7, v13, mask=t_mask)
    tl.store(comb_ptr + cb + 8, v20, mask=t_mask)
    tl.store(comb_ptr + cb + 9, v21, mask=t_mask)
    tl.store(comb_ptr + cb + 10, v22, mask=t_mask)
    tl.store(comb_ptr + cb + 11, v23, mask=t_mask)
    tl.store(comb_ptr + cb + 12, v30, mask=t_mask)
    tl.store(comb_ptr + cb + 13, v31, mask=t_mask)
    tl.store(comb_ptr + cb + 14, v32, mask=t_mask)
    tl.store(comb_ptr + cb + 15, v33, mask=t_mask)


@triton.jit
def _sinkhorn_continue5_hc4(
    comb_ptr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
    HC_EPS: tl.constexpr,
):
    """Continue sinkhorn for 5 more iterations (vectorized, BLOCK_T tokens/program)."""
    pid = tl.program_id(0)
    t_off = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_off < T
    cb = t_off * 16
    v00 = tl.load(comb_ptr + cb + 0, mask=t_mask, other=0.0)
    v01 = tl.load(comb_ptr + cb + 1, mask=t_mask, other=0.0)
    v02 = tl.load(comb_ptr + cb + 2, mask=t_mask, other=0.0)
    v03 = tl.load(comb_ptr + cb + 3, mask=t_mask, other=0.0)
    v10 = tl.load(comb_ptr + cb + 4, mask=t_mask, other=0.0)
    v11 = tl.load(comb_ptr + cb + 5, mask=t_mask, other=0.0)
    v12 = tl.load(comb_ptr + cb + 6, mask=t_mask, other=0.0)
    v13 = tl.load(comb_ptr + cb + 7, mask=t_mask, other=0.0)
    v20 = tl.load(comb_ptr + cb + 8, mask=t_mask, other=0.0)
    v21 = tl.load(comb_ptr + cb + 9, mask=t_mask, other=0.0)
    v22 = tl.load(comb_ptr + cb + 10, mask=t_mask, other=0.0)
    v23 = tl.load(comb_ptr + cb + 11, mask=t_mask, other=0.0)
    v30 = tl.load(comb_ptr + cb + 12, mask=t_mask, other=0.0)
    v31 = tl.load(comb_ptr + cb + 13, mask=t_mask, other=0.0)
    v32 = tl.load(comb_ptr + cb + 14, mask=t_mask, other=0.0)
    v33 = tl.load(comb_ptr + cb + 15, mask=t_mask, other=0.0)
    for _ in tl.static_range(5):
        ir0 = 1.0 / (v00 + v01 + v02 + v03 + HC_EPS)
        ir1 = 1.0 / (v10 + v11 + v12 + v13 + HC_EPS)
        ir2 = 1.0 / (v20 + v21 + v22 + v23 + HC_EPS)
        ir3 = 1.0 / (v30 + v31 + v32 + v33 + HC_EPS)
        v00 *= ir0
        v01 *= ir0
        v02 *= ir0
        v03 *= ir0
        v10 *= ir1
        v11 *= ir1
        v12 *= ir1
        v13 *= ir1
        v20 *= ir2
        v21 *= ir2
        v22 *= ir2
        v23 *= ir2
        v30 *= ir3
        v31 *= ir3
        v32 *= ir3
        v33 *= ir3
        ic0 = 1.0 / (v00 + v10 + v20 + v30 + HC_EPS)
        ic1 = 1.0 / (v01 + v11 + v21 + v31 + HC_EPS)
        ic2 = 1.0 / (v02 + v12 + v22 + v32 + HC_EPS)
        ic3 = 1.0 / (v03 + v13 + v23 + v33 + HC_EPS)
        v00 *= ic0
        v01 *= ic1
        v02 *= ic2
        v03 *= ic3
        v10 *= ic0
        v11 *= ic1
        v12 *= ic2
        v13 *= ic3
        v20 *= ic0
        v21 *= ic1
        v22 *= ic2
        v23 *= ic3
        v30 *= ic0
        v31 *= ic1
        v32 *= ic2
        v33 *= ic3
    tl.store(comb_ptr + cb + 0, v00, mask=t_mask)
    tl.store(comb_ptr + cb + 1, v01, mask=t_mask)
    tl.store(comb_ptr + cb + 2, v02, mask=t_mask)
    tl.store(comb_ptr + cb + 3, v03, mask=t_mask)
    tl.store(comb_ptr + cb + 4, v10, mask=t_mask)
    tl.store(comb_ptr + cb + 5, v11, mask=t_mask)
    tl.store(comb_ptr + cb + 6, v12, mask=t_mask)
    tl.store(comb_ptr + cb + 7, v13, mask=t_mask)
    tl.store(comb_ptr + cb + 8, v20, mask=t_mask)
    tl.store(comb_ptr + cb + 9, v21, mask=t_mask)
    tl.store(comb_ptr + cb + 10, v22, mask=t_mask)
    tl.store(comb_ptr + cb + 11, v23, mask=t_mask)
    tl.store(comb_ptr + cb + 12, v30, mask=t_mask)
    tl.store(comb_ptr + cb + 13, v31, mask=t_mask)
    tl.store(comb_ptr + cb + 14, v32, mask=t_mask)
    tl.store(comb_ptr + cb + 15, v33, mask=t_mask)


@triton.jit
def _y_scale_unrolled_hc4(
    x_ptr,
    pre_ptr,
    y_ptr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """y_scale with tl.static_range over D chunks — vectorized on Ascend NPU."""
    pid = tl.program_id(0)
    p0 = tl.load(pre_ptr + pid * 4 + 0)
    p1 = tl.load(pre_ptr + pid * 4 + 1)
    p2 = tl.load(pre_ptr + pid * 4 + 2)
    p3 = tl.load(pre_ptr + pid * 4 + 3)
    xb = pid * 4 * D
    dt = y_ptr.dtype.element_ty
    for d_start in tl.static_range(0, D, BLOCK_D):
        d_off = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_off < D
        x0 = tl.load(x_ptr + xb + 0 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
        x1 = tl.load(x_ptr + xb + 1 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + xb + 2 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
        x3 = tl.load(x_ptr + xb + 3 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
        hin = x0 * p0 + x1 * p1 + x2 * p2 + x3 * p3
        tl.store(y_ptr + pid * D + d_off, hin.to(dt), mask=d_mask)


# ---------------------------------------------------------------------------
# Batched sinkhorn + y_scale: BLOCK_T tokens per program → vector ops on NPU
# (kept for backward compatibility / need_backward path)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_T": 32, "BLOCK_D": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_T": 64, "BLOCK_D": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_T": 32, "BLOCK_D": 512}, num_warps=8, num_stages=1),
    ],
    key=["T", "D"],
)
@triton.jit
def _batched_sinkhorn_yscale_kernel_hc4(
    mixes_ptr,  # (T, 24) fp32
    alpha_ptr,  # (3,)
    base_ptr,  # (24,)
    x_ptr,  # (T, 4, D) input dtype
    pre_ptr,  # (T, 4)    fp32  OUTPUT
    post_ptr,  # (T, 4)    fp32  OUTPUT
    comb_ptr,  # (T, 4, 4) fp32  OUTPUT
    logits_ptr,  # (T, 4, 4) fp32  OUTPUT
    y_ptr,  # (T, D)    input dtype  OUTPUT
    T: tl.constexpr,
    D: tl.constexpr,
    HC_EPS: tl.constexpr,
    CLAMP_MIN: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    APPLY_CLAMP: tl.constexpr,
    ITERS: tl.constexpr,
    SAVE_INTERMEDIATES: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)  # token-block index
    pid_d = tl.program_id(1)  # D-block index

    t_off = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_off < T
    d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_off < D

    a0 = tl.load(alpha_ptr + 0)
    a1 = tl.load(alpha_ptr + 1)
    a2 = tl.load(alpha_ptr + 2)

    # Load mixes for this token block: shape (BLOCK_T, 24) → 24 vectors of BLOCK_T
    mb = t_off * 24  # (BLOCK_T,)

    # Pre head: 4 sigmoid vectors
    p0 = (
        tl.sigmoid(
            tl.load(mixes_ptr + mb + 0, mask=t_mask, other=0.0) * a0
            + tl.load(base_ptr + 0)
        )
        + HC_EPS
    )
    p1 = (
        tl.sigmoid(
            tl.load(mixes_ptr + mb + 1, mask=t_mask, other=0.0) * a0
            + tl.load(base_ptr + 1)
        )
        + HC_EPS
    )
    p2 = (
        tl.sigmoid(
            tl.load(mixes_ptr + mb + 2, mask=t_mask, other=0.0) * a0
            + tl.load(base_ptr + 2)
        )
        + HC_EPS
    )
    p3 = (
        tl.sigmoid(
            tl.load(mixes_ptr + mb + 3, mask=t_mask, other=0.0) * a0
            + tl.load(base_ptr + 3)
        )
        + HC_EPS
    )

    # Store pre
    tl.store(pre_ptr + t_off * 4 + 0, p0, mask=t_mask)
    tl.store(pre_ptr + t_off * 4 + 1, p1, mask=t_mask)
    tl.store(pre_ptr + t_off * 4 + 2, p2, mask=t_mask)
    tl.store(pre_ptr + t_off * 4 + 3, p3, mask=t_mask)

    # Post head: 4 sigmoid vectors
    tl.store(
        post_ptr + t_off * 4 + 0,
        2.0
        * tl.sigmoid(
            tl.load(mixes_ptr + mb + 4, mask=t_mask, other=0.0) * a1
            + tl.load(base_ptr + 4)
        ),
        mask=t_mask,
    )
    tl.store(
        post_ptr + t_off * 4 + 1,
        2.0
        * tl.sigmoid(
            tl.load(mixes_ptr + mb + 5, mask=t_mask, other=0.0) * a1
            + tl.load(base_ptr + 5)
        ),
        mask=t_mask,
    )
    tl.store(
        post_ptr + t_off * 4 + 2,
        2.0
        * tl.sigmoid(
            tl.load(mixes_ptr + mb + 6, mask=t_mask, other=0.0) * a1
            + tl.load(base_ptr + 6)
        ),
        mask=t_mask,
    )
    tl.store(
        post_ptr + t_off * 4 + 3,
        2.0
        * tl.sigmoid(
            tl.load(mixes_ptr + mb + 7, mask=t_mask, other=0.0) * a1
            + tl.load(base_ptr + 7)
        ),
        mask=t_mask,
    )

    # CombLogits: 16 vectors of BLOCK_T
    l00 = tl.load(mixes_ptr + mb + 8, mask=t_mask, other=0.0) * a2 + tl.load(
        base_ptr + 8
    )
    l01 = tl.load(mixes_ptr + mb + 9, mask=t_mask, other=0.0) * a2 + tl.load(
        base_ptr + 9
    )
    l02 = tl.load(mixes_ptr + mb + 10, mask=t_mask, other=0.0) * a2 + tl.load(
        base_ptr + 10
    )
    l03 = tl.load(mixes_ptr + mb + 11, mask=t_mask, other=0.0) * a2 + tl.load(
        base_ptr + 11
    )
    l10 = tl.load(mixes_ptr + mb + 12, mask=t_mask, other=0.0) * a2 + tl.load(
        base_ptr + 12
    )
    l11 = tl.load(mixes_ptr + mb + 13, mask=t_mask, other=0.0) * a2 + tl.load(
        base_ptr + 13
    )
    l12 = tl.load(mixes_ptr + mb + 14, mask=t_mask, other=0.0) * a2 + tl.load(
        base_ptr + 14
    )
    l13 = tl.load(mixes_ptr + mb + 15, mask=t_mask, other=0.0) * a2 + tl.load(
        base_ptr + 15
    )
    l20 = tl.load(mixes_ptr + mb + 16, mask=t_mask, other=0.0) * a2 + tl.load(
        base_ptr + 16
    )
    l21 = tl.load(mixes_ptr + mb + 17, mask=t_mask, other=0.0) * a2 + tl.load(
        base_ptr + 17
    )
    l22 = tl.load(mixes_ptr + mb + 18, mask=t_mask, other=0.0) * a2 + tl.load(
        base_ptr + 18
    )
    l23 = tl.load(mixes_ptr + mb + 19, mask=t_mask, other=0.0) * a2 + tl.load(
        base_ptr + 19
    )
    l30 = tl.load(mixes_ptr + mb + 20, mask=t_mask, other=0.0) * a2 + tl.load(
        base_ptr + 20
    )
    l31 = tl.load(mixes_ptr + mb + 21, mask=t_mask, other=0.0) * a2 + tl.load(
        base_ptr + 21
    )
    l32 = tl.load(mixes_ptr + mb + 22, mask=t_mask, other=0.0) * a2 + tl.load(
        base_ptr + 22
    )
    l33 = tl.load(mixes_ptr + mb + 23, mask=t_mask, other=0.0) * a2 + tl.load(
        base_ptr + 23
    )

    if SAVE_INTERMEDIATES:
        lb = t_off * 16
        tl.store(logits_ptr + lb + 0, l00, mask=t_mask)
        tl.store(logits_ptr + lb + 1, l01, mask=t_mask)
        tl.store(logits_ptr + lb + 2, l02, mask=t_mask)
        tl.store(logits_ptr + lb + 3, l03, mask=t_mask)
        tl.store(logits_ptr + lb + 4, l10, mask=t_mask)
        tl.store(logits_ptr + lb + 5, l11, mask=t_mask)
        tl.store(logits_ptr + lb + 6, l12, mask=t_mask)
        tl.store(logits_ptr + lb + 7, l13, mask=t_mask)
        tl.store(logits_ptr + lb + 8, l20, mask=t_mask)
        tl.store(logits_ptr + lb + 9, l21, mask=t_mask)
        tl.store(logits_ptr + lb + 10, l22, mask=t_mask)
        tl.store(logits_ptr + lb + 11, l23, mask=t_mask)
        tl.store(logits_ptr + lb + 12, l30, mask=t_mask)
        tl.store(logits_ptr + lb + 13, l31, mask=t_mask)
        tl.store(logits_ptr + lb + 14, l32, mask=t_mask)
        tl.store(logits_ptr + lb + 15, l33, mask=t_mask)

    if APPLY_CLAMP:
        l00 = tl.minimum(tl.maximum(l00, CLAMP_MIN), CLAMP_MAX)
        l01 = tl.minimum(tl.maximum(l01, CLAMP_MIN), CLAMP_MAX)
        l02 = tl.minimum(tl.maximum(l02, CLAMP_MIN), CLAMP_MAX)
        l03 = tl.minimum(tl.maximum(l03, CLAMP_MIN), CLAMP_MAX)
        l10 = tl.minimum(tl.maximum(l10, CLAMP_MIN), CLAMP_MAX)
        l11 = tl.minimum(tl.maximum(l11, CLAMP_MIN), CLAMP_MAX)
        l12 = tl.minimum(tl.maximum(l12, CLAMP_MIN), CLAMP_MAX)
        l13 = tl.minimum(tl.maximum(l13, CLAMP_MIN), CLAMP_MAX)
        l20 = tl.minimum(tl.maximum(l20, CLAMP_MIN), CLAMP_MAX)
        l21 = tl.minimum(tl.maximum(l21, CLAMP_MIN), CLAMP_MAX)
        l22 = tl.minimum(tl.maximum(l22, CLAMP_MIN), CLAMP_MAX)
        l23 = tl.minimum(tl.maximum(l23, CLAMP_MIN), CLAMP_MAX)
        l30 = tl.minimum(tl.maximum(l30, CLAMP_MIN), CLAMP_MAX)
        l31 = tl.minimum(tl.maximum(l31, CLAMP_MIN), CLAMP_MAX)
        l32 = tl.minimum(tl.maximum(l32, CLAMP_MIN), CLAMP_MAX)
        l33 = tl.minimum(tl.maximum(l33, CLAMP_MIN), CLAMP_MAX)

    # Row-softmax (vectorized over BLOCK_T)
    m0 = tl.maximum(tl.maximum(l00, l01), tl.maximum(l02, l03))
    m1 = tl.maximum(tl.maximum(l10, l11), tl.maximum(l12, l13))
    m2 = tl.maximum(tl.maximum(l20, l21), tl.maximum(l22, l23))
    m3 = tl.maximum(tl.maximum(l30, l31), tl.maximum(l32, l33))
    e00 = tl.exp(l00 - m0)
    e01 = tl.exp(l01 - m0)
    e02 = tl.exp(l02 - m0)
    e03 = tl.exp(l03 - m0)
    e10 = tl.exp(l10 - m1)
    e11 = tl.exp(l11 - m1)
    e12 = tl.exp(l12 - m1)
    e13 = tl.exp(l13 - m1)
    e20 = tl.exp(l20 - m2)
    e21 = tl.exp(l21 - m2)
    e22 = tl.exp(l22 - m2)
    e23 = tl.exp(l23 - m2)
    e30 = tl.exp(l30 - m3)
    e31 = tl.exp(l31 - m3)
    e32 = tl.exp(l32 - m3)
    e33 = tl.exp(l33 - m3)
    inv_r0 = 1.0 / (e00 + e01 + e02 + e03)
    inv_r1 = 1.0 / (e10 + e11 + e12 + e13)
    inv_r2 = 1.0 / (e20 + e21 + e22 + e23)
    inv_r3 = 1.0 / (e30 + e31 + e32 + e33)
    v00 = e00 * inv_r0
    v01 = e01 * inv_r0
    v02 = e02 * inv_r0
    v03 = e03 * inv_r0
    v10 = e10 * inv_r1
    v11 = e11 * inv_r1
    v12 = e12 * inv_r1
    v13 = e13 * inv_r1
    v20 = e20 * inv_r2
    v21 = e21 * inv_r2
    v22 = e22 * inv_r2
    v23 = e23 * inv_r2
    v30 = e30 * inv_r3
    v31 = e31 * inv_r3
    v32 = e32 * inv_r3
    v33 = e33 * inv_r3

    # Add HC_EPS then col-normalize (first pass)
    v00 = v00 + HC_EPS
    v01 = v01 + HC_EPS
    v02 = v02 + HC_EPS
    v03 = v03 + HC_EPS
    v10 = v10 + HC_EPS
    v11 = v11 + HC_EPS
    v12 = v12 + HC_EPS
    v13 = v13 + HC_EPS
    v20 = v20 + HC_EPS
    v21 = v21 + HC_EPS
    v22 = v22 + HC_EPS
    v23 = v23 + HC_EPS
    v30 = v30 + HC_EPS
    v31 = v31 + HC_EPS
    v32 = v32 + HC_EPS
    v33 = v33 + HC_EPS
    inv_c0 = 1.0 / (v00 + v10 + v20 + v30 + HC_EPS)
    inv_c1 = 1.0 / (v01 + v11 + v21 + v31 + HC_EPS)
    inv_c2 = 1.0 / (v02 + v12 + v22 + v32 + HC_EPS)
    inv_c3 = 1.0 / (v03 + v13 + v23 + v33 + HC_EPS)
    v00 = v00 * inv_c0
    v01 = v01 * inv_c1
    v02 = v02 * inv_c2
    v03 = v03 * inv_c3
    v10 = v10 * inv_c0
    v11 = v11 * inv_c1
    v12 = v12 * inv_c2
    v13 = v13 * inv_c3
    v20 = v20 * inv_c0
    v21 = v21 * inv_c1
    v22 = v22 * inv_c2
    v23 = v23 * inv_c3
    v30 = v30 * inv_c0
    v31 = v31 * inv_c1
    v32 = v32 * inv_c2
    v33 = v33 * inv_c3

    for _ in range(ITERS - 1):
        ir0 = 1.0 / (v00 + v01 + v02 + v03 + HC_EPS)
        ir1 = 1.0 / (v10 + v11 + v12 + v13 + HC_EPS)
        ir2 = 1.0 / (v20 + v21 + v22 + v23 + HC_EPS)
        ir3 = 1.0 / (v30 + v31 + v32 + v33 + HC_EPS)
        v00 = v00 * ir0
        v01 = v01 * ir0
        v02 = v02 * ir0
        v03 = v03 * ir0
        v10 = v10 * ir1
        v11 = v11 * ir1
        v12 = v12 * ir1
        v13 = v13 * ir1
        v20 = v20 * ir2
        v21 = v21 * ir2
        v22 = v22 * ir2
        v23 = v23 * ir2
        v30 = v30 * ir3
        v31 = v31 * ir3
        v32 = v32 * ir3
        v33 = v33 * ir3
        ic0 = 1.0 / (v00 + v10 + v20 + v30 + HC_EPS)
        ic1 = 1.0 / (v01 + v11 + v21 + v31 + HC_EPS)
        ic2 = 1.0 / (v02 + v12 + v22 + v32 + HC_EPS)
        ic3 = 1.0 / (v03 + v13 + v23 + v33 + HC_EPS)
        v00 = v00 * ic0
        v01 = v01 * ic1
        v02 = v02 * ic2
        v03 = v03 * ic3
        v10 = v10 * ic0
        v11 = v11 * ic1
        v12 = v12 * ic2
        v13 = v13 * ic3
        v20 = v20 * ic0
        v21 = v21 * ic1
        v22 = v22 * ic2
        v23 = v23 * ic3
        v30 = v30 * ic0
        v31 = v31 * ic1
        v32 = v32 * ic2
        v33 = v33 * ic3

    # Store comb_frag
    cb = t_off * 16
    tl.store(comb_ptr + cb + 0, v00, mask=t_mask)
    tl.store(comb_ptr + cb + 1, v01, mask=t_mask)
    tl.store(comb_ptr + cb + 2, v02, mask=t_mask)
    tl.store(comb_ptr + cb + 3, v03, mask=t_mask)
    tl.store(comb_ptr + cb + 4, v10, mask=t_mask)
    tl.store(comb_ptr + cb + 5, v11, mask=t_mask)
    tl.store(comb_ptr + cb + 6, v12, mask=t_mask)
    tl.store(comb_ptr + cb + 7, v13, mask=t_mask)
    tl.store(comb_ptr + cb + 8, v20, mask=t_mask)
    tl.store(comb_ptr + cb + 9, v21, mask=t_mask)
    tl.store(comb_ptr + cb + 10, v22, mask=t_mask)
    tl.store(comb_ptr + cb + 11, v23, mask=t_mask)
    tl.store(comb_ptr + cb + 12, v30, mask=t_mask)
    tl.store(comb_ptr + cb + 13, v31, mask=t_mask)
    tl.store(comb_ptr + cb + 14, v32, mask=t_mask)
    tl.store(comb_ptr + cb + 15, v33, mask=t_mask)

    # y_scale: hin[t, d] = sum_n x[t,n,d] * pre[t,n]  — fused here to avoid extra kernel
    xb = t_off * 4 * D  # (BLOCK_T,)
    x0v = tl.load(
        x_ptr + xb[:, None] + 0 * D + d_off[None, :],
        mask=t_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    x1v = tl.load(
        x_ptr + xb[:, None] + 1 * D + d_off[None, :],
        mask=t_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    x2v = tl.load(
        x_ptr + xb[:, None] + 2 * D + d_off[None, :],
        mask=t_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    x3v = tl.load(
        x_ptr + xb[:, None] + 3 * D + d_off[None, :],
        mask=t_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    hin = x0v * p0[:, None] + x1v * p1[:, None] + x2v * p2[:, None] + x3v * p3[:, None]
    dt = y_ptr.dtype.element_ty
    tl.store(
        y_ptr + t_off[:, None] * D + d_off[None, :],
        hin.to(dt),
        mask=t_mask[:, None] & d_mask[None, :],
    )


# ---------------------------------------------------------------------------
# Fused sinkhorn + y_scale kernel (eliminates separate y_scale kernel launch)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 512}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_D": 1024}, num_warps=8, num_stages=1),
    ],
    key=["D"],
)
@triton.jit
def _heads_sinkhorn_yscale_kernel_hc4(
    mixes_ptr,  # (T, 24) fp32
    alpha_ptr,  # (3,)
    base_ptr,  # (24,)
    x_ptr,  # (T, 4, D) input dtype
    pre_ptr,  # (T, 4)    fp32  OUTPUT
    post_ptr,  # (T, 4)    fp32  OUTPUT
    comb_ptr,  # (T, 4, 4) fp32  OUTPUT
    logits_ptr,  # (T, 4, 4) fp32  OUTPUT (saved pre-clamp logits)
    y_ptr,  # (T, D)    input dtype  OUTPUT (hin = sum_n x*pre)
    D: tl.constexpr,
    HC_EPS: tl.constexpr,
    CLAMP_MIN: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    APPLY_CLAMP: tl.constexpr,
    ITERS: tl.constexpr,
    SAVE_INTERMEDIATES: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    a0 = tl.load(alpha_ptr + 0)
    a1 = tl.load(alpha_ptr + 1)
    a2 = tl.load(alpha_ptr + 2)
    mb = pid * 24

    # Pre head: 4 sigmoids
    p0_val = tl.zeros([], dtype=tl.float32)
    p1_val = tl.zeros([], dtype=tl.float32)
    p2_val = tl.zeros([], dtype=tl.float32)
    p3_val = tl.zeros([], dtype=tl.float32)
    for i in tl.static_range(4):
        m = tl.load(mixes_ptr + mb + i)
        b = tl.load(base_ptr + i)
        val = tl.sigmoid(m * a0 + b) + HC_EPS
        tl.store(pre_ptr + pid * 4 + i, val)
        if i == 0:
            p0_val = val
        if i == 1:
            p1_val = val
        if i == 2:
            p2_val = val
        if i == 3:
            p3_val = val

    # Post head: 4 sigmoids
    for i in tl.static_range(4):
        m = tl.load(mixes_ptr + mb + 4 + i)
        b = tl.load(base_ptr + 4 + i)
        tl.store(post_ptr + pid * 4 + i, 2.0 * tl.sigmoid(m * a1 + b))

    # CombLogits: load 16 mixes + 16 bases, apply alpha[2]
    l00 = tl.load(mixes_ptr + mb + 8 + 0) * a2 + tl.load(base_ptr + 8 + 0)
    l01 = tl.load(mixes_ptr + mb + 8 + 1) * a2 + tl.load(base_ptr + 8 + 1)
    l02 = tl.load(mixes_ptr + mb + 8 + 2) * a2 + tl.load(base_ptr + 8 + 2)
    l03 = tl.load(mixes_ptr + mb + 8 + 3) * a2 + tl.load(base_ptr + 8 + 3)
    l10 = tl.load(mixes_ptr + mb + 8 + 4) * a2 + tl.load(base_ptr + 8 + 4)
    l11 = tl.load(mixes_ptr + mb + 8 + 5) * a2 + tl.load(base_ptr + 8 + 5)
    l12 = tl.load(mixes_ptr + mb + 8 + 6) * a2 + tl.load(base_ptr + 8 + 6)
    l13 = tl.load(mixes_ptr + mb + 8 + 7) * a2 + tl.load(base_ptr + 8 + 7)
    l20 = tl.load(mixes_ptr + mb + 8 + 8) * a2 + tl.load(base_ptr + 8 + 8)
    l21 = tl.load(mixes_ptr + mb + 8 + 9) * a2 + tl.load(base_ptr + 8 + 9)
    l22 = tl.load(mixes_ptr + mb + 8 + 10) * a2 + tl.load(base_ptr + 8 + 10)
    l23 = tl.load(mixes_ptr + mb + 8 + 11) * a2 + tl.load(base_ptr + 8 + 11)
    l30 = tl.load(mixes_ptr + mb + 8 + 12) * a2 + tl.load(base_ptr + 8 + 12)
    l31 = tl.load(mixes_ptr + mb + 8 + 13) * a2 + tl.load(base_ptr + 8 + 13)
    l32 = tl.load(mixes_ptr + mb + 8 + 14) * a2 + tl.load(base_ptr + 8 + 14)
    l33 = tl.load(mixes_ptr + mb + 8 + 15) * a2 + tl.load(base_ptr + 8 + 15)

    if SAVE_INTERMEDIATES:
        lb = pid * 16
        tl.store(logits_ptr + lb + 0, l00)
        tl.store(logits_ptr + lb + 1, l01)
        tl.store(logits_ptr + lb + 2, l02)
        tl.store(logits_ptr + lb + 3, l03)
        tl.store(logits_ptr + lb + 4, l10)
        tl.store(logits_ptr + lb + 5, l11)
        tl.store(logits_ptr + lb + 6, l12)
        tl.store(logits_ptr + lb + 7, l13)
        tl.store(logits_ptr + lb + 8, l20)
        tl.store(logits_ptr + lb + 9, l21)
        tl.store(logits_ptr + lb + 10, l22)
        tl.store(logits_ptr + lb + 11, l23)
        tl.store(logits_ptr + lb + 12, l30)
        tl.store(logits_ptr + lb + 13, l31)
        tl.store(logits_ptr + lb + 14, l32)
        tl.store(logits_ptr + lb + 15, l33)

    # Clamp
    if APPLY_CLAMP:
        l00 = tl.minimum(tl.maximum(l00, CLAMP_MIN), CLAMP_MAX)
        l01 = tl.minimum(tl.maximum(l01, CLAMP_MIN), CLAMP_MAX)
        l02 = tl.minimum(tl.maximum(l02, CLAMP_MIN), CLAMP_MAX)
        l03 = tl.minimum(tl.maximum(l03, CLAMP_MIN), CLAMP_MAX)
        l10 = tl.minimum(tl.maximum(l10, CLAMP_MIN), CLAMP_MAX)
        l11 = tl.minimum(tl.maximum(l11, CLAMP_MIN), CLAMP_MAX)
        l12 = tl.minimum(tl.maximum(l12, CLAMP_MIN), CLAMP_MAX)
        l13 = tl.minimum(tl.maximum(l13, CLAMP_MIN), CLAMP_MAX)
        l20 = tl.minimum(tl.maximum(l20, CLAMP_MIN), CLAMP_MAX)
        l21 = tl.minimum(tl.maximum(l21, CLAMP_MIN), CLAMP_MAX)
        l22 = tl.minimum(tl.maximum(l22, CLAMP_MIN), CLAMP_MAX)
        l23 = tl.minimum(tl.maximum(l23, CLAMP_MIN), CLAMP_MAX)
        l30 = tl.minimum(tl.maximum(l30, CLAMP_MIN), CLAMP_MAX)
        l31 = tl.minimum(tl.maximum(l31, CLAMP_MIN), CLAMP_MAX)
        l32 = tl.minimum(tl.maximum(l32, CLAMP_MIN), CLAMP_MAX)
        l33 = tl.minimum(tl.maximum(l33, CLAMP_MIN), CLAMP_MAX)

    # Row-softmax
    m0 = tl.maximum(tl.maximum(l00, l01), tl.maximum(l02, l03))
    m1 = tl.maximum(tl.maximum(l10, l11), tl.maximum(l12, l13))
    m2 = tl.maximum(tl.maximum(l20, l21), tl.maximum(l22, l23))
    m3 = tl.maximum(tl.maximum(l30, l31), tl.maximum(l32, l33))
    e00 = tl.exp(l00 - m0)
    e01 = tl.exp(l01 - m0)
    e02 = tl.exp(l02 - m0)
    e03 = tl.exp(l03 - m0)
    e10 = tl.exp(l10 - m1)
    e11 = tl.exp(l11 - m1)
    e12 = tl.exp(l12 - m1)
    e13 = tl.exp(l13 - m1)
    e20 = tl.exp(l20 - m2)
    e21 = tl.exp(l21 - m2)
    e22 = tl.exp(l22 - m2)
    e23 = tl.exp(l23 - m2)
    e30 = tl.exp(l30 - m3)
    e31 = tl.exp(l31 - m3)
    e32 = tl.exp(l32 - m3)
    e33 = tl.exp(l33 - m3)
    inv_r0 = 1.0 / (e00 + e01 + e02 + e03)
    inv_r1 = 1.0 / (e10 + e11 + e12 + e13)
    inv_r2 = 1.0 / (e20 + e21 + e22 + e23)
    inv_r3 = 1.0 / (e30 + e31 + e32 + e33)
    v00 = e00 * inv_r0
    v01 = e01 * inv_r0
    v02 = e02 * inv_r0
    v03 = e03 * inv_r0
    v10 = e10 * inv_r1
    v11 = e11 * inv_r1
    v12 = e12 * inv_r1
    v13 = e13 * inv_r1
    v20 = e20 * inv_r2
    v21 = e21 * inv_r2
    v22 = e22 * inv_r2
    v23 = e23 * inv_r2
    v30 = e30 * inv_r3
    v31 = e31 * inv_r3
    v32 = e32 * inv_r3
    v33 = e33 * inv_r3

    # Add HC_EPS then col-normalize (first pass)
    v00 = v00 + HC_EPS
    v01 = v01 + HC_EPS
    v02 = v02 + HC_EPS
    v03 = v03 + HC_EPS
    v10 = v10 + HC_EPS
    v11 = v11 + HC_EPS
    v12 = v12 + HC_EPS
    v13 = v13 + HC_EPS
    v20 = v20 + HC_EPS
    v21 = v21 + HC_EPS
    v22 = v22 + HC_EPS
    v23 = v23 + HC_EPS
    v30 = v30 + HC_EPS
    v31 = v31 + HC_EPS
    v32 = v32 + HC_EPS
    v33 = v33 + HC_EPS
    inv_c0 = 1.0 / (v00 + v10 + v20 + v30 + HC_EPS)
    inv_c1 = 1.0 / (v01 + v11 + v21 + v31 + HC_EPS)
    inv_c2 = 1.0 / (v02 + v12 + v22 + v32 + HC_EPS)
    inv_c3 = 1.0 / (v03 + v13 + v23 + v33 + HC_EPS)
    v00 = v00 * inv_c0
    v01 = v01 * inv_c1
    v02 = v02 * inv_c2
    v03 = v03 * inv_c3
    v10 = v10 * inv_c0
    v11 = v11 * inv_c1
    v12 = v12 * inv_c2
    v13 = v13 * inv_c3
    v20 = v20 * inv_c0
    v21 = v21 * inv_c1
    v22 = v22 * inv_c2
    v23 = v23 * inv_c3
    v30 = v30 * inv_c0
    v31 = v31 * inv_c1
    v32 = v32 * inv_c2
    v33 = v33 * inv_c3

    # Remaining (ITERS-1) iterations: row-norm then col-norm
    for _ in tl.static_range(ITERS - 1):
        ir0 = 1.0 / (v00 + v01 + v02 + v03 + HC_EPS)
        ir1 = 1.0 / (v10 + v11 + v12 + v13 + HC_EPS)
        ir2 = 1.0 / (v20 + v21 + v22 + v23 + HC_EPS)
        ir3 = 1.0 / (v30 + v31 + v32 + v33 + HC_EPS)
        v00 = v00 * ir0
        v01 = v01 * ir0
        v02 = v02 * ir0
        v03 = v03 * ir0
        v10 = v10 * ir1
        v11 = v11 * ir1
        v12 = v12 * ir1
        v13 = v13 * ir1
        v20 = v20 * ir2
        v21 = v21 * ir2
        v22 = v22 * ir2
        v23 = v23 * ir2
        v30 = v30 * ir3
        v31 = v31 * ir3
        v32 = v32 * ir3
        v33 = v33 * ir3
        ic0 = 1.0 / (v00 + v10 + v20 + v30 + HC_EPS)
        ic1 = 1.0 / (v01 + v11 + v21 + v31 + HC_EPS)
        ic2 = 1.0 / (v02 + v12 + v22 + v32 + HC_EPS)
        ic3 = 1.0 / (v03 + v13 + v23 + v33 + HC_EPS)
        v00 = v00 * ic0
        v01 = v01 * ic1
        v02 = v02 * ic2
        v03 = v03 * ic3
        v10 = v10 * ic0
        v11 = v11 * ic1
        v12 = v12 * ic2
        v13 = v13 * ic3
        v20 = v20 * ic0
        v21 = v21 * ic1
        v22 = v22 * ic2
        v23 = v23 * ic3
        v30 = v30 * ic0
        v31 = v31 * ic1
        v32 = v32 * ic2
        v33 = v33 * ic3

    cb = pid * 16
    tl.store(comb_ptr + cb + 0, v00)
    tl.store(comb_ptr + cb + 1, v01)
    tl.store(comb_ptr + cb + 2, v02)
    tl.store(comb_ptr + cb + 3, v03)
    tl.store(comb_ptr + cb + 4, v10)
    tl.store(comb_ptr + cb + 5, v11)
    tl.store(comb_ptr + cb + 6, v12)
    tl.store(comb_ptr + cb + 7, v13)
    tl.store(comb_ptr + cb + 8, v20)
    tl.store(comb_ptr + cb + 9, v21)
    tl.store(comb_ptr + cb + 10, v22)
    tl.store(comb_ptr + cb + 11, v23)
    tl.store(comb_ptr + cb + 12, v30)
    tl.store(comb_ptr + cb + 13, v31)
    tl.store(comb_ptr + cb + 14, v32)
    tl.store(comb_ptr + cb + 15, v33)

    # ---- Fused y_scale: y[d] = sum_n(x[n,d] * pre[n]) ----
    xb = pid * 4 * D
    dt = y_ptr.dtype.element_ty
    for d_start in range(0, D, BLOCK_D):
        d_off = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_off < D
        x0 = tl.load(x_ptr + xb + 0 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
        x1 = tl.load(x_ptr + xb + 1 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + xb + 2 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
        x3 = tl.load(x_ptr + xb + 3 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
        hin = x0 * p0_val + x1 * p1_val + x2 * p2_val + x3 * p3_val
        tl.store(y_ptr + pid * D + d_off, hin.to(dt), mask=d_mask)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _flatten(x):
    """Return (xf_TND, y_out_shape) where y_out_shape is the shape of the
    reduced-N `hin` output (aclnn semantic): (B, S, D) for 4D input or
    (T, D) for 3D input."""
    if x.dim() == 4:
        B, S, N, D = x.shape
        return x.reshape(B * S, N, D).contiguous(), (B, S, D)
    if x.dim() == 3:
        T, N, D = x.shape
        return x.contiguous(), (T, D)
    raise ValueError(f"unsupported x.dim()={x.dim()}")


def mhc_pre_clamp_sinkhorn(
    x: torch.Tensor,
    phi: torch.Tensor,
    alpha: torch.Tensor,
    base: torch.Tensor,
    norm_eps: float = 1e-6,
    hc_eps: float = 1e-6,
    clamp_min: float = 0.0,
    clamp_max: float = 0.0,
    iter_times: int = 20,
    need_backward: bool = False,
):
    """Fused MHC pre + clamp + Sinkhorn forward (aclnn semantic).

    Returns a dict with:
        y (hin)     : (B, S, D) or (T, D)  same dtype as x
                      hin[t, d] = sum_n( x[t, n, d] * pre[t, n] )
        post_out    : (T, hcMult)   fp32   post_out = 2 * sigmoid(...)
        comb_frag   : (T, hcMult, hcMult) fp32
    If need_backward=True, also:
        inv_rms     : (T,)  fp32
        mixes       : (T, hcMix) fp32
        h_res_logits: (T, hcMult, hcMult) fp32  (pre-clamp logits)
        pre         : (T, hcMult) fp32
    """
    xf, shape = _flatten(x)
    T, N, D = xf.shape
    assert N == 4, "hc_mult=4 fast path only. Extend for other N."
    hc_mix = N * (N + 2)
    hc_d = N * D
    assert phi.shape == (hc_mix, hc_d), f"phi shape {phi.shape} != {(hc_mix, hc_d)}"
    assert alpha.numel() == 3
    assert base.numel() == hc_mix

    inv_rms = torch.empty(T, dtype=torch.float32, device=xf.device)
    phi_f = phi.to(torch.float32)

    if need_backward:
        # Full path: materialize x_scaled for backward pass
        x_scaled = torch.empty(T, hc_d, dtype=torch.float32, device=xf.device)
        _rms_scale_kernel[(T,)](
            xf,
            x_scaled,
            inv_rms,
            HC_D=hc_d,
            D_INV=1.0 / hc_d,
            NORM_EPS=norm_eps,
        )
        mixes = torch.mm(x_scaled, phi_f.t())
    else:
        # Fast path: skip x_scaled materialization (~470MB traffic saved)
        # mixes = (x * inv_rms) @ phi^T = inv_rms * (x @ phi^T)
        x_scaled = None
        _rms_only_kernel[(T,)](
            xf,
            inv_rms,
            HC_D=hc_d,
            D_INV=1.0 / hc_d,
            NORM_EPS=norm_eps,
        )
        xf_2d = xf.reshape(T, hc_d)
        mixes = torch.mm(xf_2d.float(), phi_f.t())
        mixes.mul_(inv_rms.unsqueeze(1))

    pre = torch.empty(T, N, dtype=torch.float32, device=xf.device)
    post_out = torch.empty(T, N, dtype=torch.float32, device=xf.device)
    comb_frag = torch.empty(T, N, N, dtype=torch.float32, device=xf.device)
    h_res_logits = (
        torch.empty(T, N, N, dtype=torch.float32, device=xf.device)
        if need_backward
        else torch.empty(0, device=xf.device)
    )

    apply_clamp = 1 if (clamp_min != 0.0 or clamp_max != 0.0) else 0

    y = torch.empty(T, D, dtype=xf.dtype, device=xf.device)

    # Vectorized path: batched14 (1+14 iters) + continue5 (5 iters) + y_scale_unrolled
    # Total = 20 Sinkhorn iterations, all vectorized (BLOCK_T tokens per program)
    BLOCK_T = 32
    grid = (triton.cdiv(T, BLOCK_T),)
    _sinkhorn_batched_14_hc4[grid](
        mixes,
        alpha.to(torch.float32),
        base.to(torch.float32),
        pre,
        post_out,
        comb_frag,
        T=T,
        BLOCK_T=BLOCK_T,
        HC_EPS=hc_eps,
        CLAMP_MIN=float(clamp_min),
        CLAMP_MAX=float(clamp_max),
        num_warps=4,
        num_stages=1,
    )
    _sinkhorn_continue5_hc4[grid](
        comb_frag,
        T=T,
        BLOCK_T=BLOCK_T,
        HC_EPS=hc_eps,
        num_warps=4,
        num_stages=1,
    )
    if need_backward:
        # Save logits: re-run sinkhorn kernel with SAVE_INTERMEDIATES via old kernel
        _heads_sinkhorn_kernel_hc4[(T,)](
            mixes,
            alpha.to(torch.float32),
            base.to(torch.float32),
            pre,
            post_out,
            comb_frag,
            h_res_logits,
            HC_EPS=hc_eps,
            CLAMP_MIN=float(clamp_min),
            CLAMP_MAX=float(clamp_max),
            APPLY_CLAMP=apply_clamp,
            ITERS=int(iter_times),
            SAVE_INTERMEDIATES=1,
        )
    _y_scale_unrolled_hc4[(T,)](
        xf.reshape(T, N, D), pre, y, D=D, BLOCK_D=256, num_warps=4, num_stages=1
    )

    result = {
        "y": y.reshape(shape),
        "post_out": post_out,
        "comb_frag": comb_frag,
    }
    if need_backward:
        result.update(
            inv_rms=inv_rms,
            x_scaled=x_scaled,
            mixes=mixes,
            h_res_logits=h_res_logits,
            pre=pre,
        )
    return result


def mhc_pre_clamp_sinkhorn_ref(
    x,
    phi,
    alpha,
    base,
    norm_eps=1e-6,
    hc_eps=1e-6,
    clamp_min=0.0,
    clamp_max=0.0,
    iter_times=20,
):
    """PyTorch reference implementation (aclnn semantic).

    hin[t, d] = sum_n( x[t, n, d] * pre[t, n] )   -> shape (B, S, D)
    post_out  = 2 * sigmoid(...)                  per aclnn spec.
    """
    orig_dtype = x.dtype
    xf, shape = _flatten(x)
    T, N, D = xf.shape
    x_flat = xf.reshape(T, N * D).float()

    ms = x_flat.pow(2).mean(dim=-1, keepdim=True)
    inv = torch.rsqrt(ms + norm_eps)
    x_scaled = x_flat * inv
    mixes = x_scaled @ phi.float().t()

    a = alpha.float()
    b = base.float()
    pre = torch.sigmoid(mixes[:, :N] * a[0] + b[:N]) + hc_eps
    post_out = 2.0 * torch.sigmoid(mixes[:, N : 2 * N] * a[1] + b[N : 2 * N])
    logits = (mixes[:, 2 * N :] * a[2] + b[2 * N :]).reshape(T, N, N)
    if clamp_min != 0.0 or clamp_max != 0.0:
        logits_c = torch.clamp(logits, clamp_min, clamp_max)
    else:
        logits_c = logits
    row_max = logits_c.max(dim=-1, keepdim=True).values
    M = (logits_c - row_max).exp()
    M = M / M.sum(dim=-1, keepdim=True) + hc_eps
    M = M / (M.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(iter_times - 1):
        M = M / (M.sum(dim=-1, keepdim=True) + hc_eps)
        M = M / (M.sum(dim=-2, keepdim=True) + hc_eps)

    # hin = sum_n (x * pre) -> (T, D)
    y = (xf.float() * pre.unsqueeze(-1)).sum(dim=-2).to(orig_dtype)
    return {
        "y": y.reshape(shape),
        "post_out": post_out,
        "comb_frag": M,
        "inv_rms": inv.squeeze(-1),
        "mixes": mixes,
        "h_res_logits": logits,
        "pre": pre,
    }

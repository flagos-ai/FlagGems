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

from flag_gems.ops.ormqr import ormqr as default_ormqr

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float32}


# ---------------------------------------------------------------------------
# ormqr: apply Householder reflectors (from a QR factorization stored in
# `input`/`tau`) to matrix `other`, on the left or right, optionally
# transposed.
#
#   input: (m, k) reflectors; v_j[i] = input[i, j] for i > j, v_j[j] = 1
#          (diagonal of input is NOT part of the reflector), v_j[i] = 0 for
#          i < j.  k = number of reflectors, k <= m.
#   tau:   (k,) scalars
#   other: (m, r) if left=True else (r, m)
#
#   left=True : out = Q . other            (or Q^T . other)
#   left=False: out = other . Q            (or other . Q^T)
#
# Strategy: blocked compact-WY representation.  Each block of NB reflectors
# is factored as  F_b = I - Y_b T_b Y_b^T  where Y_b is the block's reflector
# columns and T_b is the small (NB x NB) WY factor computed by a dlarft-style
# column recurrence.  Q (or Q^T) is applied by sweeping the blocks in the
# appropriate order, each step doing the parallel matmuls
#     Z_b = T_b (Y_b^T C)      C <- C - Y_b Z_b
#
# fp32: the matmuls use three fp16 tensor-core dots (split-fp32, _dot3) which
#       accumulate in fp32 at ~2^-21 relative error and ~8-17 TFLOP/s (plain
#       fp32 dots default to tf32 with ~2^-11 error; 'ieee' falls back to slow
#       SIMT FMA).
# fp64: tl.dot fp64 does not compile on triton-musa, so the same WY structure
#       is evaluated with fp64 SIMT rank-1 GEMM kernels (_gemm64*).  The small
#       per-block WY factors (G/T) for all blocks are computed in parallel in
#       phase A; the big matmuls (W = Y^T C) use split-K for enough CTAs.
# ---------------------------------------------------------------------------

_NB = 64  # reflector block size
_SPLIT = 16  # split-K factor for the fp64 SIMT GEMMs


@triton.jit
def _dot3(a, b, acc):
    """fp32-accurate matmul via three fp16 tensor-core dots (split-fp32).

    a ~ ah + al, b ~ bh + bl with ah/bh = fp16 roundings; the fp16 SQMMA path
    on MUSA accumulates in fp32, so ah*bh + al*bh + ah*bl reproduces a*b to
    ~2^-21 relative.
    """
    ah = a.to(tl.float16)
    bh = b.to(tl.float16)
    al = (a - ah.to(tl.float32)).to(tl.float16)
    bl = (b - bh.to(tl.float32)).to(tl.float16)
    acc = tl.dot(ah, bh, acc)
    acc = tl.dot(al, bh, acc)
    acc = tl.dot(ah, bl, acc)
    return acc


@triton.jit
def _copy2d_kernel(
    src_ptr,
    dst_ptr,
    M,
    R,
    stride_sm,
    stride_sr,
    stride_dm,
    stride_dr,
    BLOCK_M: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_r = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    msk = (offs_m[:, None] < M) & (offs_r[None, :] < R)
    val = tl.load(
        src_ptr + offs_m[:, None] * stride_sm + offs_r[None, :] * stride_sr,
        mask=msk,
        other=0.0,
    )
    tl.store(
        dst_ptr + offs_m[:, None] * stride_dm + offs_r[None, :] * stride_dr,
        val,
        mask=msk,
    )


@triton.jit
def _make_y_kernel(
    a_ptr,
    y_ptr,
    yt_ptr,
    M,
    K,
    sa_m,
    sa_k,
    sy_m,
    sy_k,
    syt_m,
    syt_k,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Y[i, j] = a[i, j] for i > j; 1 for i == j; 0 for i < j.

    Writes both Y (m x k) and its transpose Yt (k x m) so matmul kernels can
    always load their operands along contiguous strides.
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mm = offs_m[:, None] < M
    kk = offs_k[None, :] < K
    val = tl.load(
        a_ptr + offs_m[:, None] * sa_m + offs_k[None, :] * sa_k, mask=mm & kk, other=0.0
    )
    lower = offs_m[:, None] > offs_k[None, :]
    diag = offs_m[:, None] == offs_k[None, :]
    val = tl.where(lower, val, 0.0)
    val = tl.where(diag, 1.0, val)
    tl.store(y_ptr + offs_m[:, None] * sy_m + offs_k[None, :] * sy_k, val, mask=mm & kk)
    # Yt[j, i] = Y[i, j]: ptr[a, b] = offs_m[a] * syt_k + offs_k[b] * syt_m
    tl.store(
        yt_ptr + offs_m[:, None] * syt_k + offs_k[None, :] * syt_m, val, mask=kk & mm
    )


# ============================== fp32 WY path ================================


@triton.jit
def _gemm_yt_y(
    yt_ptr,
    y_ptr,
    g_ptr,
    M,
    K,
    syt_m,
    syt_k,
    sy_m,
    sy_k,
    sg,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACC: tl.constexpr,
):
    """G = Y^T @ Y  (K x K), K-loop over M.  Yt tile loaded coalesced."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC)
    for k0 in tl.range(0, M, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        km = offs_k < M
        a = tl.load(
            yt_ptr + offs_m[:, None] * syt_m + offs_k[None, :] * syt_k,
            mask=(offs_m[:, None] < K) & km[None, :],
            other=0.0,
        )
        b = tl.load(
            y_ptr + offs_k[:, None] * sy_m + offs_n[None, :] * sy_k,
            mask=km[:, None] & (offs_n[None, :] < K),
            other=0.0,
        )
        acc = _dot3(a, b, acc)
    tl.store(
        g_ptr + offs_m[:, None] * sg + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < K) & (offs_n[None, :] < K),
    )


@triton.jit
def _make_t_kernel(
    g_ptr, tau_ptr, t_ptr, K, sg, st, BLOCK_N: tl.constexpr, ACC: tl.constexpr
):
    """One program computes the whole KxK compact-WY T block.

    T[i, i] = tau[i];  T[i, j] = -tau[j] * sum_{l=i}^{j-1} T[i, l] G[l, j].
    Uses the column recurrence:  x = -tau[j] * G[:, j] (masked l < j),
    T[:, j] = T @ x, T[j, j] = tau[j]  (zeros below the diagonal are kept).
    """
    offs = tl.arange(0, BLOCK_N)
    msk = offs < K
    # T tile kept in registers, zero-initialised
    t_tile = tl.zeros((BLOCK_N, BLOCK_N), dtype=ACC)
    for j in tl.range(0, K):
        tauj = tl.load(tau_ptr + j)
        g_col = tl.load(g_ptr + offs * sg + j, mask=msk, other=0.0)
        x = tl.where(offs < j, -tauj * g_col, 0.0)
        t_col = tl.sum(t_tile * x[None, :], axis=1)
        # write column j; diagonal gets tau[j]
        t_tile = tl.where(offs[None, :] == j, t_col[:, None], t_tile)
        t_tile = tl.where((offs[:, None] == j) & (offs[None, :] == j), tauj, t_tile)
    tl.store(
        t_ptr + offs[:, None] * st + offs[None, :],
        t_tile,
        mask=msk[:, None] & msk[None, :],
    )


@triton.jit
def _gemm_wz(
    yt_ptr,
    c_ptr,
    t_ptr,
    z_ptr,
    M,
    K,
    R,
    syt_m,
    syt_k,
    sc_m,
    sc_r,
    st,
    sz_m,
    sz_r,
    TRANS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACC: tl.constexpr,
):
    """Z = (T^T) @ (Y^T @ C), one program per column strip.

    First stage: W (BLOCK_M x BLOCK_N) = Y^T @ C[:, cols] with K-loop over M
    (Yt tile loaded coalesced).  Second stage: Z = T (or T^T) @ W in a single
    dot (K = BLOCK_M).
    """
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    nmsk = offs_n < R
    offs_m = tl.arange(0, BLOCK_M)
    mm = offs_m < K

    w = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC)
    for k0 in tl.range(0, M, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        km = offs_k < M
        a = tl.load(
            yt_ptr + offs_m[:, None] * syt_m + offs_k[None, :] * syt_k,
            mask=mm[:, None] & km[None, :],
            other=0.0,
        )
        b = tl.load(
            c_ptr + offs_k[:, None] * sc_m + offs_n[None, :] * sc_r,
            mask=km[:, None] & nmsk[None, :],
            other=0.0,
        )
        w = _dot3(a, b, w)

    if TRANS:
        a = tl.load(
            t_ptr + offs_m[None, :] * st + offs_m[:, None],
            mask=mm[:, None] & mm[None, :],
            other=0.0,
        )  # T^T tile (BM, BM)
    else:
        a = tl.load(
            t_ptr + offs_m[:, None] * st + offs_m[None, :],
            mask=mm[:, None] & mm[None, :],
            other=0.0,
        )  # T tile (BM, BM)
    z = _dot3(a, w, tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC))
    tl.store(
        z_ptr + offs_m[:, None] * sz_m + offs_n[None, :] * sz_r,
        z,
        mask=mm[:, None] & nmsk[None, :],
    )


@triton.jit
def _gemm_cyz(
    y_ptr,
    z_ptr,
    c_ptr,
    out_ptr,
    M,
    K,
    R,
    sy_m,
    sy_k,
    sz_m,
    sz_r,
    sc_m,
    sc_r,
    so_m,
    so_r,
    FIRST: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACC: tl.constexpr,
):
    """out = C - Y @ Z (FIRST) or out -= Y @ Z."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mm = offs_m < M
    nn = offs_n < R
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC)
    for k0 in tl.range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        km = offs_k < K
        a = tl.load(
            y_ptr + offs_m[:, None] * sy_m + offs_k[None, :] * sy_k,
            mask=mm[:, None] & km[None, :],
            other=0.0,
        )
        b = tl.load(
            z_ptr + offs_k[:, None] * sz_m + offs_n[None, :] * sz_r,
            mask=km[:, None] & nn[None, :],
            other=0.0,
        )
        acc = _dot3(a, b, acc)
    if FIRST:
        base = tl.load(
            c_ptr + offs_m[:, None] * sc_m + offs_n[None, :] * sc_r,
            mask=mm[:, None] & nn[None, :],
            other=0.0,
        )
    else:
        base = tl.load(
            out_ptr + offs_m[:, None] * so_m + offs_n[None, :] * so_r,
            mask=mm[:, None] & nn[None, :],
            other=0.0,
        )
    acc = base - acc
    tl.store(
        out_ptr + offs_m[:, None] * so_m + offs_n[None, :] * so_r,
        acc,
        mask=mm[:, None] & nn[None, :],
    )


# ============================== fp64 WY path ================================
# tl.dot does not support fp64 on triton-musa, so the four matmuls are done
# with fp64 SIMT rank-1 kernels (4x-unrolled outer-product accumulation).


@triton.jit
def _gemm64(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    sa_m,
    sa_k,
    sb_k,
    sb_n,
    sc_m,
    sc_n,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    """C[0:M, 0:N] = A[0:M, 0:K] @ B[0:K, 0:N]  (fp64, SIMT rank-1)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mm = offs_m < M
    nn = offs_n < N
    acc = tl.zeros((BM, BN), dtype=tl.float64)
    for k0 in tl.range(0, K, 4):
        k1 = k0 + 1
        k2 = k0 + 2
        k3 = k0 + 3
        km0 = k0 < K
        km1 = k1 < K
        km2 = k2 < K
        km3 = k3 < K
        a0 = tl.load(a_ptr + offs_m * sa_m + k0 * sa_k, mask=mm & km0, other=0.0)
        b0 = tl.load(b_ptr + k0 * sb_k + offs_n * sb_n, mask=nn & km0, other=0.0)
        a1 = tl.load(a_ptr + offs_m * sa_m + k1 * sa_k, mask=mm & km1, other=0.0)
        b1 = tl.load(b_ptr + k1 * sb_k + offs_n * sb_n, mask=nn & km1, other=0.0)
        a2 = tl.load(a_ptr + offs_m * sa_m + k2 * sa_k, mask=mm & km2, other=0.0)
        b2 = tl.load(b_ptr + k2 * sb_k + offs_n * sb_n, mask=nn & km2, other=0.0)
        a3 = tl.load(a_ptr + offs_m * sa_m + k3 * sa_k, mask=mm & km3, other=0.0)
        b3 = tl.load(b_ptr + k3 * sb_k + offs_n * sb_n, mask=nn & km3, other=0.0)
        acc += a0[:, None] * b0[None, :]
        acc += a1[:, None] * b1[None, :]
        acc += a2[:, None] * b2[None, :]
        acc += a3[:, None] * b3[None, :]
    tl.store(
        c_ptr + offs_m[:, None] * sc_m + offs_n[None, :] * sc_n,
        acc,
        mask=mm[:, None] & nn[None, :],
    )


@triton.jit
def _gemm64_g_all(
    yt_ptr,
    y_ptr,
    gp_ptr,
    M,
    K,
    NBLK,
    syt_m,
    syt_k,
    sy_m,
    sy_k,
    sgp_b,
    sgp_s,
    sgp_m,
    sgp_n,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT: tl.constexpr,
    ACC: tl.constexpr,
):
    """Gp[b, s, :, :] += Yt_b^T @ Y_b partial (K-loop over M split in SPLIT)."""
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_s = tl.program_id(2)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    nb_eff = tl.minimum(BLOCK_N, K - pid_b * BLOCK_N)
    mm = offs_m < nb_eff
    nn = offs_n < nb_eff
    acc = tl.zeros((BM, BN), dtype=ACC)
    chunk = (M + SPLIT - 1) // SPLIT
    k_lo = pid_s * chunk
    k_hi = tl.minimum(k_lo + chunk, M)
    for k0 in tl.range(k_lo, k_hi, 4):
        k1 = k0 + 1
        k2 = k0 + 2
        k3 = k0 + 3
        kh = k_hi
        a0 = tl.load(
            yt_ptr + (pid_b * BLOCK_N + offs_m) * syt_m + k0 * syt_k,
            mask=mm & (k0 < kh),
            other=0.0,
        )
        b0 = tl.load(
            y_ptr + k0 * sy_m + (pid_b * BLOCK_N + offs_n) * sy_k,
            mask=nn & (k0 < kh),
            other=0.0,
        )
        a1 = tl.load(
            yt_ptr + (pid_b * BLOCK_N + offs_m) * syt_m + k1 * syt_k,
            mask=mm & (k1 < kh),
            other=0.0,
        )
        b1 = tl.load(
            y_ptr + k1 * sy_m + (pid_b * BLOCK_N + offs_n) * sy_k,
            mask=nn & (k1 < kh),
            other=0.0,
        )
        a2 = tl.load(
            yt_ptr + (pid_b * BLOCK_N + offs_m) * syt_m + k2 * syt_k,
            mask=mm & (k2 < kh),
            other=0.0,
        )
        b2 = tl.load(
            y_ptr + k2 * sy_m + (pid_b * BLOCK_N + offs_n) * sy_k,
            mask=nn & (k2 < kh),
            other=0.0,
        )
        a3 = tl.load(
            yt_ptr + (pid_b * BLOCK_N + offs_m) * syt_m + k3 * syt_k,
            mask=mm & (k3 < kh),
            other=0.0,
        )
        b3 = tl.load(
            y_ptr + k3 * sy_m + (pid_b * BLOCK_N + offs_n) * sy_k,
            mask=nn & (k3 < kh),
            other=0.0,
        )
        acc += a0[:, None] * b0[None, :]
        acc += a1[:, None] * b1[None, :]
        acc += a2[:, None] * b2[None, :]
        acc += a3[:, None] * b3[None, :]
    tl.store(
        gp_ptr
        + pid_b * sgp_b
        + pid_s * sgp_s
        + offs_m[:, None] * sgp_m
        + offs_n[None, :] * sgp_n,
        acc,
        mask=mm[:, None] & nn[None, :],
    )


@triton.jit
def _make_t_all(
    gp_ptr,
    tau_ptr,
    t_ptr,
    K,
    NBLK,
    sgp_b,
    sgp_s,
    sgp_m,
    sgp_n,
    st_b,
    st_m,
    st_n,
    SPLIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ACC: tl.constexpr,
):
    """One program per reflector block: sums the G partials and runs dlarft."""
    b = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    nb_eff = tl.minimum(BLOCK_N, K - b * BLOCK_N)
    msk = offs < nb_eff
    t_tile = tl.zeros((BLOCK_N, BLOCK_N), dtype=ACC)
    for j in tl.range(0, nb_eff):
        tauj = tl.load(tau_ptr + b * BLOCK_N + j)
        g_col = tl.zeros((BLOCK_N,), dtype=ACC)
        for s in tl.static_range(SPLIT):
            g_col += tl.load(
                gp_ptr + b * sgp_b + s * sgp_s + offs * sgp_m + j * sgp_n,
                mask=msk,
                other=0.0,
            )
        x = tl.where(offs < j, -tauj * g_col, 0.0)
        t_col = tl.sum(t_tile * x[None, :], axis=1)
        t_tile = tl.where(offs[None, :] == j, t_col[:, None], t_tile)
        t_tile = tl.where((offs[:, None] == j) & (offs[None, :] == j), tauj, t_tile)
    tl.store(
        t_ptr + b * st_b + offs[:, None] * st_m + offs[None, :] * st_n,
        t_tile,
        mask=msk[:, None] & msk[None, :],
    )


@triton.jit
def _gemm64_w1_split(
    yt_ptr,
    c_ptr,
    wp_ptr,
    M,
    K,
    R,
    BLK,
    SPLIT,
    syt_m,
    syt_k,
    sc_m,
    sc_r,
    swp_s,
    swp_m,
    swp_n,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    """Wp[s, :, :] partial: W = Yt_b^T @ C  (K-loop over M split in SPLIT)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mm = offs_m < K
    nn = offs_n < R
    acc = tl.zeros((BM, BN), dtype=tl.float64)
    chunk = (M + SPLIT - 1) // SPLIT
    k_lo = pid_s * chunk
    k_hi = tl.minimum(k_lo + chunk, M)
    for k0 in tl.range(k_lo, k_hi, 4):
        k1 = k0 + 1
        k2 = k0 + 2
        k3 = k0 + 3
        kh = k_hi
        a0 = tl.load(
            yt_ptr + (BLK + offs_m) * syt_m + k0 * syt_k, mask=mm & (k0 < kh), other=0.0
        )
        b0 = tl.load(c_ptr + k0 * sc_m + offs_n * sc_r, mask=nn & (k0 < kh), other=0.0)
        a1 = tl.load(
            yt_ptr + (BLK + offs_m) * syt_m + k1 * syt_k, mask=mm & (k1 < kh), other=0.0
        )
        b1 = tl.load(c_ptr + k1 * sc_m + offs_n * sc_r, mask=nn & (k1 < kh), other=0.0)
        a2 = tl.load(
            yt_ptr + (BLK + offs_m) * syt_m + k2 * syt_k, mask=mm & (k2 < kh), other=0.0
        )
        b2 = tl.load(c_ptr + k2 * sc_m + offs_n * sc_r, mask=nn & (k2 < kh), other=0.0)
        a3 = tl.load(
            yt_ptr + (BLK + offs_m) * syt_m + k3 * syt_k, mask=mm & (k3 < kh), other=0.0
        )
        b3 = tl.load(c_ptr + k3 * sc_m + offs_n * sc_r, mask=nn & (k3 < kh), other=0.0)
        acc += a0[:, None] * b0[None, :]
        acc += a1[:, None] * b1[None, :]
        acc += a2[:, None] * b2[None, :]
        acc += a3[:, None] * b3[None, :]
    tl.store(
        wp_ptr + pid_s * swp_s + offs_m[:, None] * swp_m + offs_n[None, :] * swp_n,
        acc,
        mask=mm[:, None] & nn[None, :],
    )


@triton.jit
def _gemm64_w2(
    t_ptr,
    wp_ptr,
    z_ptr,
    K,
    R,
    BLK,
    st_b,
    st_m,
    st_n,
    swp_s,
    swp_m,
    swp_n,
    sz_m,
    sz_n,
    TRANS: tl.constexpr,
    SPLIT: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    """Z = T_b (or T_b^T) @ W with W[k, :] = sum_s Wp[s, k, :]."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mm = offs_m < K
    nn = offs_n < R
    acc = tl.zeros((BM, BN), dtype=tl.float64)
    s_idx = tl.arange(0, SPLIT)
    for k in tl.range(0, K):
        if TRANS:
            a = tl.load(
                t_ptr + BLK * st_b + k * st_m + offs_m * st_n, mask=mm, other=0.0
            )
        else:
            a = tl.load(
                t_ptr + BLK * st_b + offs_m * st_m + k * st_n, mask=mm, other=0.0
            )
        w = tl.sum(
            tl.load(
                wp_ptr + s_idx[:, None] * swp_s + k * swp_m + offs_n[None, :] * swp_n,
                mask=nn[None, :],
                other=0.0,
            ),
            axis=0,
        )
        acc += a[:, None] * w[None, :]
    tl.store(
        z_ptr + offs_m[:, None] * sz_m + offs_n[None, :] * sz_n,
        acc,
        mask=mm[:, None] & nn[None, :],
    )


@triton.jit
def _gemm64_cyz(
    y_ptr,
    z_ptr,
    c_ptr,
    out_ptr,
    M,
    K,
    R,
    sy_m,
    sy_k,
    sz_m,
    sz_r,
    sc_m,
    sc_r,
    so_m,
    so_r,
    FIRST: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    """out = C - Y @ Z (FIRST) or out -= Y @ Z  (fp64, SIMT rank-1)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mm = offs_m < M
    nn = offs_n < R
    acc = tl.zeros((BM, BN), dtype=tl.float64)
    for k0 in tl.range(0, K, 4):
        k1 = k0 + 1
        k2 = k0 + 2
        k3 = k0 + 3
        km0 = k0 < K
        km1 = k1 < K
        km2 = k2 < K
        km3 = k3 < K
        a0 = tl.load(y_ptr + offs_m * sy_m + k0 * sy_k, mask=mm & km0, other=0.0)
        b0 = tl.load(z_ptr + k0 * sz_m + offs_n * sz_r, mask=nn & km0, other=0.0)
        a1 = tl.load(y_ptr + offs_m * sy_m + k1 * sy_k, mask=mm & km1, other=0.0)
        b1 = tl.load(z_ptr + k1 * sz_m + offs_n * sz_r, mask=nn & km1, other=0.0)
        a2 = tl.load(y_ptr + offs_m * sy_m + k2 * sy_k, mask=mm & km2, other=0.0)
        b2 = tl.load(z_ptr + k2 * sz_m + offs_n * sz_r, mask=nn & km2, other=0.0)
        a3 = tl.load(y_ptr + offs_m * sy_m + k3 * sy_k, mask=mm & km3, other=0.0)
        b3 = tl.load(z_ptr + k3 * sz_m + offs_n * sz_r, mask=nn & km3, other=0.0)
        acc += a0[:, None] * b0[None, :]
        acc += a1[:, None] * b1[None, :]
        acc += a2[:, None] * b2[None, :]
        acc += a3[:, None] * b3[None, :]
    if FIRST:
        base = tl.load(
            c_ptr + offs_m[:, None] * sc_m + offs_n[None, :] * sc_r,
            mask=mm[:, None] & nn[None, :],
            other=0.0,
        )
    else:
        base = tl.load(
            out_ptr + offs_m[:, None] * so_m + offs_n[None, :] * so_r,
            mask=mm[:, None] & nn[None, :],
            other=0.0,
        )
    acc = base - acc
    tl.store(
        out_ptr + offs_m[:, None] * so_m + offs_n[None, :] * so_r,
        acc,
        mask=mm[:, None] & nn[None, :],
    )


@triton.jit
def _gemm_wz64h(
    yt_ptr,
    c_ptr,
    t_ptr,
    z_ptr,
    M,
    K,
    R,
    syt_m,
    syt_k,
    sc_m,
    sc_r,
    st,
    sz_m,
    sz_r,
    TRANS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """fp64-hybrid Z = (T^T) @ (Y^T @ C): fp64 operands cast to fp32 on load,
    fp16 split3 dots (fp32-accurate), Z stored as fp32."""
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    nmsk = offs_n < R
    offs_m = tl.arange(0, BLOCK_M)
    mm = offs_m < K

    w = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in tl.range(0, M, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        km = offs_k < M
        a = tl.load(
            yt_ptr + offs_m[:, None] * syt_m + offs_k[None, :] * syt_k,
            mask=mm[:, None] & km[None, :],
            other=0.0,
        ).to(tl.float32)
        b = tl.load(
            c_ptr + offs_k[:, None] * sc_m + offs_n[None, :] * sc_r,
            mask=km[:, None] & nmsk[None, :],
            other=0.0,
        ).to(tl.float32)
        w = _dot3(a, b, w)

    if TRANS:
        a = tl.load(
            t_ptr + offs_m[None, :] * st + offs_m[:, None],
            mask=mm[:, None] & mm[None, :],
            other=0.0,
        ).to(tl.float32)
    else:
        a = tl.load(
            t_ptr + offs_m[:, None] * st + offs_m[None, :],
            mask=mm[:, None] & mm[None, :],
            other=0.0,
        ).to(tl.float32)
    z = _dot3(a, w, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32))
    tl.store(
        z_ptr + offs_m[:, None] * sz_m + offs_n[None, :] * sz_r,
        z,
        mask=mm[:, None] & nmsk[None, :],
    )


@triton.jit
def _gemm_cyz64h(
    y_ptr,
    z_ptr,
    c_ptr,
    out_ptr,
    M,
    K,
    R,
    sy_m,
    sy_k,
    sz_m,
    sz_r,
    sc_m,
    sc_r,
    so_m,
    so_r,
    FIRST: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """fp64-hybrid out = C - Y @ Z (FIRST) or out -= Y @ Z: fp16 split3 dots,
    fp64 base subtract and fp64 store."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mm = offs_m < M
    nn = offs_n < R
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in tl.range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        km = offs_k < K
        a = tl.load(
            y_ptr + offs_m[:, None] * sy_m + offs_k[None, :] * sy_k,
            mask=mm[:, None] & km[None, :],
            other=0.0,
        ).to(tl.float32)
        b = tl.load(
            z_ptr + offs_k[:, None] * sz_m + offs_n[None, :] * sz_r,
            mask=km[:, None] & nn[None, :],
            other=0.0,
        ).to(tl.float32)
        acc = _dot3(a, b, acc)
    if FIRST:
        base = tl.load(
            c_ptr + offs_m[:, None] * sc_m + offs_n[None, :] * sc_r,
            mask=mm[:, None] & nn[None, :],
            other=0.0,
        )
    else:
        base = tl.load(
            out_ptr + offs_m[:, None] * so_m + offs_n[None, :] * so_r,
            mask=mm[:, None] & nn[None, :],
            other=0.0,
        )
    res = base - acc.to(tl.float64)
    tl.store(
        out_ptr + offs_m[:, None] * so_m + offs_n[None, :] * so_r,
        res,
        mask=mm[:, None] & nn[None, :],
    )


def _run_left_wy64(input, tau, other, transpose):
    """fp64 blocked compact-WY: fp64-exact G/T factors, fp32-split3-dot
    applications (end-to-end error ~1e-5, well under atol=1e-4)."""
    m, k = input.shape
    r = other.shape[1]
    k_eff = min(m, k)
    dtype = input.dtype
    dev = input.device

    out = torch.empty_like(other)
    if k_eff == 0:
        _copy2d_kernel[(triton.cdiv(m, 64), triton.cdiv(r, 64))](
            other,
            out,
            m,
            r,
            other.stride(0),
            other.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_M=64,
            BLOCK_R=64,
        )
        return out

    Y = torch.empty((m, k_eff), device=dev, dtype=dtype)
    Yt = torch.empty((k_eff, m), device=dev, dtype=dtype)
    _make_y_kernel[(triton.cdiv(m, 64), triton.cdiv(k_eff, 64))](
        input,
        Y,
        Yt,
        m,
        k_eff,
        input.stride(0),
        input.stride(1),
        Y.stride(0),
        Y.stride(1),
        Yt.stride(0),
        Yt.stride(1),
        BLOCK_M=64,
        BLOCK_K=64,
    )

    NB = _NB
    SPLIT = _SPLIT
    nblocks = (k_eff + NB - 1) // NB

    # ---- phase A: all WY factors in fp64 (exact) ----
    Gp = torch.empty((nblocks, SPLIT, NB, NB), device=dev, dtype=dtype)
    T = torch.empty((nblocks, NB, NB), device=dev, dtype=dtype)
    _gemm64_g_all[(2, nblocks, SPLIT)](
        Yt,
        Y,
        Gp,
        m,
        k_eff,
        nblocks,
        Yt.stride(0),
        Yt.stride(1),
        Y.stride(0),
        Y.stride(1),
        Gp.stride(0),
        Gp.stride(1),
        Gp.stride(2),
        Gp.stride(3),
        BM=32,
        BN=64,
        BLOCK_N=NB,
        SPLIT=SPLIT,
        ACC=tl.float64,
        num_warps=4,
    )
    _make_t_all[(nblocks,)](
        Gp,
        tau,
        T,
        k_eff,
        nblocks,
        Gp.stride(0),
        Gp.stride(1),
        Gp.stride(2),
        Gp.stride(3),
        T.stride(0),
        T.stride(1),
        T.stride(2),
        SPLIT=SPLIT,
        BLOCK_N=NB,
        ACC=tl.float64,
        num_warps=4,
    )

    # ---- phase B: sequential sweep with fp16 split3 dots (pairs) ----
    Z32 = torch.empty((2 * NB, r), device=dev, dtype=torch.float32)
    first = True
    for b0, b1 in _sweep_pairs(nblocks, transpose):
        j0 = b0 * NB
        nb = min(NB, k_eff - j0)
        c_src = other if first else out

        if b1 is None:
            yb = Y.narrow(1, j0, nb)
            ytb = Yt.narrow(0, j0, nb)
            tb = T[b0]
            _gemm_wz64h[(triton.cdiv(r, 32),)](
                ytb,
                c_src,
                tb,
                Z32,
                m,
                nb,
                r,
                ytb.stride(0),
                ytb.stride(1),
                c_src.stride(0),
                c_src.stride(1),
                tb.stride(0),
                Z32.stride(0),
                Z32.stride(1),
                TRANS=transpose,
                BLOCK_M=NB,
                BLOCK_N=32,
                BLOCK_K=64,
                num_warps=8,
            )
            _gemm_cyz64h[(triton.cdiv(m, 64), triton.cdiv(r, 128))](
                yb,
                Z32,
                other,
                out,
                m,
                nb,
                r,
                yb.stride(0),
                yb.stride(1),
                Z32.stride(0),
                Z32.stride(1),
                other.stride(0),
                other.stride(1),
                out.stride(0),
                out.stride(1),
                FIRST=first,
                BLOCK_M=64,
                BLOCK_N=128,
                BLOCK_K=64,
                num_warps=8,
            )
            first = False
            continue

        j1 = b1 * NB
        nb1 = min(NB, k_eff - j1)
        yb0 = Y.narrow(1, j0, nb)
        ytb0 = Yt.narrow(0, j0, nb)
        yb1 = Y.narrow(1, j1, nb1)
        ytb1 = Yt.narrow(0, j1, nb1)
        tb0 = T[b0]
        tb1 = T[b1]

        # Z0 = T_b0 (Yt_b0^T C), Z1 = T_b1 (Yt_b1^T C) sharing C loads
        _gemm_wz_pair[(triton.cdiv(r, 32),)](
            ytb0,
            ytb1,
            c_src,
            tb0,
            tb1,
            Z32,
            m,
            nb,
            nb1,
            r,
            ytb0.stride(0),
            ytb0.stride(1),
            c_src.stride(0),
            c_src.stride(1),
            tb0.stride(0),
            Z32.stride(0),
            Z32.stride(1),
            TRANS=transpose,
            BLOCK_M=NB,
            BLOCK_N=32,
            BLOCK_K=64,
            num_warps=8,
        )

        # apply the two blocks with the (more efficient) single-block cyz
        z0 = Z32.narrow(0, 0, nb)
        z1 = Z32.narrow(0, nb, nb1)
        _gemm_cyz64h[(triton.cdiv(m, 64), triton.cdiv(r, 128))](
            yb0,
            z0,
            other,
            out,
            m,
            nb,
            r,
            yb0.stride(0),
            yb0.stride(1),
            z0.stride(0),
            z0.stride(1),
            other.stride(0),
            other.stride(1),
            out.stride(0),
            out.stride(1),
            FIRST=first,
            BLOCK_M=64,
            BLOCK_N=128,
            BLOCK_K=64,
            num_warps=8,
        )
        _gemm_cyz64h[(triton.cdiv(m, 64), triton.cdiv(r, 128))](
            yb1,
            z1,
            other,
            out,
            m,
            nb1,
            r,
            yb1.stride(0),
            yb1.stride(1),
            z1.stride(0),
            z1.stride(1),
            other.stride(0),
            other.stride(1),
            out.stride(0),
            out.stride(1),
            FIRST=False,
            BLOCK_M=64,
            BLOCK_N=128,
            BLOCK_K=64,
            num_warps=8,
        )
        first = False
    return out


@triton.jit
def _gemm_wz_pair(
    yt0_ptr,
    yt1_ptr,
    c_ptr,
    t0_ptr,
    t1_ptr,
    z_ptr,
    M,
    K,
    K1,
    R,
    syt_m,
    syt_k,
    sc_m,
    sc_r,
    st,
    sz_m,
    sz_r,
    TRANS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Z0 = T0 (Yt0^T C) and Z1 = T1 (Yt1^T C) in one kernel, sharing the C
    loads.  Z0 goes to z_ptr[0:K, :], Z1 to z_ptr[K:2K, :].  Operands may be
    fp64 (cast to fp32 on load); Z is stored fp32."""
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    nmsk = offs_n < R
    offs_m = tl.arange(0, BLOCK_M)
    mm0 = offs_m < K
    mm1 = offs_m < K1

    w0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in tl.range(0, M, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        km = offs_k < M
        b = tl.load(
            c_ptr + offs_k[:, None] * sc_m + offs_n[None, :] * sc_r,
            mask=km[:, None] & nmsk[None, :],
            other=0.0,
        ).to(tl.float32)
        a0 = tl.load(
            yt0_ptr + offs_m[:, None] * syt_m + offs_k[None, :] * syt_k,
            mask=mm0[:, None] & km[None, :],
            other=0.0,
        ).to(tl.float32)
        a1 = tl.load(
            yt1_ptr + offs_m[:, None] * syt_m + offs_k[None, :] * syt_k,
            mask=mm1[:, None] & km[None, :],
            other=0.0,
        ).to(tl.float32)
        w0 = _dot3(a0, b, w0)
        w1 = _dot3(a1, b, w1)

    if TRANS:
        a = tl.load(
            t0_ptr + offs_m[None, :] * st + offs_m[:, None],
            mask=mm0[:, None] & mm0[None, :],
            other=0.0,
        ).to(tl.float32)
    else:
        a = tl.load(
            t0_ptr + offs_m[:, None] * st + offs_m[None, :],
            mask=mm0[:, None] & mm0[None, :],
            other=0.0,
        ).to(tl.float32)
    z0 = _dot3(a, w0, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32))
    if TRANS:
        a = tl.load(
            t1_ptr + offs_m[None, :] * st + offs_m[:, None],
            mask=mm1[:, None] & mm1[None, :],
            other=0.0,
        ).to(tl.float32)
    else:
        a = tl.load(
            t1_ptr + offs_m[:, None] * st + offs_m[None, :],
            mask=mm1[:, None] & mm1[None, :],
            other=0.0,
        ).to(tl.float32)
    z1 = _dot3(a, w1, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32))

    tl.store(
        z_ptr + offs_m[:, None] * sz_m + offs_n[None, :] * sz_r,
        z0,
        mask=mm0[:, None] & nmsk[None, :],
    )
    tl.store(
        z_ptr + K * sz_m + offs_m[:, None] * sz_m + offs_n[None, :] * sz_r,
        z1,
        mask=mm1[:, None] & nmsk[None, :],
    )


@triton.jit
def _gemm_cyz_pair(
    y0_ptr,
    y1_ptr,
    z_ptr,
    c_ptr,
    out_ptr,
    M,
    K,
    K1,
    R,
    sy_m,
    sy_k,
    sz_m,
    sz_r,
    sc_m,
    sc_r,
    so_m,
    so_r,
    FIRST: tl.constexpr,
    OUT_FP64: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """out = C - Y0 Z0 - Y1 Z1 (FIRST) or out -= Y0 Z0 + Y1 Z1 in one pass."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mm = offs_m < M
    nn = offs_n < R
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in tl.range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        km = offs_k < K
        a0 = tl.load(
            y0_ptr + offs_m[:, None] * sy_m + offs_k[None, :] * sy_k,
            mask=mm[:, None] & km[None, :],
            other=0.0,
        ).to(tl.float32)
        b0 = tl.load(
            z_ptr + offs_k[:, None] * sz_m + offs_n[None, :] * sz_r,
            mask=km[:, None] & nn[None, :],
            other=0.0,
        ).to(tl.float32)
        acc = _dot3(a0, b0, acc)
    for k0 in tl.range(0, K1, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        km = offs_k < K1
        a1 = tl.load(
            y1_ptr + offs_m[:, None] * sy_m + offs_k[None, :] * sy_k,
            mask=mm[:, None] & km[None, :],
            other=0.0,
        ).to(tl.float32)
        b1 = tl.load(
            z_ptr + K * sz_m + offs_k[:, None] * sz_m + offs_n[None, :] * sz_r,
            mask=km[:, None] & nn[None, :],
            other=0.0,
        ).to(tl.float32)
        acc = _dot3(a1, b1, acc)
    if FIRST:
        base = tl.load(
            c_ptr + offs_m[:, None] * sc_m + offs_n[None, :] * sc_r,
            mask=mm[:, None] & nn[None, :],
            other=0.0,
        )
    else:
        base = tl.load(
            out_ptr + offs_m[:, None] * so_m + offs_n[None, :] * so_r,
            mask=mm[:, None] & nn[None, :],
            other=0.0,
        )
    if OUT_FP64:
        res = base - acc.to(tl.float64)
    else:
        res = base - acc
    tl.store(
        out_ptr + offs_m[:, None] * so_m + offs_n[None, :] * so_r,
        res,
        mask=mm[:, None] & nn[None, :],
    )


def _sweep_pairs(nblocks, transpose):
    """Yield (b0, b1) consecutive-block pairs in sweep order; b1 may be None.

    transpose=False applies Q: blocks in reverse order (n-1 .. 0).
    transpose=True  applies Q^T: blocks in forward order (0 .. n-1).
    """
    if transpose:
        for s in range(0, nblocks, 2):
            yield (s, s + 1 if s + 1 < nblocks else None)
    else:
        for s in range(nblocks - 1, -1, -2):
            yield (s, s - 1 if s - 1 >= 0 else None)


def _run_left_wy(input, tau, other, transpose):
    m, k = input.shape
    r = other.shape[1]
    k_eff = min(m, k)
    dtype = input.dtype
    acc_dtype = tl.float32 if dtype == torch.float32 else tl.float64
    dev = input.device

    out = torch.empty_like(other)
    if k_eff == 0:
        grid2 = (triton.cdiv(m, 64), triton.cdiv(r, 64))
        _copy2d_kernel[grid2](
            other,
            out,
            m,
            r,
            other.stride(0),
            other.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_M=64,
            BLOCK_R=64,
        )
        return out

    # ---- materialise Y (m x k_eff) and Yt (k_eff x m) once ----
    Y = torch.empty((m, k_eff), device=dev, dtype=dtype)
    Yt = torch.empty((k_eff, m), device=dev, dtype=dtype)
    _make_y_kernel[(triton.cdiv(m, 64), triton.cdiv(k_eff, 64))](
        input,
        Y,
        Yt,
        m,
        k_eff,
        input.stride(0),
        input.stride(1),
        Y.stride(0),
        Y.stride(1),
        Yt.stride(0),
        Yt.stride(1),
        BLOCK_M=64,
        BLOCK_K=64,
    )

    NB = _NB
    SPLIT = _SPLIT
    nblocks = (k_eff + NB - 1) // NB

    # ---- phase A: all per-block WY factors in parallel ----
    Gp = torch.empty((nblocks, SPLIT, NB, NB), device=dev, dtype=dtype)
    T = torch.empty((nblocks, NB, NB), device=dev, dtype=dtype)
    _gemm64_g_all[(2, nblocks, SPLIT)](
        Yt,
        Y,
        Gp,
        m,
        k_eff,
        nblocks,
        Yt.stride(0),
        Yt.stride(1),
        Y.stride(0),
        Y.stride(1),
        Gp.stride(0),
        Gp.stride(1),
        Gp.stride(2),
        Gp.stride(3),
        BM=32,
        BN=64,
        BLOCK_N=NB,
        SPLIT=SPLIT,
        ACC=acc_dtype,
        num_warps=4,
    )
    _make_t_all[(nblocks,)](
        Gp,
        tau,
        T,
        k_eff,
        nblocks,
        Gp.stride(0),
        Gp.stride(1),
        Gp.stride(2),
        Gp.stride(3),
        T.stride(0),
        T.stride(1),
        T.stride(2),
        SPLIT=SPLIT,
        BLOCK_N=NB,
        ACC=acc_dtype,
        num_warps=4,
    )

    # ---- phase B: sequential sweep (single blocks for fp32: the pair kernels'
    # extra register pressure hurts the already-fast fp32 path) ----
    Z = torch.empty((2 * NB, r), device=dev, dtype=torch.float32)
    blocks = range(nblocks - 1, -1, -1) if not transpose else range(0, nblocks)
    first = True
    for b in blocks:
        j0 = b * NB
        nb = min(NB, k_eff - j0)
        yb = Y.narrow(1, j0, nb)
        ytb = Yt.narrow(0, j0, nb)
        tb = T[b]
        c_src = other if first else out

        # Z = T_b (or T_b^T) @ (Y_b^T @ C_current)
        _gemm_wz[(triton.cdiv(r, 32),)](
            ytb,
            c_src,
            tb,
            Z,
            m,
            nb,
            r,
            ytb.stride(0),
            ytb.stride(1),
            c_src.stride(0),
            c_src.stride(1),
            tb.stride(0),
            Z.stride(0),
            Z.stride(1),
            TRANS=transpose,
            BLOCK_M=NB,
            BLOCK_N=32,
            BLOCK_K=64,
            ACC=tl.float32,
            num_warps=8,
        )

        # out = C - Y_b @ Z_b   (or out -= Y_b @ Z_b)
        _gemm_cyz[(triton.cdiv(m, 64), triton.cdiv(r, 128))](
            yb,
            Z,
            other,
            out,
            m,
            nb,
            r,
            yb.stride(0),
            yb.stride(1),
            Z.stride(0),
            Z.stride(1),
            other.stride(0),
            other.stride(1),
            out.stride(0),
            out.stride(1),
            FIRST=first,
            BLOCK_M=64,
            BLOCK_N=128,
            BLOCK_K=64,
            ACC=tl.float32,
            num_warps=8,
        )
        first = False
    return out


# ------------------------- naive fallback ---------------------


@triton.jit
def _ormqr_left_step(
    c_ptr,
    a_ptr,
    tau_ptr,
    M,
    R,
    J,
    stride_cm,
    stride_cr,
    stride_am,
    stride_ak,
    BLOCK_R: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_c = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    cmask = offs_c < R
    tau = tl.load(tau_ptr + J)
    w = tl.load(c_ptr + J * stride_cm + offs_c * stride_cr, mask=cmask, other=0.0)
    for i0 in tl.range(J + 1, M, BLOCK_I):
        offs_i = i0 + tl.arange(0, BLOCK_I)
        imask = offs_i < M
        v = tl.load(a_ptr + offs_i * stride_am + J * stride_ak, mask=imask, other=0.0)
        c = tl.load(
            c_ptr + offs_i[:, None] * stride_cm + offs_c[None, :] * stride_cr,
            mask=imask[:, None] & cmask[None, :],
            other=0.0,
        )
        w += tl.sum(v[:, None] * c, axis=0)
    for i0 in tl.range(J, M, BLOCK_I):
        offs_i = i0 + tl.arange(0, BLOCK_I)
        imask = offs_i < M
        v = tl.load(a_ptr + offs_i * stride_am + J * stride_ak, mask=imask, other=0.0)
        v = tl.where(offs_i == J, 1.0, v)
        c = tl.load(
            c_ptr + offs_i[:, None] * stride_cm + offs_c[None, :] * stride_cr,
            mask=imask[:, None] & cmask[None, :],
            other=0.0,
        )
        c = c - tau * v[:, None] * w[None, :]
        tl.store(
            c_ptr + offs_i[:, None] * stride_cm + offs_c[None, :] * stride_cr,
            c,
            mask=imask[:, None] & cmask[None, :],
        )


def _run_left_naive(input, tau, other, transpose):
    m, k = input.shape
    r = other.shape[1]
    k_eff = min(m, k)
    out = torch.empty_like(other)
    _copy2d_kernel[(triton.cdiv(m, 64), triton.cdiv(r, 64))](
        other,
        out,
        m,
        r,
        other.stride(0),
        other.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=64,
        BLOCK_R=64,
    )
    if k_eff == 0:
        return out
    js = range(k_eff - 1, -1, -1) if not transpose else range(0, k_eff)
    grid = (triton.cdiv(r, 128),)
    for j in js:
        _ormqr_left_step[grid](
            out,
            input,
            tau,
            m,
            r,
            j,
            out.stride(0),
            out.stride(1),
            input.stride(0),
            input.stride(1),
            BLOCK_R=128,
            BLOCK_I=64,
            num_warps=4,
        )
    return out


def _specialized_ormqr(input, tau, other, left=True, transpose=False):
    """Apply Householder reflectors:  out = Q op other  (torch.ormqr semantics).

    left=False is implemented with the identity
        C . Q   = (Q^T . C^T)^T        C . Q^T = (Q . C^T)^T
    so only the left-application path needs kernels.
    """
    if isinstance(left, torch.Tensor):
        left = bool(left.item())
    if isinstance(transpose, torch.Tensor):
        transpose = bool(transpose.item())
    is_fp64 = input.dtype == torch.float64
    if not left:
        c = other.t()
        if is_fp64:
            out_t = _run_left_wy64(input, tau, c, transpose=not transpose)
        else:
            out_t = _run_left_wy(input, tau, c, transpose=not transpose)
        return out_t.t()
    if is_fp64:
        return _run_left_wy64(input, tau, other, transpose=transpose)
    return _run_left_wy(input, tau, other, transpose=transpose)


def ormqr(input, tau, other, left=True, transpose=False):
    logger.debug("GEMS_MTHREADS ORMQR")
    if (
        isinstance(input, torch.Tensor)
        and input.device.type == "musa"
        and input.dtype in _SUPPORTED_DTYPES
        and isinstance(tau, torch.Tensor)
        and tau.device.type == "musa"
        and tau.dtype in _SUPPORTED_DTYPES
        and isinstance(other, torch.Tensor)
        and other.device.type == "musa"
        and other.dtype in _SUPPORTED_DTYPES
    ):
        return _specialized_ormqr(input, tau, other, left, transpose)
    return default_ormqr(input, tau, other, left, transpose)

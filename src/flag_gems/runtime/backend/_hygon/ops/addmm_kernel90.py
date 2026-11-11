# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Grouped GEMM and small-output reductions for Hygon addmm."""

import torch
import triton
import triton.language as tl

from flag_gems.ops.addmm import addmm as _generic_addmm
from flag_gems.ops.addmm import addmm_out as _generic_addmm_out
from flag_gems.runtime import torch_device_fn


@triton.jit
def _gemv(
    A,
    B,
    I,
    O,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    IM: tl.constexpr,
    IN: tl.constexpr,
    OM: tl.constexpr,
    ON: tl.constexpr,
    ALPHA: tl.constexpr,
    BETA: tl.constexpr,
    BO: tl.constexpr,
    RK: tl.constexpr,
):
    o = tl.program_id(0) * BO + tl.arange(0, BO)
    m, n = o // N, o % N
    k = tl.arange(0, RK)
    a = tl.load(
        A + m[:, None] * AM + k[None, :] * AK,
        (m[:, None] < M) & (k[None, :] < K),
        other=0,
    ).to(tl.float32)
    b = tl.load(
        B + n[:, None] * BN + k[None, :] * BK,
        (o[:, None] < M * N) & (k[None, :] < K),
        other=0,
    ).to(tl.float32)
    value = tl.sum(a * b, 1) * ALPHA
    if BETA != 0:
        bias = tl.load(I + m * IM + n * IN, o < M * N, other=0).to(tl.float32)
        value += bias * BETA
    tl.store(O + m * OM + n * ON, value, o < M * N)


@triton.jit
def _grouped_addmm(
    A,
    B,
    I,
    O,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    IM: tl.constexpr,
    IN: tl.constexpr,
    OM: tl.constexpr,
    ON: tl.constexpr,
    ALPHA: tl.constexpr,
    BETA: tl.constexpr,
    BM: tl.constexpr,
    BN_TILE: tl.constexpr,
    BK_TILE: tl.constexpr,
    PRECISE: tl.constexpr,
    GROUP: tl.constexpr = 4,
    UNROLL: tl.constexpr = 1,
    PARTS16: tl.constexpr = False,
    ALIGNED_K: tl.constexpr = False,
):
    pid = tl.program_id(0)
    nm, nn = tl.cdiv(M, BM), tl.cdiv(N, BN_TILE)
    group = pid // (GROUP * nn)
    first = group * GROUP
    gm = tl.minimum(nm - first, GROUP)
    pm = first + pid % (GROUP * nn) % gm
    pn = pid % (GROUP * nn) // gm
    m = pm * BM + tl.arange(0, BM)
    n = pn * BN_TILE + tl.arange(0, BN_TILE)
    k = tl.arange(0, BK_TILE)
    if PRECISE:
        acc = tl.zeros((BM, BN_TILE), tl.float64)
    else:
        acc = tl.zeros((BM, BN_TILE), tl.float32)
    for step in tl.range(0, tl.cdiv(K, BK_TILE), loop_unroll_factor=UNROLL):
        kk = step * BK_TILE + k
        if (
            ALIGNED_K
            or (not PRECISE and K % BK_TILE == 0)
            or (M % BM == 0 and N % BN_TILE == 0 and K % BK_TILE == 0)
        ):
            # Wrap only discarded output lanes, so the main loop has full,
            # unpredicated shared-memory tiles even when M has a short tail.
            a = tl.load(A + (m % M)[:, None] * AM + kk[None, :] * AK)
            b = tl.load(B + kk[:, None] * BK + (n % N)[None, :] * BN)
        else:
            a = tl.load(
                A + m[:, None] * AM + kk[None, :] * AK,
                (m[:, None] < M) & (kk[None, :] < K),
                other=0,
            )
            b = tl.load(
                B + kk[:, None] * BK + n[None, :] * BN,
                (kk[:, None] < K) & (n[None, :] < N),
                other=0,
            )
        if PRECISE:
            if PARTS16:
                a0, a1 = tl.split(tl.permute(tl.reshape(a, (BM, 2, 16)), (0, 2, 1)))
                b0, b1 = tl.split(
                    tl.permute(tl.reshape(b, (2, 16, BN_TILE)), (1, 2, 0))
                )
                acc += tl.dot(a0, b0, allow_tf32=False).to(tl.float64)
                acc += tl.dot(a1, b1, allow_tf32=False).to(tl.float64)
            else:
                acc += tl.dot(a, b, allow_tf32=False).to(tl.float64)
        else:
            acc = tl.dot(a, b, acc, allow_tf32=False)
    value = acc * ALPHA
    if BETA != 0:
        bias = tl.load(
            I + m[:, None] * IM + n[None, :] * IN,
            (m[:, None] < M) & (n[None, :] < N),
            other=0,
        )
        value += bias * BETA
    tl.store(
        O + m[:, None] * OM + n[None, :] * ON,
        value,
        (m[:, None] < M) & (n[None, :] < N),
    )


@triton.jit
def _dual_bf16_addmm(
    A,
    B,
    I,
    O,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    # Independent accumulators shorten the dependency chain across K tiles.
    # Keep partial sums in FP32 and round only after adding the bias.
    tl.static_assert(K % (2 * BK) == 0)
    pid = tl.program_id(0)
    nn = tl.cdiv(N, BN)
    pm, pn = pid // nn, pid % nn
    if nn % 8 == 0:
        # Spread column-tile traffic between neighboring row tiles.
        pn = pn ^ (pm % 8)
    m = pm * BM + tl.arange(0, BM)
    n = pn * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc0 = tl.zeros((BM, BN), tl.float32)
    acc1 = tl.zeros((BM, BN), tl.float32)
    for step in tl.range(0, K // (2 * BK), loop_unroll_factor=1):
        kk = step * (2 * BK) + k
        a0 = tl.load(A + (m % M)[:, None] * K + kk[None, :])
        b0 = tl.load(B + kk[:, None] + (n % N)[None, :] * K)
        acc0 = tl.dot(a0, b0, acc0, allow_tf32=False)
        a1 = tl.load(A + (m % M)[:, None] * K + kk[None, :] + BK)
        b1 = tl.load(B + kk[:, None] + BK + (n % N)[None, :] * K)
        acc1 = tl.dot(a1, b1, acc1, allow_tf32=False)
    bias = tl.load(I + n, n < N, other=0).to(tl.float32)
    value = acc0 + acc1 + bias[None, :]
    # Half-tile stores avoid scalar BF16 LDS writes and keep the loop spill-free.
    lo, hi = tl.split(tl.permute(tl.reshape(value, (2, BM // 2, BN)), (1, 2, 0)))
    r0 = pm * BM + tl.arange(0, BM // 2)
    tl.store(
        O + r0[:, None] * N + n[None, :],
        lo,
        (r0[:, None] < M) & (n[None, :] < N),
    )
    r1 = r0 + BM // 2
    tl.store(
        O + r1[:, None] * N + n[None, :],
        hi,
        (r1[:, None] < M) & (n[None, :] < N),
    )


@triton.jit
def _single_bf16_addmm(
    A,
    B,
    I,
    O,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    # A single accumulator and quarter-tile stores let two 256x256 blocks
    # fit in a CU's register file and 64 KiB LDS for the shorter K path.
    pid = tl.program_id(0)
    nm, nn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    first = pid // (4 * nn) * 4
    gm = tl.minimum(nm - first, 4)
    pm = first + pid % (4 * nn) % gm
    pn = pid % (4 * nn) // gm
    m = pm * BM + tl.arange(0, BM)
    n = pn * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for step in tl.range(K // BK, loop_unroll_factor=1):
        k = step * BK + kk
        a = tl.load(A + (m % M)[:, None] * K + k[None, :])
        b = tl.load(B + k[:, None] + (n % N)[None, :] * K)
        acc = tl.dot(a, b, acc, allow_tf32=False)
    bias = tl.load(I + n, n < N, other=0).to(tl.float32)
    value = acc + bias[None, :]
    lo, hi = tl.split(tl.permute(tl.reshape(value, (2, BM // 2, BN)), (1, 2, 0)))
    q0, q1 = tl.split(tl.permute(tl.reshape(lo, (2, BM // 4, BN)), (1, 2, 0)))
    q2, q3 = tl.split(tl.permute(tl.reshape(hi, (2, BM // 4, BN)), (1, 2, 0)))
    r0 = pm * BM + tl.arange(0, BM // 4)
    tl.store(
        O + r0[:, None] * N + n[None, :],
        q0,
        (r0[:, None] < M) & (n[None, :] < N),
    )
    r1 = r0 + BM // 4
    tl.store(
        O + r1[:, None] * N + n[None, :],
        q1,
        (r1[:, None] < M) & (n[None, :] < N),
    )
    r2 = r0 + BM // 2
    tl.store(
        O + r2[:, None] * N + n[None, :],
        q2,
        (r2[:, None] < M) & (n[None, :] < N),
    )
    r3 = r0 + 3 * BM // 4
    tl.store(
        O + r3[:, None] * N + n[None, :],
        q3,
        (r3[:, None] < M) & (n[None, :] < N),
    )


def _use_dual_bf16(bias, a, b, out, alpha, beta):
    m, k = a.shape
    n = b.shape[1]
    return (
        a.dtype == b.dtype == out.dtype == torch.bfloat16
        and m >= 8192
        and n >= 2048
        and n % 256 == 0
        and k >= 2048
        and k % 64 == 0
        and a.stride() == (k, 1)
        and b.stride() == (1, k)
        and out.stride() == (n, 1)
        and bias.ndim == 1
        and bias.shape[0] == n
        and bias.stride() == (1,)
        and alpha == 1
        and beta == 1
    )


def _medium_fp32(a, b):
    m, k = a.shape
    n = b.shape[1]
    return (
        a.dtype == torch.float32
        and 128 <= m <= 512
        and 256 <= n <= 512
        and 256 <= k <= 512
    )


def _small_fp32(a, b):
    m, k = a.shape
    n = b.shape[1]
    return (
        a.dtype == b.dtype == torch.float32
        and 8 < m <= 64
        and 0 < n <= 64
        and 0 < k <= 128
    )


def _expanded_fp32(a, b):
    m, k = a.shape
    n = b.shape[1]
    return a.dtype == b.dtype == torch.float32 and (
        (m == 1 and k >= 256)
        or (n == 1 and k >= 4096)
        or (8 < m <= 4096 and 0 < n <= 128 and 0 < k <= 128)
        or (512 <= m <= 4096 and 128 <= n <= 512 and 128 <= k <= 512)
    )


def _eligible(a, b):
    m, k = a.shape
    n = b.shape[1]
    return a.dtype in (torch.float16, torch.bfloat16, torch.float32) and (
        (m <= 8 and n <= 256 and k <= 4096)
        or _small_fp32(a, b)
        or _expanded_fp32(a, b)
        or (a.dtype in (torch.float16, torch.bfloat16) and m >= 1024 and n >= 256)
        or _medium_fp32(a, b)
        or (
            a.dtype == torch.float32
            and k >= 256
            and ((m >= 4096 and n >= 64) or (m >= 256 and n >= 512 and k >= 1024))
        )
    )


@triton.jit
def _precise_vector(
    A,
    B,
    I,
    O,
    P,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    IM: tl.constexpr,
    IN: tl.constexpr,
    OM: tl.constexpr,
    ON: tl.constexpr,
    ALPHA: tl.constexpr,
    BETA: tl.constexpr,
    BO: tl.constexpr,
    RK: tl.constexpr,
    SPLIT: tl.constexpr,
    PRECISE: tl.constexpr,
):
    o = tl.program_id(0) * BO + tl.arange(0, BO)
    m = o // N
    n = o % N
    k = tl.program_id(1) * RK + tl.arange(0, RK)
    dtype: tl.constexpr = tl.float64 if PRECISE else tl.float32
    a = tl.load(
        A + m[:, None] * AM + k[None, :] * AK,
        (m[:, None] < M) & (k[None, :] < K),
        other=0,
    ).to(dtype)
    b = tl.load(
        B + n[:, None] * BN + k[None, :] * BK,
        (o[:, None] < M * N) & (k[None, :] < K),
        other=0,
    ).to(dtype)
    v = tl.sum(a * b, 1)
    if SPLIT:
        tl.store(P + tl.program_id(1) * M * N + o, v, o < M * N)
    else:
        if BETA != 0:
            bias = tl.load(I + m * IM + n * IN, o < M * N, other=0).to(dtype)
            v = v * ALPHA + bias * BETA
        else:
            v = v * ALPHA
        tl.store(O + m * OM + n * ON, v, o < M * N)


@triton.jit
def _precise_vector_finish(
    P,
    I,
    O,
    M: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    IM: tl.constexpr,
    IN: tl.constexpr,
    OM: tl.constexpr,
    ON: tl.constexpr,
    ALPHA: tl.constexpr,
    BETA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    o = tl.program_id(0)
    s = tl.arange(0, BLOCK)
    v = tl.load(P + s * M * N + o, s < S, other=0)
    value = tl.sum(v, 0) * ALPHA
    if BETA != 0:
        value += tl.load(I + (o // N) * IM + (o % N) * IN).to(tl.float64) * BETA
    tl.store(O + (o // N) * OM + (o % N) * ON, value)


@triton.jit
def _pack_addmm_rows(
    A,
    O,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    KP: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = i // KP, i % KP
    value = tl.load(A + row * S0 + col * S1, (row < ROWS) & (col < K), other=0)
    tl.store(O + i, value, i < ROWS * KP)


def _use_aligned_fp32(a, b, out):
    m, k = a.shape
    n = b.shape[1]
    # Bound temporary storage while amortizing packing on large ragged GEMMs.
    return (
        a.dtype == b.dtype == out.dtype == torch.float32
        and m >= 4096
        and n >= 512
        and k >= 1024
        and k % 32 != 0
        and a.stride() == (k, 1)
        and b.stride() == (1, k)
        and (m + n) * triton.cdiv(k, 32) * 32 <= 32 * 1024 * 1024
    )


def _use_short_rowmajor_fp32(bias, a, b, out, alpha, beta):
    m, k = a.shape
    n = b.shape[1]
    return (
        a.dtype == b.dtype == out.dtype == torch.float32
        and 512 <= m <= 1024
        and 512 <= n <= 2048
        and 0 < k <= 1024
        and m % 32 == 0
        and n % 64 == 0
        and k % 32 == 0
        and a.stride() == (k, 1)
        and b.stride() == (n, 1)
        and out.stride() == (n, 1)
        and bias.ndim == 1
        and bias.shape[0] == n
        and bias.stride() == (1,)
        and alpha == 1
        and beta == 1
    )


def _launch(bias, a, b, out, alpha, beta):
    m, k = a.shape
    n = b.shape[1]
    if not m or not n:
        return out
    bias_view = bias.broadcast_to((m, n))
    with torch_device_fn.device(a.device):
        aligned_k = _use_aligned_fp32(a, b, out)
        if aligned_k:
            kp = triton.cdiv(k, 32) * 32
            ap = torch.empty((m, kp), dtype=a.dtype, device=a.device)
            bp = torch.empty((n, kp), dtype=b.dtype, device=b.device)
            _pack_addmm_rows[(triton.cdiv(m * kp, 512),)](
                a, ap, m, k, kp, *a.stride(), 512, num_warps=4
            )
            _pack_addmm_rows[(triton.cdiv(n * kp, 512),)](
                b, bp, n, k, kp, b.stride(1), b.stride(0), 512, num_warps=4
            )
            a, b, k = ap, bp.t(), kp
            if m >= 4 * n:
                # Evaluate the transposed product for tall matrices. This
                # exchanges the MFMA input layouts while writing through the
                # output strides, without another allocation or copy.
                _grouped_addmm[(triton.cdiv(n, 64) * triton.cdiv(m, 64),)](
                    bp,
                    ap.t(),
                    bias,
                    out,
                    n,
                    m,
                    kp,
                    *bp.stride(),
                    *ap.t().stride(),
                    bias_view.stride(1),
                    bias_view.stride(0),
                    out.stride(1),
                    out.stride(0),
                    alpha,
                    beta,
                    64,
                    64,
                    32,
                    True,
                    num_warps=4,
                    num_stages=2,
                    num_ldmatrixes=1,
                    matrix_instr_nonkdim=16,
                    kpack=1,
                    UNROLL=1,
                    ALIGNED_K=True,
                )
                return out
        if a.dtype == torch.float32 and (
            (m == 1 and k >= 256) or (n == 1 and k >= 4096)
        ):
            bo = 1 if m == 1 else 4
            rk = min(triton.next_power_of_2(k), 4096)
            parts = triton.cdiv(k, rk)
            partial = (
                torch.empty((parts, m * n), dtype=torch.float64, device=a.device)
                if parts > 1
                else out
            )
            _precise_vector[(triton.cdiv(m * n, bo), parts)](
                a,
                b,
                bias,
                out,
                partial,
                m,
                n,
                k,
                *a.stride(),
                *b.stride(),
                *bias_view.stride(),
                *out.stride(),
                alpha,
                beta,
                bo,
                rk,
                parts > 1,
                True,
                num_warps=8 if m == 1 and k >= 4096 else 4,
                enable_fp_fusion=False,
            )
            if parts > 1:
                _precise_vector_finish[(m * n,)](
                    partial,
                    bias,
                    out,
                    m,
                    n,
                    parts,
                    *bias_view.stride(),
                    *out.stride(),
                    alpha,
                    beta,
                    triton.next_power_of_2(parts),
                    num_warps=4,
                    enable_fp_fusion=False,
                )
        elif _expanded_fp32(a, b):
            bm, bn, bk = 32, 32 if k > 32 and n <= 128 else 64, 32
            _grouped_addmm[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
                a,
                b,
                bias,
                out,
                m,
                n,
                k,
                *a.stride(),
                *b.stride(),
                *bias_view.stride(),
                *out.stride(),
                alpha,
                beta,
                bm,
                bn,
                bk,
                True,
                PARTS16=bn == 32,
                num_warps=4,
                num_stages=2,
                num_ldmatrixes=1,
                matrix_instr_nonkdim=16,
                kpack=1,
                UNROLL=1,
            )
        elif _use_dual_bf16(bias, a, b, out, alpha, beta):
            single = k == 2048
            if not single:
                from .addmm_schedule import launch

                if launch(_dual_bf16_addmm, a, b, bias, out):
                    return out
            kernel = _single_bf16_addmm if single else _dual_bf16_addmm
            kernel[(triton.cdiv(m, 256) * triton.cdiv(n, 256),)](
                a,
                b,
                bias,
                out,
                m,
                n,
                k,
                256,
                256,
                32,
                num_warps=8 if single else 16,
                num_stages=1 if single else 2,
                matrix_instr_nonkdim=16,
                kpack=2,
                optimize_epilogue=False,
            )
        elif (m <= 8 and n <= 256 and k <= 4096) or _small_fp32(a, b):
            bo = 8 if m > 8 else 4
            _gemv[(triton.cdiv(m * n, bo),)](
                a,
                b,
                bias,
                out,
                m,
                n,
                k,
                *a.stride(),
                *b.stride(),
                *bias_view.stride(),
                *out.stride(),
                alpha,
                beta,
                bo,
                triton.next_power_of_2(max(1, k)),
                num_warps=4,
            )
        else:
            precise = a.dtype == torch.float32
            bm, bn, bk = (
                (64 if m >= 4096 else 32, 64, 32) if precise else (256, 256, 32)
            )
            if _medium_fp32(a, b):
                bm, bn, bk = 32, 32, 32
            _grouped_addmm[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
                a,
                b,
                bias,
                out,
                m,
                n,
                k,
                *a.stride(),
                *b.stride(),
                *bias_view.stride(),
                *out.stride(),
                alpha,
                beta,
                bm,
                bn,
                bk,
                precise and not _use_short_rowmajor_fp32(bias, a, b, out, alpha, beta),
                num_warps=4 if precise else 16,
                num_stages=2,
                num_ldmatrixes=1,
                matrix_instr_nonkdim=16,
                kpack=1 if precise else 2,
                UNROLL=1 if precise else 2,
                PARTS16=_medium_fp32(a, b),
                ALIGNED_K=aligned_k,
            )
    return out


def addmm(bias, a, b, *, beta=1, alpha=1):
    if not _eligible(a, b):
        return _generic_addmm(bias, a, b, beta=beta, alpha=alpha)
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    out = torch.empty((a.shape[0], b.shape[1]), dtype=a.dtype, device=a.device)
    return _launch(bias, a, b, out, alpha, beta)


def addmm_out(bias, a, b, *, beta=1, alpha=1, out=None):
    if not _eligible(a, b):
        return _generic_addmm_out(bias, a, b, beta=beta, alpha=alpha, out=out)
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    if out is None:
        out = torch.empty((a.shape[0], b.shape[1]), dtype=a.dtype, device=a.device)
    assert out.shape == (a.shape[0], b.shape[1]), "Incompatible output shape"
    return _launch(bias, a, b, out, alpha, beta)

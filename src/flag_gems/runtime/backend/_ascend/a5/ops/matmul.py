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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.shape_utils import broadcast

_CORES = 32
_MIN_BLOCK = 16


@triton.jit
def _mm_fast_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SB_K: tl.constexpr,
    SB_N: tl.constexpr,
    NUM_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    B_TRANS: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m = pid % NUM_M
    pid_n = pid // NUM_M
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    if B_TRANS:
        b_ptrs = b_ptr + offs_n[:, None] * SB_N + offs_k[None, :] * SB_K
    else:
        b_ptrs = b_ptr + offs_k[:, None] * SB_K + offs_n[None, :] * SB_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if EVEN_N:
        for _ in range(0, K // BLOCK_K):
            a = tl.load(a_ptrs)
            if B_TRANS:
                b = tl.trans(tl.load(b_ptrs))
            else:
                b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * SB_K
    else:
        n_mask = offs_n < N
        for _ in range(0, K // BLOCK_K):
            a = tl.load(a_ptrs)
            if B_TRANS:
                b = tl.trans(tl.load(b_ptrs, mask=n_mask[:, None], other=0.0))
            else:
                b = tl.load(b_ptrs, mask=n_mask[None, :], other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * SB_K

    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    if EVEN_N:
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))
    else:
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=n_mask[None, :])


@triton.jit
def _mm_fast_splitk_kernel(
    a_ptr,
    b_ptr,
    p_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SB_K: tl.constexpr,
    SB_N: tl.constexpr,
    MN: tl.constexpr,
    NUM_M: tl.constexpr,
    SEG: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    B_TRANS: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m = (pid // SPLIT) % NUM_M
    pid_n = (pid // SPLIT) // NUM_M
    pid_k = pid % SPLIT

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    k_start = pid_k * (SEG * BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * K + (k_start + offs_k)[None, :]
    if B_TRANS:
        b_ptrs = b_ptr + offs_n[:, None] * SB_N + (k_start + offs_k)[None, :] * SB_K
    else:
        b_ptrs = b_ptr + (k_start + offs_k)[:, None] * SB_K + offs_n[None, :] * SB_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if EVEN_N:
        for _ in range(0, SEG):
            a = tl.load(a_ptrs)
            if B_TRANS:
                b = tl.trans(tl.load(b_ptrs))
            else:
                b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * SB_K
    else:
        n_mask = offs_n < N
        for _ in range(0, SEG):
            a = tl.load(a_ptrs)
            if B_TRANS:
                b = tl.trans(tl.load(b_ptrs, mask=n_mask[:, None], other=0.0))
            else:
                b = tl.load(b_ptrs, mask=n_mask[None, :], other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * SB_K

    p_ptrs = p_ptr + pid_k * MN + offs_m[:, None] * N + offs_n[None, :]
    if EVEN_N:
        tl.store(p_ptrs, acc)
    else:
        tl.store(p_ptrs, acc, mask=n_mask[None, :])


@triton.jit
def _mm_reduce_fast_kernel(
    p_ptr, c_ptr, MN: tl.constexpr, SPLIT: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(0, SPLIT):
        acc = acc + tl.load(p_ptr + s * MN + offs)
    tl.store(c_ptr + offs, acc.to(c_ptr.dtype.element_ty))


@triton.jit
def _mm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    sa_b,
    sa_m,
    sa_k,
    sb_b,
    sb_k,
    sb_n,
    sc_b,
    sc_m,
    sc_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    B_TRANS: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_b = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + pid_b * sa_b + offs_m[:, None] * sa_m + offs_k[None, :] * sa_k
    if B_TRANS:
        b_ptrs = b_ptr + pid_b * sb_b + offs_n[:, None] * sb_n + offs_k[None, :] * sb_k
    else:
        b_ptrs = b_ptr + pid_b * sb_b + offs_k[:, None] * sb_k + offs_n[None, :] * sb_n

    if EVEN_M:
        m_mask = tl.full((BLOCK_M,), 1, tl.int1)
    else:
        m_mask = offs_m < M
    if EVEN_N:
        n_mask = tl.full((BLOCK_N,), 1, tl.int1)
    else:
        n_mask = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kk in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = (
                tl.load(a_ptrs)
                if EVEN_M
                else tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
            )
            if B_TRANS:
                bl = (
                    tl.load(b_ptrs)
                    if EVEN_N
                    else tl.load(b_ptrs, mask=n_mask[:, None], other=0.0)
                )
                b = tl.trans(bl)
            else:
                b = (
                    tl.load(b_ptrs)
                    if EVEN_N
                    else tl.load(b_ptrs, mask=n_mask[None, :], other=0.0)
                )
        else:
            kmask = (kk * BLOCK_K + offs_k) < K
            a = tl.load(a_ptrs, mask=m_mask[:, None] & kmask[None, :], other=0.0)
            if B_TRANS:
                b = tl.trans(
                    tl.load(b_ptrs, mask=n_mask[:, None] & kmask[None, :], other=0.0)
                )
            else:
                b = tl.load(b_ptrs, mask=kmask[:, None] & n_mask[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * sa_k
        b_ptrs += BLOCK_K * sb_k

    c_ptrs = c_ptr + pid_b * sc_b + offs_m[:, None] * sc_m + offs_n[None, :] * sc_n
    if EVEN_M and EVEN_N:
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))
    else:
        tl.store(
            c_ptrs,
            acc.to(c_ptr.dtype.element_ty),
            mask=m_mask[:, None] & n_mask[None, :],
        )


@triton.jit
def _mm_generic_splitk_kernel(
    a_ptr,
    b_ptr,
    p_ptr,
    M,
    N,
    K,
    sa_m,
    sa_k,
    sb_k,
    sb_n,
    MN,
    SPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    B_TRANS: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    pid_m = (pid // SPLIT) % num_m
    pid_n = (pid // SPLIT) // num_m
    pid_k = pid % SPLIT

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    seg = tl.cdiv(tl.cdiv(K, BLOCK_K), SPLIT)
    k_start = pid_k * seg * BLOCK_K

    a_ptrs = a_ptr + offs_m[:, None] * sa_m + (k_start + offs_k)[None, :] * sa_k
    if B_TRANS:
        b_ptrs = b_ptr + offs_n[:, None] * sb_n + (k_start + offs_k)[None, :] * sb_k
    else:
        b_ptrs = b_ptr + (k_start + offs_k)[:, None] * sb_k + offs_n[None, :] * sb_n

    if EVEN_M:
        m_mask = tl.full((BLOCK_M,), 1, tl.int1)
    else:
        m_mask = offs_m < M
    if EVEN_N:
        n_mask = tl.full((BLOCK_N,), 1, tl.int1)
    else:
        n_mask = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in range(0, seg):
        if EVEN_K:
            a = (
                tl.load(a_ptrs)
                if EVEN_M
                else tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
            )
            if B_TRANS:
                bl = (
                    tl.load(b_ptrs)
                    if EVEN_N
                    else tl.load(b_ptrs, mask=n_mask[:, None], other=0.0)
                )
                b = tl.trans(bl)
            else:
                b = (
                    tl.load(b_ptrs)
                    if EVEN_N
                    else tl.load(b_ptrs, mask=n_mask[None, :], other=0.0)
                )
        else:
            kmask = (k_start + i * BLOCK_K + offs_k) < K
            a = tl.load(a_ptrs, mask=m_mask[:, None] & kmask[None, :], other=0.0)
            if B_TRANS:
                b = tl.trans(
                    tl.load(b_ptrs, mask=n_mask[:, None] & kmask[None, :], other=0.0)
                )
            else:
                b = tl.load(b_ptrs, mask=kmask[:, None] & n_mask[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * sa_k
        b_ptrs += BLOCK_K * sb_k

    p_ptrs = p_ptr + pid_k * MN + offs_m[:, None] * N + offs_n[None, :]
    if EVEN_M and EVEN_N:
        tl.store(p_ptrs, acc)
    else:
        tl.store(p_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _mm_reduce_kernel(p_ptr, c_ptr, MN, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(0, SPLIT):
        acc = acc + tl.load(p_ptr + s * MN + offs, mask=mask, other=0.0)
    tl.store(c_ptr + offs, acc.to(c_ptr.dtype.element_ty), mask=mask)


_FAST_BLOCK_N = 128
_FAST_BLOCK_K = 128
_FAST_DTYPES = (torch.bfloat16, torch.float16)
_FAST_SMALL_M = 8192
# (BLOCK_M, NPROG, num_warps). Program p owns the contiguous row band
# [p*BASE, (p+1)*BASE) with BASE = trip*BLOCK_M; the tile count is exactly
# NPROG*trip so the descriptor walk needs no bounds handling. Two lexically
# independent accumulators per unrolled pair of tiles keep tile i's drain off
# tile i+1's accumulator (L0C holds two 128x128 fp32 accumulators).
_FAST_CFG_SMALL = (64, 8, 4)
_FAST_CFG_LARGE = (128, 32, 2)


@triton.jit
def _mm_persist_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M_TOTAL: tl.constexpr,
    TRIP: tl.constexpr,
    BASE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    b_bp = tl.make_block_ptr(
        base=b_ptr,
        shape=(BLOCK_K, BLOCK_N),
        strides=(BLOCK_N, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )
    a_bp = tl.make_block_ptr(
        base=a_ptr,
        shape=(M_TOTAL, BLOCK_K),
        strides=(BLOCK_K, 1),
        offsets=(pid * BASE, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    c_bp = tl.make_block_ptr(
        base=c_ptr,
        shape=(M_TOTAL, BLOCK_N),
        strides=(BLOCK_N, 1),
        offsets=(pid * BASE, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    # K == BLOCK_K == N == BLOCK_N, so the whole 32 KiB B panel is loop
    # invariant: one descriptor load per program, then a pure tile walk.
    b = tl.load(b_bp)
    a_bp1 = tl.advance(a_bp, (BLOCK_M, 0))
    c_bp1 = tl.advance(c_bp, (BLOCK_M, 0))
    for _ in range(TRIP // 2):
        a0 = tl.load(a_bp)
        a1 = tl.load(a_bp1)
        acc0 = tl.dot(a0, b)
        tl.store(c_bp, acc0.to(c_ptr.dtype.element_ty))
        acc1 = tl.dot(a1, b)
        tl.store(c_bp1, acc1.to(c_ptr.dtype.element_ty))
        a_bp = tl.advance(a_bp, (2 * BLOCK_M, 0))
        a_bp1 = tl.advance(a_bp1, (2 * BLOCK_M, 0))
        c_bp = tl.advance(c_bp, (2 * BLOCK_M, 0))
        c_bp1 = tl.advance(c_bp1, (2 * BLOCK_M, 0))


def _fast_cfg(M):
    return _FAST_CFG_SMALL if M < _FAST_SMALL_M else _FAST_CFG_LARGE


def _fast_ok(a, b, out, M, N, K):
    """Runtime-only rule: dtype/shape/stride facts, exact tile coverage."""
    if N != _FAST_BLOCK_N or K != _FAST_BLOCK_K:
        return False
    if a.dtype != b.dtype or a.dtype not in _FAST_DTYPES:
        return False
    if not (a.is_contiguous() and b.is_contiguous() and out.is_contiguous()):
        return False
    bm, nprog, _ = _fast_cfg(M)
    if M % bm:
        return False
    tiles = M // bm
    if tiles % nprog:
        return False
    if (tiles // nprog) % 2:
        return False
    return True


def _launch_fast(a, b, out, M):
    bm, nprog, nw = _fast_cfg(M)
    trip = (M // bm) // nprog
    _mm_persist_kernel[(nprog,)](
        a,
        b,
        out,
        M_TOTAL=M,
        TRIP=trip,
        BASE=trip * bm,
        BLOCK_M=bm,
        BLOCK_N=_FAST_BLOCK_N,
        BLOCK_K=_FAST_BLOCK_K,
        num_warps=nw,
    )


def _pick_split(K, dtype):
    """Match the reference's fp32 K-reduction granularity.

    Measured on this target: the ATen reference reduces a long fp32 K by ~1024
    element partial sums. A single fused accumulator chain costs ~47 ULP at
    K=16384 (and is BLOCK_K invariant), while this granularity reproduces the
    reference bit-exactly, so the split is a correctness requirement.
    """
    if dtype == torch.float32 and K >= 2048:
        return max(1, min(16, K // 1024))
    return 1


def _pick_blocks(M, N, K, split):
    bm = min(128, max(_MIN_BLOCK, triton.next_power_of_2(M)))
    bn = min(128, max(_MIN_BLOCK, triton.next_power_of_2(N)))
    while True:
        progs = triton.cdiv(M, bm) * triton.cdiv(N, bn) * split
        if progs >= _CORES:
            break
        if bn > _MIN_BLOCK and bn >= bm:
            bn //= 2
        elif bm > _MIN_BLOCK:
            bm //= 2
        elif bn > _MIN_BLOCK:
            bn //= 2
        else:
            break
    # The B panel is re-read once per M block, so B traffic is
    # (M/BLOCK_M)*N*K and only BLOCK_M reduces it. Grow BLOCK_M while at
    # least two full waves remain (so the tile count still hides DMA latency)
    # and the fp32 accumulator stays within one cube accumulator bank.
    if split == 1:
        # The tile count that still hides DMA latency is two waves while the
        # grid is already wider than the machine, but such a requirement makes
        # a tiny shape stop at one block per core, where there is no
        # cross-block overlap left to preserve and the only remaining lever on
        # per-block cost is a larger tile. So the floor becomes half a wave
        # once the grid is already at most one block per core. (Measured on the
        # g35 1024x128x128 bf16 row: doubling that grid cost +0.74 us, while
        # doubling the k-loop trips was flat, so the halved-grid direction is
        # the untested one.)
        floor_progs = (
            2 * _CORES
            if triton.cdiv(M, bm) * triton.cdiv(N, bn) > _CORES
            else _CORES // 2
        )
        while True:
            nbm = bm * 2
            if nbm > 256 or M % nbm != 0:
                break
            if triton.cdiv(M, nbm) * triton.cdiv(N, bn) < floor_progs:
                break
            if nbm * bn * 4 > 131072:
                break
            bm = nbm
    bk = min(256, max(16, triton.next_power_of_2(min(K, 256))))
    if split > 1:
        # The split-K grid already provides many tiles per core, and the
        # measured axis there is the per-core outer iteration count, where
        # fewer is better (BLOCK_M=128 beat BLOCK_M=64). Grow BLOCK_M while M
        # divides and the accumulator stays small; the wider tile no longer
        # fits the cube buffer with the largest BLOCK_K, so halve that.
        while bm < 512 and M % (bm * 2) == 0 and (bm * 2) * bn * 4 <= 65536:
            bm *= 2
        if bm >= 512:
            bk = min(bk, 64)
        elif bm >= 256:
            bk = min(bk, 128)
    return bm, bn, bk


def _b_layout(b, K, N):
    """(B_TRANS, sb_k, sb_n) for a 2-D operand, or None if it is neither
    contiguous nor a plain transposed view of a contiguous buffer."""
    sk, sn = b.stride(0), b.stride(1)
    if sn == 1 and sk == N:
        return False, N, 1
    if sk == 1 and sn == K:
        return True, 1, K
    return None


def _launch_reduce(part, out, mn, split):
    rblk = 128
    if mn % rblk == 0:
        _mm_reduce_fast_kernel[(mn // rblk,)](part, out, MN=mn, SPLIT=split, BLOCK=rblk)
    else:
        _mm_reduce_kernel[(triton.cdiv(mn, rblk),)](
            part, out, mn, SPLIT=split, BLOCK=rblk
        )


def _matmul2d(a, b, out):
    M, K = a.shape
    N = b.shape[1]
    if _fast_ok(a, b, out, M, N, K):
        _launch_fast(a, b, out, M)
        return
    split = _pick_split(K, a.dtype)
    bm, bn, bk = _pick_blocks(M, N, K, split)
    num_m = triton.cdiv(M, bm)
    num_n = triton.cdiv(N, bn)
    even_n = N % bn == 0
    layout = _b_layout(b, K, N)

    fast = (
        layout is not None
        and a.stride(1) == 1
        and a.stride(0) == K
        and out.stride(1) == 1
        and out.stride(0) == N
        and M % bm == 0
        and K % bk == 0
        and (split == 1 or (K // bk) % split == 0)
    )

    if fast:
        b_trans, sb_k, sb_n = layout
        if split == 1:
            _mm_fast_kernel[(num_m * num_n,)](
                a,
                b,
                out,
                M=M,
                N=N,
                K=K,
                SB_K=sb_k,
                SB_N=sb_n,
                NUM_M=num_m,
                BLOCK_M=bm,
                BLOCK_N=bn,
                BLOCK_K=bk,
                B_TRANS=b_trans,
                EVEN_N=even_n,
            )
            return
        mn = M * N
        part = torch.empty((split, M, N), dtype=torch.float32, device=out.device)
        _mm_fast_splitk_kernel[(num_m * num_n * split,)](
            a,
            b,
            part,
            M=M,
            N=N,
            K=K,
            SB_K=sb_k,
            SB_N=sb_n,
            MN=mn,
            NUM_M=num_m,
            SEG=(K // bk) // split,
            SPLIT=split,
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
            B_TRANS=b_trans,
            EVEN_N=even_n,
        )
        _launch_reduce(part, out, mn, split)
        return

    b_trans = bool(b.stride(0) == 1 and b.stride(1) != 1)
    even_k = triton.cdiv(K, bk) * bk == K
    even_m = M % bm == 0
    if split == 1:
        _mm_kernel[(num_m * num_n, 1)](
            a,
            b,
            out,
            M,
            N,
            K,
            0,
            a.stride(0),
            a.stride(1),
            0,
            b.stride(0),
            b.stride(1),
            0,
            out.stride(0),
            out.stride(1),
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
            B_TRANS=b_trans,
            EVEN_M=even_m,
            EVEN_N=even_n,
            EVEN_K=even_k,
        )
        return

    mn = M * N
    part = torch.empty((split, M, N), dtype=torch.float32, device=out.device)
    _mm_generic_splitk_kernel[(num_m * num_n * split,)](
        a,
        b,
        part,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        mn,
        SPLIT=split,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        B_TRANS=b_trans,
        EVEN_M=even_m,
        EVEN_N=even_n,
        EVEN_K=even_k,
    )
    _launch_reduce(part, out, mn, split)


def _launch(self, other):
    k, n = other.shape
    flat = self.reshape(-1, k)
    out = torch.empty((flat.shape[0], n), device=self.device, dtype=self.dtype)
    _matmul2d(flat, other, out)
    return out.view(*self.shape[:-1], n)


def matmul(self, other):
    """A5 matmul with a dense matrix fast path and batch-broadcast decomposition."""
    if (
        self.ndim == 0
        or other.ndim == 0
        or self.shape[-1] != (other.shape[-2] if other.ndim >= 2 else other.shape[0])
    ):
        raise RuntimeError("matmul expects compatible nonscalar tensors")
    if self.dtype != other.dtype or self.device != other.device:
        raise RuntimeError("matmul operands must have matching dtype and device")
    if (
        other.ndim != 2
        or self.ndim < 2
        or not self.is_contiguous()
        or not self.numel()
        or not other.numel()
        or self.dtype not in (torch.float32, torch.bfloat16)
        or not (other.is_contiguous() or other.T.is_contiguous())
    ):
        import flag_gems

        a = self.unsqueeze(0) if self.ndim == 1 else self
        b = other.unsqueeze(-1) if other.ndim == 1 else other
        batch = broadcast(a.shape[:-2], b.shape[:-2])
        m, k = a.shape[-2:]
        n = b.shape[-1]
        if not batch:
            if a.dtype in (torch.bfloat16, torch.float32):
                if not m or not n or not k:
                    result = torch.zeros((m, n), dtype=a.dtype, device=a.device)
                else:
                    with torch_device_fn.device(a.device):
                        result = _launch(
                            a.contiguous(),
                            (
                                b
                                if b.is_contiguous() or b.T.is_contiguous()
                                else b.contiguous()
                            ),
                        )
            else:
                result = flag_gems.mm(a, b)
        else:
            import math

            count = math.prod(batch)
            left = a.expand(*batch, m, k).reshape(count, m, k)
            right = b.expand(*batch, k, n).reshape(count, k, n)
            result = flag_gems.bmm(left, right).reshape(*batch, m, n)
        if self.ndim == 1:
            result = result.squeeze(-2)
        if other.ndim == 1:
            result = result.squeeze(-1)
        return result
    if self.device.type != "npu" or other.device != self.device:
        raise ValueError("expected inputs on the same NPU")
    with torch_device_fn.device(self.device):
        return _launch(self, other)

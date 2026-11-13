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
import math
import time

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


# The enflame external toolchain (gcu-compiler-compile / the llvm-ir
# converter script) segfaults intermittently under high host load while
# compiling Triton kernels (compiler_problems P02 family; ~44% of fresh
# compiles crash at load >= 24, observed again on this op's gate run).
# Failed compiles are not cached, so re-attempting the launch recompiles
# and usually succeeds; once a kernel is compiled the retry is a no-op.
_COMPILE_RETRY_ATTEMPTS = 6
_COMPILE_RETRY_DELAY_S = 1.0


def _launch_retry(launch):
    for attempt in range(_COMPILE_RETRY_ATTEMPTS):
        try:
            return launch()
        except Exception:
            if attempt == _COMPILE_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(_COMPILE_RETRY_DELAY_S)


# ---------------------------------------------------------------------------
# Whole-matrix register-tile Cholesky (N <= 64, one program per batch matrix).
#
# The factor is produced by an unblocked right-looking loop of rank-1
# updates: every step extracts the current pivot column from the register
# tile, normalises it and applies the rank-1 update before committing the
# column, so past columns are only ever touched in their (zero) upper part
# and future columns receive the exact Schur update.
#
# Only the lower triangle of the input is referenced (matching
# torch.linalg.cholesky semantics) and the legacy ``+eps`` diagonal
# stabilisation is folded into the load, so the host wrapper performs no
# tensor math at all.  Each step stores the full column -- the entries
# above the pivot are exactly zero -- which allows ``torch.empty_like``
# output allocation.  num_warps=1 is measurably fastest for this serial
# chain on GCU (the per-step column extraction dominates).
# ---------------------------------------------------------------------------
@triton.jit
def _potrf_tile(
    A_ptr,
    L_ptr,
    N,
    bsa,
    sma,
    ska,
    bsl,
    sml,
    skl,
    eps,
    FIRST: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NSTEPS: tl.constexpr,
):
    pid_b = tl.program_id(0)
    base_a = pid_b * bsa
    base_l = pid_b * bsl
    r = tl.arange(0, BLOCK_N)
    ri = r[:, None]
    cj = r[None, :]
    inb = r < N
    D = tl.load(
        A_ptr + base_a + ri * sma + cj * ska,
        mask=inb[:, None] & inb[None, :] & (ri >= cj),
        other=0.0,
    )
    if FIRST:
        D = D + tl.where(ri == cj, eps, 0.0)
    for j in tl.static_range(NSTEPS):
        col = tl.sum(tl.where(cj == j, D, 0.0), axis=1)
        pivot = tl.sum(tl.where(r == j, col, 0.0))
        d = tl.sqrt(pivot)
        colv = tl.where(r > j, col / d, tl.where(r == j, d, 0.0))
        # Rank-1 update before committing column j: columns < j are only
        # touched in their (zero) upper part, columns > j receive the exact
        # Schur update, and column j itself is dead after the store below.
        D = D - colv[:, None] * colv[None, :]
        tl.store(L_ptr + base_l + r * sml + j * skl, colv, mask=inb)


# ---------------------------------------------------------------------------
# Symmetrise the input and apply the legacy +eps diagonal shift in one
# launch:  W = (A + A^T) / 2 + eps * I.  This replaces the host-side
# prelude of the legacy implementation (transpose + add + mul + eye + mul +
# add), which cost ~0.4 ms per call when those aten ops are routed through
# the FlagGems python wrappers.
# ---------------------------------------------------------------------------
@triton.jit
def _symmetrize_eps(
    A_ptr,
    W_ptr,
    N,
    bsa,
    sma,
    ska,
    bsw,
    smw,
    skw,
    eps,
    BLOCK: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_c = tl.program_id(2)
    base_a = pid_b * bsa
    base_w = pid_b * bsw
    rows = pid_r * BLOCK + tl.arange(0, BLOCK)
    cols = pid_c * BLOCK + tl.arange(0, BLOCK)
    m = (rows[:, None] < N) & (cols[None, :] < N)
    a = tl.load(
        A_ptr + base_a + rows[:, None] * sma + cols[None, :] * ska, mask=m, other=0.0
    )
    at = tl.load(
        A_ptr + base_a + cols[:, None] * sma + rows[None, :] * ska,
        mask=(cols[:, None] < N) & (rows[None, :] < N),
        other=0.0,
    )
    w = (a + tl.trans(at)) * 0.5 + tl.where(rows[:, None] == cols[None, :], eps, 0.0)
    tl.store(W_ptr + base_w + rows[:, None] * smw + cols[None, :] * skw, w, mask=m)


# ---------------------------------------------------------------------------
# Multi-launch blocked path (N > 64): column-block Cholesky with NB-sized
# diagonal blocks.  No explicit copy of A is needed: block column 0 reads
# the symmetrised work matrix and the first syrk materializes the updated
# trailing matrix into scratch S; all later phases read S.
# ---------------------------------------------------------------------------
@triton.jit
def _potrf_diag(
    src_ptr,
    L_ptr,
    N,
    bss,
    sms,
    sks,
    bsl,
    sml,
    skl,
    s,
    NB: tl.constexpr,
    FP64: tl.constexpr,
):
    # Right-looking factor of the NB x NB diagonal block: the trailing block
    # D is kept in registers and updated with rank-1 products, so each column
    # step needs only one full-tile column extraction plus one rank-1 FMA
    # (no row extraction, no separate contrib reduction).
    pid_b = tl.program_id(0)
    base_s = pid_b * bss
    base_l = pid_b * bsl
    sb = base_s + s * sms + s * sks
    lb = base_l + s * sml + s * skl
    offs = tl.arange(0, NB)
    inb = (s + offs) < N
    D = tl.load(
        src_ptr + sb + offs[:, None] * sms + offs[None, :] * sks,
        mask=inb[:, None] & inb[None, :],
        other=0.0,
    )
    for k in tl.static_range(NB):
        col = tl.sum(tl.where(offs[None, :] == k, D, 0.0), axis=1)
        diag = tl.sqrt(tl.sum(tl.where(offs == k, col, 0.0)))
        colv = tl.where(offs > k, col / diag, tl.where(offs == k, diag, 0.0))
        D = D - colv[:, None] * colv[None, :]
        tl.store(L_ptr + lb + offs * sml + k * skl, colv, mask=inb & (offs >= k))


@triton.jit
def _trsm_panel(
    src_ptr,
    L_ptr,
    N,
    bss,
    sms,
    sks,
    bsl,
    sml,
    skl,
    s,
    NB: tl.constexpr,
    TM: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    base_s = pid_b * bss
    base_l = pid_b * bsl
    lb = base_l + s * sml + s * skl
    offs = tl.arange(0, NB)
    rows = s + NB + pid_t * TM + tl.arange(0, TM)
    rmask = rows < N
    A21 = tl.load(
        src_ptr + base_s + rows[:, None] * sms + (s + offs)[None, :] * sks,
        mask=rmask[:, None],
        other=0.0,
    )
    X = A21
    for c in range(NB):
        rowc = tl.load(L_ptr + lb + c * sml + offs * skl)
        diag_c = tl.load(L_ptr + lb + c * sml + c * skl)
        contrib = tl.sum(tl.where(offs[None, :] < c, X * rowc[None, :], 0.0), axis=1)
        A21_c = tl.sum(tl.where(offs[None, :] == c, A21, 0.0), axis=1)
        Xc = (A21_c - contrib) / diag_c
        X = tl.where(offs[None, :] == c, Xc[:, None], X)
        tl.store(L_ptr + base_l + rows * sml + (s + c) * skl, Xc, mask=rmask)


@triton.jit
def _syrk_update(
    src_ptr,
    dst_ptr,
    L_ptr,
    N,
    bss,
    sms,
    sks,
    bsd,
    smd,
    skd,
    bsl,
    sml,
    skl,
    s,
    NB: tl.constexpr,
    TM: tl.constexpr,
    TN: tl.constexpr,
    FP64: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_c = tl.program_id(2)
    base_s = pid_b * bss
    base_d = pid_b * bsd
    base_l = pid_b * bsl
    offs = tl.arange(0, NB)
    rows = s + NB + pid_r * TM + tl.arange(0, TM)
    cols = s + NB + pid_c * TN + tl.arange(0, TN)
    rmask = rows < N
    cmask = cols < N
    lhs = tl.load(
        L_ptr + base_l + rows[:, None] * sml + (s + offs)[None, :] * skl,
        mask=rmask[:, None],
        other=0.0,
    )
    rhs = tl.load(
        L_ptr + base_l + cols[:, None] * sml + (s + offs)[None, :] * skl,
        mask=cmask[:, None],
        other=0.0,
    )
    if FP64:
        acc = tl.zeros((TM, TN), dtype=tl.float64)
        for nb in tl.static_range(NB):
            lcol = tl.sum(tl.where(offs[None, :] == nb, lhs, 0.0), axis=1)
            rcol = tl.sum(tl.where(offs[None, :] == nb, rhs, 0.0), axis=1)
            acc += lcol[:, None] * rcol[None, :]
    else:
        acc = tl.dot(lhs, tl.trans(rhs), input_precision="ieee")
    a22 = tl.load(
        src_ptr + base_s + rows[:, None] * sms + cols[None, :] * sks,
        mask=rmask[:, None] & cmask[None, :],
        other=0.0,
    )
    tl.store(
        dst_ptr + base_d + rows[:, None] * smd + cols[None, :] * skd,
        a22 - acc,
        mask=rmask[:, None] & cmask[None, :],
    )


# ---------------------------------------------------------------------------
# Zero-fill the strict upper triangle of L (the factor kernels only write
# the lower triangle in the blocked path, and the output is allocated with
# torch.empty_like).
# ---------------------------------------------------------------------------
@triton.jit
def _zero_upper(L_ptr, N, bsl, sml, skl, BLOCK: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_c = tl.program_id(2)
    if pid_r * BLOCK >= (pid_c + 1) * BLOCK:
        return
    rows = pid_r * BLOCK + tl.arange(0, BLOCK)
    cols = pid_c * BLOCK + tl.arange(0, BLOCK)
    m = (rows[:, None] < cols[None, :]) & (rows[:, None] < N) & (cols[None, :] < N)
    z = tl.zeros((BLOCK, BLOCK), dtype=L_ptr.dtype.element_ty)
    tl.store(L_ptr + pid_b * bsl + rows[:, None] * sml + cols[None, :] * skl, z, mask=m)


# ---------------------------------------------------------------------------
# upper=True: U = L^T (full-square store, works on empty output)
# ---------------------------------------------------------------------------
@triton.jit
def _transpose_kernel(
    L_ptr, U_ptr, N, bsl, sml, skl, bsu, smu, sku, BLOCK: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_c = tl.program_id(2)
    base_l = pid_b * bsl
    base_u = pid_b * bsu
    rows = pid_r * BLOCK + tl.arange(0, BLOCK)
    cols = pid_c * BLOCK + tl.arange(0, BLOCK)
    m = (rows[:, None] < N) & (cols[None, :] < N)
    val = tl.load(
        L_ptr + base_l + cols[None, :] * sml + rows[:, None] * skl, mask=m, other=0.0
    )
    tl.store(U_ptr + base_u + rows[:, None] * smu + cols[None, :] * sku, val, mask=m)


# ---------------------------------------------------------------------------
# Host wrapper
# ---------------------------------------------------------------------------
_MAX_TILE_N = 64


def _strides(t):
    if t.dim() == 2:
        return 0, t.stride(-2), t.stride(-1)
    return t.stride(-3), t.stride(-2), t.stride(-1)


def linalg_cholesky(A, upper=False):
    """Cholesky factorisation on Enflame GCU."""
    logger.debug("GEMS_ENFLAME LINALG_CHOLESKY")
    if not isinstance(A, torch.Tensor):
        A = torch.as_tensor(A)
    if A.dtype not in (torch.float32, torch.float64):
        raise TypeError("linalg_cholesky only supports float32 and float64")
    if A.dim() < 2:
        raise ValueError("A must be at least 2D")
    if A.shape[-2] != A.shape[-1]:
        raise ValueError("A must be a square matrix")
    if A.numel() == 0:
        return A

    up = bool(upper)
    fp64 = A.dtype == torch.float64
    N = A.shape[-1]
    batch = 1 if A.dim() == 2 else math.prod(A.shape[:-2])
    # Preserve the backend's established behaviour for approximately
    # symmetric or ill-conditioned benchmark inputs (A + eps * I shift).
    eps = 1e-5

    bsa, sma, ska = _strides(A)

    if N <= _MAX_TILE_N:
        # Single launch: the whole factor stays in one register tile.
        L = torch.empty_like(A)
        bsl, sml, skl = _strides(L)
        _launch_retry(
            lambda: _potrf_tile[(batch,)](
                A,
                L,
                N,
                bsa,
                sma,
                ska,
                bsl,
                sml,
                skl,
                eps,
                FIRST=True,
                BLOCK_N=triton.next_power_of_2(N),
                NSTEPS=N,
                num_warps=1,
            )
        )
    else:
        # Multi-launch blocked path: symmetrise + eps-shift once, then
        # factor NB-sized diagonal blocks, solve the panel below them and
        # update the trailing matrix into scratch.
        W = torch.empty_like(A)
        bsw, smw, skw = _strides(W)
        nt = triton.cdiv(N, 64)
        _launch_retry(
            lambda: _symmetrize_eps[(batch, nt, nt)](
                A, W, N, bsa, sma, ska, bsw, smw, skw, eps, BLOCK=64, num_warps=4
            )
        )
        L = torch.empty_like(A)
        S = torch.empty_like(A)
        bsl, sml, skl = _strides(L)
        bss, sms, sks = _strides(S)
        _launch_retry(
            lambda: _zero_upper[(batch, nt, nt)](
                L, N, bsl, sml, skl, BLOCK=64, num_warps=4
            )
        )
        NB = 32
        TM = 32
        TN = 32
        for jb in range((N + NB - 1) // NB):
            s = jb * NB
            M = N - s - NB
            if jb == 0:
                src_ptr, sb_, sm_, sk_ = W, bsw, smw, skw
            else:
                src_ptr, sb_, sm_, sk_ = S, bss, sms, sks
            _launch_retry(
                lambda: _potrf_diag[(batch,)](
                    src_ptr,
                    L,
                    N,
                    sb_,
                    sm_,
                    sk_,
                    bsl,
                    sml,
                    skl,
                    s,
                    NB=NB,
                    FP64=fp64,
                    num_warps=1,
                )
            )
            if M > 0:
                nrt = triton.cdiv(M, TM)
                nct = triton.cdiv(M, TN)
                _launch_retry(
                    lambda: _trsm_panel[(batch, nrt)](
                        src_ptr,
                        L,
                        N,
                        sb_,
                        sm_,
                        sk_,
                        bsl,
                        sml,
                        skl,
                        s,
                        NB=NB,
                        TM=TM,
                        num_warps=8,
                    )
                )
                _launch_retry(
                    lambda: _syrk_update[(batch, nrt, nct)](
                        src_ptr,
                        S,
                        L,
                        N,
                        sb_,
                        sm_,
                        sk_,
                        bss,
                        sms,
                        sks,
                        bsl,
                        sml,
                        skl,
                        s,
                        NB=NB,
                        TM=TM,
                        TN=TN,
                        FP64=fp64,
                        num_warps=8,
                    )
                )

    if up:
        out = torch.empty_like(A)
        bsu, smu, sku = _strides(out)
        nt = triton.cdiv(N, 64)
        _launch_retry(
            lambda: _transpose_kernel[(batch, nt, nt)](
                L, out, N, bsl, sml, skl, bsu, smu, sku, BLOCK=64, num_warps=4
            )
        )
        return out
    return L

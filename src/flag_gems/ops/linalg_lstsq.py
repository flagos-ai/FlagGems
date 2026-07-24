# Copyright 2026, The FlagOS Contributors.
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
"""linalg_lstsq via Householder TSQR on the augmented matrix [A | B].

Scope mirrors PyTorch's CUDA path (driver="gels"), real float32/float64, both
overdetermined (m >= n, TSQR) and underdetermined (m < n, min-norm via QR of
A^T); rank / singular_values are not produced (returned empty, as gels does).
Anything outside that - complex, a non-gels driver, or shapes beyond the native
tile / register-spill ceiling - falls back to the reference implementation
on CPU.

Method: QR([A | B]) = [[R_A, C], [0, R_D]]; back-substitute R_A X = C. Q is
never formed, so there is no ormqr step (the serial bottleneck in PyTorch's CUDA
composed path). Residual per RHS is ||R_D[:,k]||^2, free from the same factor.
Three stages: per-chunk panel QR -> tree-reduction combine -> back-substitution.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Row budget for any single stacked block; drives the reduction fan-in.
# BLOCK_S * BLOCK_NC * 4 bytes is the SRAM/UB footprint.
_TARGET_STACK_ROWS = 256
# Byte budget for a panel tile, used by the block_m heuristic.
_TARGET_TILE_BYTES = 96 * 1024
# Tall-path routing ceiling on NC = n + nrhs: NC <= ceiling takes the
# monolithic kernel, above it the blocked TSQR path (both native).
# fp32 (measured on H20): the monolithic tile spills registers once BLOCK_NC
# reaches 256; NC<=128 beats torch-GPU, NC=160/256 spills to 230-370ms.
# fp64 (measured on H20, probe_f64.py, 2026-07-23): the 8-byte tile makes the
# monolithic kernel 3.5-10x SLOWER than blocked at EVERY NC (already 3.7x at
# NC=33) and it exhausts shared memory outright at NC>=129 — so fp64 tall
# always routes to the blocked path (ceiling 0).
_TALL_MAX_NC_F32 = 128
_TALL_MAX_NC_F64 = 0

# native compute dtypes (torch supports fp32/fp64 gels on CUDA)
_SUPPORTED_DTYPES = (torch.float32, torch.float64)


def _tl_dtype(torch_dtype):
    """Triton compute dtype for a supported torch float dtype."""
    return tl.float64 if torch_dtype == torch.float64 else tl.float32


# ---------------------------------------------------------------------------
# Shared device function: in-register Householder QR of a (BLOCK_R, BLOCK_C)
# tile. Returns the tile with R in its upper triangle (rows 0..NC-1).
# ---------------------------------------------------------------------------
@triton.jit
def _householder_qr(V, NC, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    row = tl.arange(0, BLOCK_R)
    col = tl.arange(0, BLOCK_C)

    # Dynamic loop (NOT unrolled): every use of j below is a mask, which works
    # with a runtime j. Unrolling was measured strictly worse: superlinear
    # compile times and slower runtime from instruction-cache pressure.
    for j in range(NC):
        # extract column j, zeroing rows above the diagonal (select-and-reduce,
        # since Triton has no dynamic column slice)
        x = tl.sum(tl.where(col[None, :] == j, V, 0.0), axis=1)
        x = tl.where(row >= j, x, 0.0)

        # reflector v = x - alpha*e1; sign opposite the pivot avoids cancellation
        norm_x = tl.sqrt(tl.sum(x * x, axis=0))
        x_j = tl.sum(tl.where(row == j, x, 0.0), axis=0)
        sign = tl.where(x_j >= 0.0, 1.0, -1.0)
        alpha = -sign * norm_x

        v = tl.where(row == j, x - alpha, x)
        v = tl.where(row >= j, v, 0.0)

        vtv = tl.sum(v * v, axis=0)
        # zero column -> identity reflector; beta = 0 makes the update a no-op
        beta = tl.where(vtv > 0.0, 2.0 / tl.where(vtv > 0.0, vtv, 1.0), 0.0)

        # apply H = I - beta*v*v^T to the trailing submatrix. v is zero for
        # rows < j, so the row restriction is implicit; only cols >= j needed.
        w = tl.sum(v[:, None] * V, axis=0)
        w = tl.where(col >= j, w, 0.0)
        V = V - beta * v[:, None] * w[None, :]

    return V


# ---------------------------------------------------------------------------
# Stage 1: QR of each row-chunk of the augmented matrix [A | B].
# ---------------------------------------------------------------------------
@libentry()
@triton.jit
def _panel_qr_kernel(
    A_ptr,
    B_ptr,
    R_ptr,
    M,
    stride_ab,
    stride_am,
    stride_an,
    stride_bb,
    stride_bm,
    stride_br,
    stride_rb,
    stride_rc,
    stride_ri,
    stride_rj,
    N,
    NC,
    BLOCK_M: tl.constexpr,
    BLOCK_NC: tl.constexpr,
    COMPUTE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    row = tl.arange(0, BLOCK_M)
    col = tl.arange(0, BLOCK_NC)
    m0 = pid_c * BLOCK_M
    rows_valid = (m0 + row) < M

    a_off = (
        pid_b * stride_ab + (m0 + row)[:, None] * stride_am + col[None, :] * stride_an
    )
    a_mask = rows_valid[:, None] & (col[None, :] < N)
    V = tl.load(A_ptr + a_off, mask=a_mask, other=0.0).to(COMPUTE)

    # B occupies columns [N, NC). rhs_idx clamped so masked lanes never form a
    # negative offset.
    rhs_idx = tl.maximum(col - N, 0)
    b_off = (
        pid_b * stride_bb
        + (m0 + row)[:, None] * stride_bm
        + rhs_idx[None, :] * stride_br
    )
    b_mask = rows_valid[:, None] & (col[None, :] >= N) & (col[None, :] < NC)
    Bblk = tl.load(B_ptr + b_off, mask=b_mask, other=0.0).to(COMPUTE)
    V = tl.where(col[None, :] >= N, Bblk, V)

    # Rows past M are zero-padded; zero rows do not change the QR factor.
    V = _householder_qr(V, NC, BLOCK_M, BLOCK_NC)

    r_off = (
        pid_b * stride_rb
        + pid_c * stride_rc
        + row[:, None] * stride_ri
        + col[None, :] * stride_rj
    )
    r_mask = (row[:, None] < NC) & (col[None, :] < NC)
    tl.store(R_ptr + r_off, tl.where(row[:, None] <= col[None, :], V, 0.0), mask=r_mask)


# ---------------------------------------------------------------------------
# Stage 2: fold G consecutive R-factors into one. Launched once per tree level.
# ---------------------------------------------------------------------------
@libentry()
@triton.jit
def _reduce_qr_kernel(
    RIN_ptr,
    ROUT_ptr,
    n_blocks,
    stride_ib,
    stride_ic,
    stride_ii,
    stride_ij,
    stride_ob,
    stride_oc,
    stride_oi,
    stride_oj,
    NC,
    G: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_NC: tl.constexpr,
    COMPUTE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    srow = tl.arange(0, BLOCK_S)
    col = tl.arange(0, BLOCK_NC)

    sub = srow // NC
    within = srow % NC
    src = pid_g * G + sub

    # Partial tail groups load as zero, leaving the QR factor unchanged.
    s_mask = (sub < G)[:, None] & (src < n_blocks)[:, None] & (col[None, :] < NC)
    s_off = (
        pid_b * stride_ib
        + src[:, None] * stride_ic
        + within[:, None] * stride_ii
        + col[None, :] * stride_ij
    )
    S = tl.load(RIN_ptr + s_off, mask=s_mask, other=0.0).to(COMPUTE)

    S = _householder_qr(S, NC, BLOCK_S, BLOCK_NC)

    o_off = (
        pid_b * stride_ob
        + pid_g * stride_oc
        + srow[:, None] * stride_oi
        + col[None, :] * stride_oj
    )
    o_mask = (srow[:, None] < NC) & (col[None, :] < NC)
    tl.store(
        ROUT_ptr + o_off, tl.where(srow[:, None] <= col[None, :], S, 0.0), mask=o_mask
    )


# ---------------------------------------------------------------------------
# Stage 3: back-substitution on the single surviving R. One program per batch.
# ---------------------------------------------------------------------------
@libentry()
@triton.jit
def _solve_kernel(
    R_ptr,
    X_ptr,
    RES_ptr,
    INFO_ptr,
    stride_rb,
    stride_ri,
    stride_rj,
    stride_xb,
    stride_xn,
    stride_xr,
    stride_eb,
    stride_er,
    RCOND,
    N,
    NC,
    BLOCK_NC: tl.constexpr,
    COMPUTE: tl.constexpr,
):
    pid_b = tl.program_id(0)

    rr = tl.arange(0, BLOCK_NC)
    col = tl.arange(0, BLOCK_NC)

    r_off = pid_b * stride_rb + rr[:, None] * stride_ri + col[None, :] * stride_rj
    r_mask = (rr[:, None] < NC) & (col[None, :] < NC)
    S = tl.load(R_ptr + r_off, mask=r_mask, other=0.0).to(COMPUTE)

    # rank test on diag(R_A): |r_ii| relative to max|r_ii| (scale-free). Only A
    # columns participate. An exactly-zero column already gave beta=0 above.
    diag = tl.sum(tl.where(rr[:, None] == col[None, :], S, 0.0), axis=0)
    diag_abs = tl.where(col < N, tl.abs(diag), 0.0)
    r_max = tl.max(diag_abs, axis=0)
    tol = RCOND * r_max

    deficient_vec = (diag_abs <= tol) & (col < N)
    first_bad = tl.min(tl.where(deficient_vec, col + 1, N + 1), axis=0)
    info = tl.where(first_bad > N, 0, first_bad)
    tl.store(INFO_ptr + pid_b, info.to(tl.int32))

    # back-substitution R_A X = C, all RHS at once, kept in S's column frame.
    X = tl.zeros((BLOCK_NC, BLOCK_NC), dtype=COMPUTE)
    for t in range(N):
        i = N - 1 - t
        r_i = tl.sum(tl.where(rr[:, None] == i, S, 0.0), axis=0)
        c_i = tl.where(col >= N, r_i, 0.0)
        r_ii = tl.sum(tl.where(col == i, r_i, 0.0), axis=0)
        coef = tl.where((col > i) & (col < N), r_i, 0.0)
        dot = tl.sum(coef[:, None] * X, axis=0)

        # NaN (not 0) on a negligible pivot: 0 is not the minimum-norm solution,
        # so it would look plausible while being wrong. NaN propagates loudly.
        deficient = tl.abs(r_ii) <= tol
        safe_rii = tl.where(deficient, 1.0, r_ii)
        x_i = (c_i - dot) / safe_rii
        x_i = tl.where(deficient, float("nan"), x_i)
        X = tl.where(rr[:, None] == i, x_i[None, :], X)

    rhs_idx = tl.maximum(col - N, 0)
    x_off = pid_b * stride_xb + rr[:, None] * stride_xn + rhs_idx[None, :] * stride_xr
    x_mask = (rr[:, None] < N) & (col[None, :] >= N) & (col[None, :] < NC)
    tl.store(X_ptr + x_off, X, mask=x_mask)

    # residual per RHS = ||R_D[:,k]||^2, i.e. column norms of the rows n.. block.
    below = (rr[:, None] >= N) & (rr[:, None] < NC)
    res = tl.sum(tl.where(below, S * S, 0.0), axis=0)
    e_off = pid_b * stride_eb + rhs_idx * stride_er
    e_mask = (col >= N) & (col < NC)
    tl.store(RES_ptr + e_off, res, mask=e_mask)


# ---------------------------------------------------------------------------
# host helpers
# ---------------------------------------------------------------------------
def _next_pow2(x):
    return 1 << (max(1, x) - 1).bit_length()


def _choose_block_m(m, block_nc):
    """Pick a power-of-two panel height.

    Small problems collapse to a single chunk (no reduction overhead); large
    ones are capped so the tile stays within _TARGET_TILE_BYTES. block_m only
    affects chunking/perf, never the result, as long as block_m >= NC.
    """
    budget = _prev_pow2(max(block_nc, _TARGET_TILE_BYTES // (block_nc * 4)))
    return max(block_nc, min(budget, _next_pow2(m)))


def _prev_pow2(x):
    p = _next_pow2(x)
    return p if p == x else p >> 1


def _lstsq_gels_tall(A, B, block_m=None, rcond=None):
    """A: (batch, m, n), m >= n.  B: (batch, m, nrhs).  fp32 or fp64, contiguous.

    Returns (X (batch, n, nrhs), residuals (batch, nrhs), info (batch,)).
    """
    batch, m, n = A.shape
    nrhs = B.shape[-1]
    dt = A.dtype
    compute = _tl_dtype(dt)
    NC = n + nrhs
    BLOCK_NC = _next_pow2(NC)
    if block_m is None:
        block_m = _choose_block_m(m, BLOCK_NC)
    assert block_m >= NC, f"block_m ({block_m}) must be >= n + nrhs ({NC})"

    n_chunks = triton.cdiv(m, block_m)
    G = max(2, _TARGET_STACK_ROWS // NC)
    BLOCK_S = _next_pow2(G * NC)

    A = A.contiguous()
    B = B.contiguous()

    buf_a = torch.zeros((batch, n_chunks, NC, NC), dtype=dt, device=A.device)
    buf_b = torch.zeros(
        (batch, max(1, triton.cdiv(n_chunks, G)), NC, NC), dtype=dt, device=A.device
    )

    if rcond is None:
        rcond = torch.finfo(dt).eps * max(m, n)

    with torch_device_fn.device(A.device):
        _panel_qr_kernel[(batch, n_chunks)](
            A,
            B,
            buf_a,
            m,
            A.stride(0),
            A.stride(1),
            A.stride(2),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            buf_a.stride(0),
            buf_a.stride(1),
            buf_a.stride(2),
            buf_a.stride(3),
            N=n,
            NC=NC,
            BLOCK_M=block_m,
            BLOCK_NC=BLOCK_NC,
            COMPUTE=compute,
        )

        cur, nxt, live = buf_a, buf_b, n_chunks
        while live > 1:
            n_groups = triton.cdiv(live, G)
            _reduce_qr_kernel[(batch, n_groups)](
                cur,
                nxt,
                live,
                cur.stride(0),
                cur.stride(1),
                cur.stride(2),
                cur.stride(3),
                nxt.stride(0),
                nxt.stride(1),
                nxt.stride(2),
                nxt.stride(3),
                NC=NC,
                G=G,
                BLOCK_S=BLOCK_S,
                BLOCK_NC=BLOCK_NC,
                COMPUTE=compute,
            )
            cur, nxt, live = nxt, cur, n_groups

        X = torch.empty((batch, n, nrhs), dtype=dt, device=A.device)
        RES = torch.empty((batch, nrhs), dtype=dt, device=A.device)
        INFO = torch.empty((batch,), dtype=torch.int32, device=A.device)

        _solve_kernel[(batch,)](
            cur,
            X,
            RES,
            INFO,
            cur.stride(0),
            cur.stride(2),
            cur.stride(3),  # cur[:, 0] view
            X.stride(0),
            X.stride(1),
            X.stride(2),
            RES.stride(0),
            RES.stride(1),
            rcond,
            N=n,
            NC=NC,
            BLOCK_NC=BLOCK_NC,
            COMPUTE=compute,
        )
    return X, RES, INFO


# ---------------------------------------------------------------------------
# Blocked (no-ceiling) tall path. Right-looking PANEL-blocked Householder QR
# streams column panels through global memory, so NC is only a loop bound (no
# BLOCK_NC-wide register tile -> no spill at any NC). One generic kernel factors
# any (ROWS x NC) tile; it serves both the per-chunk panel factor and the TSQR
# reduce (stacked R-factors). Used when n+nrhs exceeds the monolithic ceiling;
# correct for all NC (validated bit-identical to monolithic). Never falls back.
# ---------------------------------------------------------------------------
def _blk_panel(block_rows, dt):
    """Panel width keeping ~3 working tiles within an SRAM budget (power of 2)."""
    bpe = 8 if dt == torch.float64 else 4
    p = (150 * 1024) // max(1, 3 * block_rows * bpe)
    return max(1, _prev_pow2(max(1, min(16, p))))


@libentry()
@triton.jit
def _blk_qr_kernel(
    W_ptr,
    ROWS,
    NC,
    swt,
    swi,
    swj,
    BLOCK_ROWS: tl.constexpr,
    PANEL: tl.constexpr,
    COMPUTE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = tl.arange(0, BLOCK_ROWS)
    pj = tl.arange(0, PANEL)
    wb = pid * swt
    for p in range(0, NC, PANEL):
        pcols = p + pj
        po = wb + row[:, None] * swi + pcols[None, :] * swj
        pm = (row[:, None] < ROWS) & (pcols[None, :] < NC)
        Pblk = tl.load(W_ptr + po, mask=pm, other=0.0).to(COMPUTE)
        Vp = tl.zeros((BLOCK_ROWS, PANEL), dtype=COMPUTE)
        betas = tl.zeros((PANEL,), dtype=COMPUTE)
        for jj in range(PANEL):
            gj = p + jj  # global col == pivot row
            x = tl.sum(tl.where(pj[None, :] == jj, Pblk, 0.0), axis=1)
            x = tl.where(row >= gj, x, 0.0)
            active = gj < NC
            nrm = tl.sqrt(tl.sum(x * x, axis=0))
            xg = tl.sum(tl.where(row == gj, x, 0.0), axis=0)
            sign = tl.where(xg >= 0.0, 1.0, -1.0)
            alpha = -sign * nrm
            v = tl.where(row == gj, x - alpha, x)
            v = tl.where(row >= gj, v, 0.0)
            vtv = tl.sum(v * v, axis=0)
            beta = tl.where(
                (vtv > 0.0) & active, 2.0 / tl.where(vtv > 0.0, vtv, 1.0), 0.0
            )
            Vp = tl.where(pj[None, :] == jj, v[:, None], Vp)
            betas = tl.where(pj == jj, beta, betas)
            w = tl.sum(v[:, None] * Pblk, axis=0)
            w = tl.where(pj >= jj, w, 0.0)
            Pblk = Pblk - beta * v[:, None] * w[None, :]
        # panel columns are FINAL here (trailing updates only touch cols > p +
        # PANEL, and reflectors live in registers, never reloaded from W) — so
        # zero the sub-diagonal reflector junk at store time. W ends up exactly
        # [R; 0] and the host needs no triu (torch-free, per the no-torch rule).
        Pblk = tl.where(row[:, None] > pcols[None, :], 0.0, Pblk)
        tl.store(W_ptr + po, Pblk, mask=pm)
        for t in range(p + PANEL, NC, PANEL):
            tcols = t + pj
            to = wb + row[:, None] * swi + tcols[None, :] * swj
            tm = (row[:, None] < ROWS) & (tcols[None, :] < NC)
            Tblk = tl.load(W_ptr + to, mask=tm, other=0.0).to(COMPUTE)
            for jj in range(PANEL):
                v = tl.sum(tl.where(pj[None, :] == jj, Vp, 0.0), axis=1)
                bj = tl.sum(tl.where(pj == jj, betas, 0.0), axis=0)
                w = tl.sum(v[:, None] * Tblk, axis=0)
                Tblk = Tblk - bj * v[:, None] * w[None, :]
            tl.store(W_ptr + to, Tblk, mask=tm)


@libentry()
@triton.jit
def _blk_solve_kernel(
    R_ptr,
    X_ptr,
    RES_ptr,
    N,
    NC,
    nrhs,
    srb,
    sri,
    srj,
    sxb,
    sxn,
    sxr,
    seb,
    ser,
    RCOND,
    BLOCK_NC: tl.constexpr,
    COMPUTE: tl.constexpr,
):
    pid = tl.program_id(0)
    cc = tl.arange(0, BLOCK_NC)
    rb = pid * srb
    di = rb + cc * sri + cc * srj  # diagonal as a strided vector
    diag = tl.load(R_ptr + di, mask=cc < NC, other=0.0).to(COMPUTE)
    diag_abs = tl.where(cc < N, tl.abs(diag), 0.0)
    tol = RCOND * tl.max(diag_abs, axis=0)

    # residual per rhs = sum_{i in [N,NC)} R[i, N+k]^2
    resv = tl.zeros((BLOCK_NC,), dtype=COMPUTE)
    for i in range(N, NC):
        ro = rb + i * sri + cc * srj
        r_i = tl.load(R_ptr + ro, mask=cc < NC, other=0.0).to(COMPUTE)
        resv += tl.where((cc >= N) & (cc < NC), r_i * r_i, 0.0)
    e_off = pid * seb + tl.maximum(cc - N, 0) * ser
    tl.store(RES_ptr + e_off, resv, mask=(cc >= N) & (cc < NC))

    for k in range(nrhs):
        x = tl.zeros((BLOCK_NC,), dtype=COMPUTE)
        for t in range(N):
            i = N - 1 - t
            ro = rb + i * sri + cc * srj
            r_i = tl.load(R_ptr + ro, mask=cc < NC, other=0.0).to(COMPUTE)
            r_ii = tl.sum(tl.where(cc == i, r_i, 0.0), axis=0)
            c_i = tl.sum(tl.where(cc == (N + k), r_i, 0.0), axis=0)
            dot = tl.sum(tl.where((cc > i) & (cc < N), r_i * x, 0.0), axis=0)
            defi = tl.abs(r_ii) <= tol
            xi = (c_i - dot) / tl.where(defi, 1.0, r_ii)
            xi = tl.where(defi, float("nan"), xi)
            x = tl.where(cc == i, xi, x)
        x_off = pid * sxb + cc * sxn + k * sxr
        tl.store(X_ptr + x_off, x, mask=cc < N)


def _tsqr_R(aug, NC, dt, compute):
    """Blocked TSQR of aug (batch, M, NC), M >= NC -> R (batch, NC, NC).
    Must run inside a torch_device_fn.device() context. Shared by the tall
    (aug = [A|B]) and wide (aug = A^T) no-ceiling paths.

    WARNING: consumes `aug` as scratch. When M is already a multiple of the
    chunk size (no padding), the reshape below ALIASES aug's storage and the
    in-place QR kernel destroys its contents. Callers must not reuse aug."""
    batch, M, _ = aug.shape
    dev = aug.device
    block_m = max(256, _next_pow2(NC))  # >= NC (need NC pivots)
    n_chunks = triton.cdiv(M, block_m)
    pad = n_chunks * block_m - M
    if pad:
        aug = torch.cat([aug, torch.zeros(batch, pad, NC, dtype=dt, device=dev)], dim=1)
    W = aug.reshape(batch, n_chunks, block_m, NC).reshape(-1, block_m, NC).contiguous()
    _blk_qr_kernel[(W.shape[0],)](
        W,
        block_m,
        NC,
        W.stride(0),
        W.stride(1),
        W.stride(2),
        BLOCK_ROWS=block_m,
        PANEL=_blk_panel(block_m, dt),
        COMPUTE=compute,
    )
    # _blk_qr_kernel stores exact [R; 0] (sub-diagonal zeroed on store), so the
    # top NC rows ARE upper-triangular R — plain slice-copy, no torch.triu.
    R = W[:, :NC, :NC].contiguous().reshape(batch, n_chunks, NC, NC)

    live = n_chunks
    G = max(2, _TARGET_STACK_ROWS // NC)
    while live > 1:
        ng = triton.cdiv(live, G)
        padb = ng * G - live
        if padb:
            R = torch.cat(
                [R, torch.zeros(batch, padb, NC, NC, dtype=dt, device=dev)], dim=1
            )
        BR = _next_pow2(G * NC)
        S = R.reshape(batch, ng, G * NC, NC)
        if BR > G * NC:
            S = torch.cat(
                [S, torch.zeros(batch, ng, BR - G * NC, NC, dtype=dt, device=dev)],
                dim=2,
            )
        Sf = S.reshape(-1, BR, NC).contiguous()
        _blk_qr_kernel[(Sf.shape[0],)](
            Sf,
            G * NC,
            NC,
            Sf.stride(0),
            Sf.stride(1),
            Sf.stride(2),
            BLOCK_ROWS=BR,
            PANEL=_blk_panel(BR, dt),
            COMPUTE=compute,
        )
        R = Sf[:, :NC, :NC].contiguous().reshape(batch, ng, NC, NC)
        live = ng
    return R.reshape(batch, NC, NC).contiguous()


def _lstsq_gels_tall_blocked(A, B, rcond=None):
    """No-ceiling tall solve via blocked TSQR (any NC). Same (X, RES, INFO)
    contract as _lstsq_gels_tall; INFO is unused by the caller (zeros)."""
    batch, m, n = A.shape
    nrhs = B.shape[-1]
    NC = n + nrhs
    dt = A.dtype
    compute = _tl_dtype(dt)
    dev = A.device
    if rcond is None:
        rcond = torch.finfo(dt).eps * max(m, n)
    aug = torch.cat([A.contiguous(), B.contiguous()], dim=-1)
    with torch_device_fn.device(dev):
        Rf = _tsqr_R(aug, NC, dt, compute)
        X = torch.empty((batch, n, nrhs), dtype=dt, device=dev)
        RES = torch.empty((batch, nrhs), dtype=dt, device=dev)
        _blk_solve_kernel[(batch,)](
            Rf,
            X,
            RES,
            n,
            NC,
            nrhs,
            Rf.stride(0),
            Rf.stride(1),
            Rf.stride(2),
            X.stride(0),
            X.stride(1),
            X.stride(2),
            RES.stride(0),
            RES.stride(1),
            rcond,
            BLOCK_NC=_next_pow2(NC),
            COMPUTE=compute,
        )
    INFO = torch.zeros((batch,), dtype=torch.int32, device=dev)
    return X, RES, INFO


@libentry()
@triton.jit
def _wide_solve_kernel(
    R_ptr,
    B_ptr,
    W_ptr,
    M,
    nrhs,
    srb,
    sri,
    srj,
    sbb,
    sbm,
    sbr,
    swb,
    swm,
    swr,
    RCOND,
    BLOCK_M: tl.constexpr,
    COMPUTE: tl.constexpr,
):
    # w = R^-1 R^-T b  (min-norm coeffs; x = A^T w done by the host GEMV).
    pid = tl.program_id(0)
    cc = tl.arange(0, BLOCK_M)
    rb = pid * srb
    di = rb + cc * sri + cc * srj
    diag = tl.load(R_ptr + di, mask=cc < M, other=0.0).to(COMPUTE)
    tol = RCOND * tl.max(tl.where(cc < M, tl.abs(diag), 0.0), axis=0)
    for k in range(nrhs):
        boff = pid * sbb + cc * sbm + k * sbr
        bk = tl.load(B_ptr + boff, mask=cc < M, other=0.0).to(COMPUTE)
        # forward: R^T z = b  (uses column i of R)
        z = tl.zeros((BLOCK_M,), dtype=COMPUTE)
        for i in range(M):
            rci = tl.load(R_ptr + rb + cc * sri + i * srj, mask=cc < M, other=0.0).to(
                COMPUTE
            )
            r_ii = tl.sum(tl.where(cc == i, rci, 0.0), axis=0)
            b_i = tl.sum(tl.where(cc == i, bk, 0.0), axis=0)
            s = tl.sum(tl.where(cc < i, rci * z, 0.0), axis=0)
            defi = tl.abs(r_ii) <= tol
            zi = tl.where(defi, float("nan"), (b_i - s) / tl.where(defi, 1.0, r_ii))
            z = tl.where(cc == i, zi, z)
        # back: R w = z  (uses row i of R)
        w = tl.zeros((BLOCK_M,), dtype=COMPUTE)
        for t in range(M):
            i = M - 1 - t
            rri = tl.load(R_ptr + rb + i * sri + cc * srj, mask=cc < M, other=0.0).to(
                COMPUTE
            )
            r_ii = tl.sum(tl.where(cc == i, rri, 0.0), axis=0)
            z_i = tl.sum(tl.where(cc == i, z, 0.0), axis=0)
            s = tl.sum(tl.where(cc > i, rri * w, 0.0), axis=0)
            defi = tl.abs(r_ii) <= tol
            wi = tl.where(defi, float("nan"), (z_i - s) / tl.where(defi, 1.0, r_ii))
            w = tl.where(cc == i, wi, w)
        tl.store(W_ptr + pid * swb + cc * swm + k * swr, w, mask=cc < M)


@libentry()
@triton.jit
def _atw_gemv_kernel(
    A_ptr,
    W_ptr,
    X_ptr,
    M,
    N,
    nrhs,
    sab,
    sam,
    san,
    swb,
    swm,
    swr,
    sxb,
    sxn,
    sxr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    COMPUTE: tl.constexpr,
):
    # X[b] = A[b]^T @ W[b]:  X[i, j] = sum_k A[k, i] * W[k, j].
    # Reads A through its strides directly — no A^T materialization.
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # i in [0, N)
    kk = tl.arange(0, BLOCK_K)
    ab = pid_b * sab
    for j in range(nrhs):
        acc = tl.zeros((BLOCK_N,), dtype=COMPUTE)
        for k0 in range(0, M, BLOCK_K):
            ks = k0 + kk
            a = tl.load(
                A_ptr + ab + ks[:, None] * sam + cols[None, :] * san,
                mask=(ks[:, None] < M) & (cols[None, :] < N),
                other=0.0,
            ).to(COMPUTE)
            wv = tl.load(
                W_ptr + pid_b * swb + ks * swm + j * swr, mask=ks < M, other=0.0
            ).to(COMPUTE)
            acc += tl.sum(a * wv[:, None], axis=0)
        tl.store(X_ptr + pid_b * sxb + cols * sxn + j * sxr, acc, mask=cols < N)


def _lstsq_gels_wide_blocked(A, B, rcond=None):
    """No-ceiling underdetermined (m < n) min-norm solve, any size.
    x = A^T R^-1 R^-T b with A^T = QR (blocked TSQR, no spill; no Q needed)."""
    batch, m, n = A.shape
    nrhs = B.shape[-1]
    dt = A.dtype
    compute = _tl_dtype(dt)
    dev = A.device
    if rcond is None:
        rcond = torch.finfo(dt).eps * max(m, n)
    with torch_device_fn.device(dev):
        # _tsqr_R consumes its input as in-place scratch, so hand it a clone of
        # the transpose (clone, not .contiguous(): for m == 1 the transpose
        # view is already "contiguous" and would alias the caller's A).
        scratch = A.transpose(-1, -2).clone(memory_format=torch.contiguous_format)
        R = _tsqr_R(scratch, m, dt, compute)  # (batch, m, m)
        Bf = B.contiguous()
        w = torch.empty((batch, m, nrhs), dtype=dt, device=dev)
        _wide_solve_kernel[(batch,)](
            R,
            Bf,
            w,
            m,
            nrhs,
            R.stride(0),
            R.stride(1),
            R.stride(2),
            Bf.stride(0),
            Bf.stride(1),
            Bf.stride(2),
            w.stride(0),
            w.stride(1),
            w.stride(2),
            rcond,
            BLOCK_M=_next_pow2(m),
            COMPUTE=compute,
        )
        # final GEMV x = A^T w with our own kernel (no torch.bmm — the op must
        # not compute through torch, per the no-torch review rule). The kernel
        # indexes A itself via strides, so no transpose is materialized.
        X = torch.empty((batch, n, nrhs), dtype=dt, device=dev)
        BLOCK_N = 128  # output-column tile (n padded to this in the grid)
        BLOCK_K = 32  # K-accumulation chunk; memory-bound GEMV, not tuning-sensitive
        _atw_gemv_kernel[(batch, triton.cdiv(n, BLOCK_N))](
            A,
            w,
            X,
            m,
            n,
            nrhs,
            A.stride(0),
            A.stride(1),
            A.stride(2),
            w.stride(0),
            w.stride(1),
            w.stride(2),
            X.stride(0),
            X.stride(1),
            X.stride(2),
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            COMPUTE=compute,
        )
    INFO = torch.zeros((batch,), dtype=torch.int32, device=dev)
    return X, INFO


def _fallback(A, b, rcond, driver):
    """Reference path for cases outside the fast-path scope. Routed through CPU
    so it does not re-enter our overridden device kernel."""
    logger.debug("GEMS LINALG_LSTSQ")
    res = torch.linalg.lstsq(A.cpu(), b.cpu(), rcond=rcond, driver=driver)
    dev = A.device
    return (
        res.solution.to(dev),
        res.residuals.to(dev),
        res.rank.to(dev),
        res.singular_values.to(dev),
    )


# ---------------------------------------------------------------------------
# Underdetermined (m < n): minimum-norm solve via QR of A^T (single tile).
#   A^T = Q R  ->  solve R^T z = b (forward subst)  ->  x = Q z  (min norm).
# Unlike the tall path this must APPLY Q (forward), so we store the Householder
# reflectors and apply them in reverse. One program per batch element; bounded
# by the tile (W + reflectors), else the caller uses the blocked wide path.
# ---------------------------------------------------------------------------
# The single-tile wide kernel holds A^T padded to next_pow2(n) x next_pow2(m)
# (plus a same-size reflector store). Routing between it and the blocked wide
# path is by tile AREA — the boundary is the product, not either dim.
# fp32 (measured on H20): area<=32768 beats torch-GPU by 1.06-40x; area=65536
# (e.g. 512x128 or 1024x64) spills to >= torch.
# fp64 (measured on H20, probe_f64.py, 2026-07-23): single-tile wins only up
# to area 8192 (0.83x of blocked); at 16384 it is already 3-6x SLOWER than
# blocked — exactly one power-of-two below half the fp32 budget, because the
# reflector store doubles the 8-byte footprint again.
_UNDERDET_TILE_F32 = 32768
_UNDERDET_TILE_F64 = 8192


@libentry()
@triton.jit
def _underdet_kernel(
    A_ptr,
    B_ptr,
    X_ptr,
    INFO_ptr,
    stride_ab,
    stride_am,
    stride_an,
    stride_bb,
    stride_bm,
    stride_br,
    stride_xb,
    stride_xn,
    stride_xr,
    RCOND,
    M,
    N,
    NRHS,
    BLOCK_R: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_NRHS: tl.constexpr,
    COMPUTE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = tl.arange(0, BLOCK_R)  # n-dim (rows of A^T)
    col = tl.arange(0, BLOCK_M)  # m-dim (cols of A^T)
    kk = tl.arange(0, BLOCK_NRHS)

    # W = A^T : W[i,j] = A[j,i]
    a_off = pid * stride_ab + col[None, :] * stride_am + row[:, None] * stride_an
    a_mask = (row[:, None] < N) & (col[None, :] < M)
    W = tl.load(A_ptr + a_off, mask=a_mask, other=0.0).to(COMPUTE)

    Vh = tl.zeros((BLOCK_R, BLOCK_M), dtype=COMPUTE)  # reflector j in col j
    tau = tl.zeros((BLOCK_M,), dtype=COMPUTE)

    # --- Householder QR of W, storing reflectors ---------------------------
    for j in range(M):
        x = tl.sum(tl.where(col[None, :] == j, W, 0.0), axis=1)
        x = tl.where(row >= j, x, 0.0)
        norm = tl.sqrt(tl.sum(x * x, axis=0))
        xj = tl.sum(tl.where(row == j, x, 0.0), axis=0)
        sign = tl.where(xj >= 0.0, 1.0, -1.0)
        alpha = -sign * norm
        v = tl.where(row == j, x - alpha, x)
        v = tl.where(row >= j, v, 0.0)
        vtv = tl.sum(v * v, axis=0)
        beta = tl.where(vtv > 0.0, 2.0 / tl.where(vtv > 0.0, vtv, 1.0), 0.0)
        Vh = tl.where(col[None, :] == j, v[:, None], Vh)
        tau = tl.where(col == j, beta, tau)
        w = tl.sum(v[:, None] * W, axis=0)
        w = tl.where(col >= j, w, 0.0)
        W = W - beta * v[:, None] * w[None, :]

    # --- rank guard on diag(R): diag[j] = W[j,j] ---------------------------
    diag = tl.sum(tl.where(row[:, None] == col[None, :], W, 0.0), axis=0)
    diag_abs = tl.where(col < M, tl.abs(diag), 0.0)
    r_max = tl.max(diag_abs, axis=0)
    tol = RCOND * r_max
    deficient = (diag_abs <= tol) & (col < M)
    first_bad = tl.min(tl.where(deficient, col + 1, M + 1), axis=0)
    info = tl.where(first_bad > M, 0, first_bad)
    tl.store(INFO_ptr + pid, info.to(tl.int32))

    # b into rows 0..M-1
    b_off = pid * stride_bb + row[:, None] * stride_bm + kk[None, :] * stride_br
    b_mask = (row[:, None] < M) & (kk[None, :] < NRHS)
    Bblk = tl.load(B_ptr + b_off, mask=b_mask, other=0.0).to(COMPUTE)

    # --- solve R^T z = b (forward substitution), z in rows 0..M-1 ----------
    z = tl.zeros((BLOCK_R, BLOCK_NRHS), dtype=COMPUTE)
    for i in range(M):
        Rcol = tl.sum(tl.where(col[None, :] == i, W, 0.0), axis=1)  # R[:,i]
        r_ii = tl.sum(tl.where(row == i, Rcol, 0.0), axis=0)
        coef = tl.where(row < i, Rcol, 0.0)
        s = tl.sum(coef[:, None] * z, axis=0)  # (nrhs,)
        b_i = tl.sum(tl.where(row[:, None] == i, Bblk, 0.0), axis=0)
        deficient_i = tl.abs(r_ii) <= tol
        safe = tl.where(deficient_i, 1.0, r_ii)
        z_i = (b_i - s) / safe
        z_i = tl.where(deficient_i, float("nan"), z_i)
        z = tl.where(row[:, None] == i, z_i[None, :], z)

    # --- x = Q z : apply H_{M-1} .. H_0 to [z; 0] --------------------------
    y = z  # rows>=M already zero
    for t in range(M):
        j = M - 1 - t
        vj = tl.sum(tl.where(col[None, :] == j, Vh, 0.0), axis=1)
        tau_j = tl.sum(tl.where(col == j, tau, 0.0), axis=0)
        d = tl.sum(vj[:, None] * y, axis=0)  # (nrhs,)
        y = y - tau_j * vj[:, None] * d[None, :]

    x_off = pid * stride_xb + row[:, None] * stride_xn + kk[None, :] * stride_xr
    x_mask = (row[:, None] < N) & (kk[None, :] < NRHS)
    tl.store(X_ptr + x_off, y, mask=x_mask)


def _lstsq_gels_wide(A, B, rcond=None):
    """A: (batch, m, n) with m < n.  B: (batch, m, nrhs).  fp32 or fp64.

    Returns (X (batch, n, nrhs), info (batch,)) — the minimum-norm solution.
    """
    batch, m, n = A.shape
    nrhs = B.shape[-1]
    dt = A.dtype
    A = A.contiguous()
    B = B.contiguous()
    X = torch.empty((batch, n, nrhs), dtype=dt, device=A.device)
    INFO = torch.empty((batch,), dtype=torch.int32, device=A.device)
    if rcond is None:
        rcond = torch.finfo(dt).eps * max(m, n)

    with torch_device_fn.device(A.device):
        _underdet_kernel[(batch,)](
            A,
            B,
            X,
            INFO,
            A.stride(0),
            A.stride(1),
            A.stride(2),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            X.stride(0),
            X.stride(1),
            X.stride(2),
            rcond,
            M=m,
            N=n,
            NRHS=nrhs,
            BLOCK_R=_next_pow2(n),
            BLOCK_M=_next_pow2(m),
            BLOCK_NRHS=_next_pow2(nrhs),
            COMPUTE=_tl_dtype(dt),
        )
    return X, INFO


def _empty_rank_sv(A):
    return (
        torch.empty(0, dtype=torch.int64, device=A.device),
        torch.empty(0, dtype=A.dtype, device=A.device),
    )


def linalg_lstsq(A, b, rcond=None, driver=None):
    logger.debug("GEMS LINALG_LSTSQ")

    # non-gels driver: torch's CUDA gels backend rejects it -> raise likewise
    # (do NOT silently CPU-fall-back and compute a result torch would refuse).
    if driver not in (None, "gels"):
        raise RuntimeError(
            "torch.linalg.lstsq: `driver` other than `gels` is not " "supported on CUDA"
        )

    # unsupported dtype / shape -> reference fallback
    if A.dtype not in _SUPPORTED_DTYPES or A.is_complex() or A.dim() < 2 or b.dim() < 1:
        return _fallback(A, b, rcond, driver)

    m, n = A.shape[-2], A.shape[-1]

    # RHS classification, matching torch.linalg.lstsq exactly:
    #   - VECTOR rhs: b has one fewer dim than A AND b.shape == A.shape[:-1]
    #     exactly. batch = A's, no broadcast.
    #   - MATRIX rhs (dim_diff == 0): b is (*, m, nrhs), same ndim as A; batch
    #     dims broadcast against A's.
    #   - anything else (incl. dim_diff == 1 that is NOT an exact vector rhs, or
    #     dim_diff not in {0,1}) torch rejects -> fall back so we raise likewise.
    dim_diff = A.dim() - b.dim()
    if dim_diff == 1 and tuple(b.shape) == tuple(A.shape[:-1]):
        vector_rhs, b2 = True, b.unsqueeze(-1)
    elif dim_diff == 0:
        vector_rhs, b2 = False, b
    else:
        return _fallback(A, b, rcond, driver)
    if b2.shape[-2] != m:  # b's row count must be m
        return _fallback(A, b, rcond, driver)
    nrhs = b2.shape[-1]

    try:
        batch_shape = torch.broadcast_shapes(A.shape[:-2], b2.shape[:-2])
    except RuntimeError:  # non-broadcastable batches
        return _fallback(A, b, rcond, driver)
    A_bc = A.expand(*batch_shape, m, n)
    B_bc = b2.expand(*batch_shape, m, nrhs)

    # degenerate dims (m/n/nrhs == 0): shape-determined, no kernel work — handled
    # NATIVELY. Exact torch semantics (BatchLinearAlgebra.cpp,
    # linalg_lstsq_out_info):
    #   - solution is (*, n, nrhs) (squeezed for vector b); explicitly
    #     zero-FILLED when m == 0, 0-numel anyway when n or nrhs is 0.
    #   - residuals: torch post-processes iff m > n (gels), summing squares of
    #     the working buffer's rows n:m — but LAPACK ?gels QUICK-RETURNS when
    #     min(m, n) == 0 or nrhs == 0 and zeroes that whole buffer (dlaset), so
    #     the result is ZEROS of shape (*, nrhs) (verified on H20: NOT sum b^2).
    #     When m <= n it stays the initial empty(0).
    #   - rank / singular_values: always empty for the gels driver.
    if m == 0 or n == 0 or nrhs == 0:
        solution = torch.zeros((*batch_shape, n, nrhs), dtype=A.dtype, device=A.device)
        if vector_rhs:
            solution = solution.squeeze(-1)
        if m > n:  # here that means n or nrhs == 0
            residuals = torch.zeros(
                (*batch_shape, nrhs), dtype=A.dtype, device=A.device
            )
        else:
            residuals = torch.empty(0, dtype=A.dtype, device=A.device)
        rank, singular_values = _empty_rank_sv(A)
        return solution, residuals, rank, singular_values

    if m < n:
        # underdetermined min-norm. Within the measured tile area budget the fast
        # single-tile kernel (with Q-apply) wins; beyond it, the blocked TSQR path
        # (QR of A^T -> R, then x = A^T R^-1 R^-T b, no Q). Native either way.
        tile = _next_pow2(n) * _next_pow2(m)
        budget = _UNDERDET_TILE_F32 if A.dtype == torch.float32 else _UNDERDET_TILE_F64
        Af = A_bc.reshape(-1, m, n)
        Bf = B_bc.reshape(-1, m, nrhs)
        if tile <= budget:
            X, _INFO = _lstsq_gels_wide(Af, Bf, rcond=rcond)
        else:
            X, _INFO = _lstsq_gels_wide_blocked(Af, Bf, rcond=rcond)
        solution = X.reshape(*batch_shape, n, nrhs)
        if vector_rhs:
            solution = solution.squeeze(-1)
        residuals = torch.empty(0, dtype=A.dtype, device=A.device)  # m < n
        rank, singular_values = _empty_rank_sv(A)
        return solution, residuals, rank, singular_values

    # overdetermined or square (m >= n): monolithic TSQR for small NC (faster),
    # blocked TSQR for large NC (no register spill). Native either way — the
    # tall path never falls back.
    max_nc = _TALL_MAX_NC_F32 if A.dtype == torch.float32 else _TALL_MAX_NC_F64
    Af = A_bc.reshape(-1, m, n)
    Bf = B_bc.reshape(-1, m, nrhs)
    if n + nrhs <= max_nc:
        X, RES, _INFO = _lstsq_gels_tall(Af, Bf, rcond=rcond)
    else:
        X, RES, _INFO = _lstsq_gels_tall_blocked(Af, Bf, rcond=rcond)

    solution = X.reshape(*batch_shape, n, nrhs)
    if vector_rhs:
        solution = solution.squeeze(-1)

    # torch returns residuals only when m > n (gels), else an empty tensor.
    # Note: torch squeezes the SOLUTION for a vector b, but keeps residuals at
    # shape (*, nrhs) — so a vector b gives residuals of shape (*, 1), NOT a
    # scalar. Do not squeeze here.
    if m > n:
        residuals = RES.reshape(*batch_shape, nrhs)
    else:
        residuals = torch.empty(0, dtype=A.dtype, device=A.device)

    rank, singular_values = _empty_rank_sv(A)
    return solution, residuals, rank, singular_values

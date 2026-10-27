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
"""Ascend override for linalg_lstsq.

The shared (NVIDIA) kernels cannot run on this backend at all, for reasons that
were measured rather than assumed:

  1. no float64 on the device (torch_npu silently downcasts it to float32)
  2. `tl.where` inside a loop lowers to `scf.if`, which defeats BiShengHIR's
     root-alloc analysis: it then cannot bound the loop-carried buffer and
     reports a nonsense UB requirement. Every in-loop select here is therefore
     an arithmetic mask, `a*(1-m) + b*m`.
  3. a masked `tl.store` whose lanes compute DUPLICATE addresses is silently
     DROPPED - nothing is written, no error. Every store here is injective;
     where a packed output was wanted, the full tile is stored and the host
     slices it. (Masked stores are fine, as long as addresses are unique.)
  4. usable UB is ~36KB of live tiles, not the nominal 192KB, because
     bishengir runs with --enable-auto-multi-buffer=True.

Algorithm (same as the shared op): Householder TSQR on [A | B], so R and Q^T B
fall out of one pass and Q is never formed. Rows are chunked and reduced by a
fan-in-2 tree, so m is unbounded. Columns are processed in panels streamed
through global memory, so NC is a loop bound rather than a tile dimension. The
back-substitution and residuals run on vectors, never an NC x NC tile.

Underdetermined (m < n) uses the minimum-norm form x = A^T R^-1 R^-T b with
A^T = QR, again without forming Q.

ENVELOPE (measured on Ascend 910B; outside it this RAISES rather than falling
back to CPU, because a fallback would not be a native implementation):
  * float32 only (the device has no fp64 at all)
  * no size limit: m, n and nrhs are all unbounded

n + nrhs is no longer capped: beyond BNC=256 the tall path switches to a
compact-WY panel QR whose row and column extents are loop bounds rather than
tile dimensions, so square and square-ish shapes work too.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_BUDGET_ELEMS = 36 * 1024 // 4


def _next_pow2(x):
    return 1 << (max(1, int(x)) - 1).bit_length()


def _prev_pow2(x):
    p = _next_pow2(x)
    return p if p == x else p >> 1


def _panel_width(rows):
    """PANEL keeping ~3 live tiles of rows x PANEL inside the budget."""
    return max(1, _prev_pow2(min(16, max(1, _BUDGET_ELEMS // (3 * rows)))))


_MAX_BNC = 256  # QR fan-in ceiling
# back-substitution frame ceiling, measured: the N-framed _vec_solve fits
# at 512 (probe_vecsolve_frame.py), unlike the old NC-framed one at 208KB
_SOLVE_MAX_FRAME = 512


@triton.jit
def _panel_qr(
    W_ptr,
    ROWS,
    NC,
    swt,
    swi,
    swj,
    BLOCK_ROWS: tl.constexpr,
    PANEL: tl.constexpr,
):
    """Right-looking blocked Householder QR, in place on W.

    NC appears only as a loop bound, never as a tile dimension — that is what
    removes the NC ceiling. Stores are masked but injective (row, col unique).
    """
    pid = tl.program_id(0)
    row = tl.arange(0, BLOCK_ROWS)
    pj = tl.arange(0, PANEL)
    wb = pid * swt

    for p in range(0, NC, PANEL):
        pcols = p + pj
        po = wb + row[:, None] * swi + pcols[None, :] * swj
        pm = (row[:, None] < ROWS) & (pcols[None, :] < NC)
        Pblk = tl.load(W_ptr + po, mask=pm, other=0.0).to(tl.float32)

        Vp = tl.zeros((BLOCK_ROWS, PANEL), dtype=tl.float32)
        betas = tl.zeros((PANEL,), dtype=tl.float32)

        for jj in range(PANEL):
            gj = p + jj
            mjj = (pj == jj).to(tl.float32)
            mge = (row >= gj).to(tl.float32)
            meq = (row == gj).to(tl.float32)

            x = tl.sum(mjj[None, :] * Pblk, axis=1) * mge
            nrm = tl.sqrt(tl.sum(x * x, axis=0))
            xg = tl.sum(meq * x, axis=0)
            sign = 1.0 - 2.0 * (xg < 0.0).to(tl.float32)
            alpha = -sign * nrm
            v = (x - meq * alpha) * mge
            vtv = tl.sum(v * v, axis=0)
            # inactive column (gj >= NC) or zero column -> beta 0, no Inf formed
            pos = (vtv > 0.0).to(tl.float32) * (gj < NC).to(tl.float32)
            beta = 2.0 / (vtv + (1.0 - pos)) * pos

            Vp = Vp * (1.0 - mjj[None, :]) + v[:, None] * mjj[None, :]
            betas = betas * (1.0 - mjj) + beta * mjj

            w = tl.sum(v[:, None] * Pblk, axis=0) * (pj >= jj).to(tl.float32)
            Pblk = Pblk - beta * v[:, None] * w[None, :]

        tl.store(W_ptr + po, Pblk, mask=pm)

        # apply this panel's reflectors to every trailing panel
        for t in range(p + PANEL, NC, PANEL):
            tcols = t + pj
            to = wb + row[:, None] * swi + tcols[None, :] * swj
            tm = (row[:, None] < ROWS) & (tcols[None, :] < NC)
            Tblk = tl.load(W_ptr + to, mask=tm, other=0.0).to(tl.float32)
            for jj in range(PANEL):
                mjj = (pj == jj).to(tl.float32)
                v = tl.sum(mjj[None, :] * Vp, axis=1)
                bj = tl.sum(mjj * betas, axis=0)
                w = tl.sum(v[:, None] * Tblk, axis=0)
                Tblk = Tblk - bj * v[:, None] * w[None, :]
            tl.store(W_ptr + to, Tblk, mask=tm)

    # Final pass: zero the sub-diagonal so the stored tile is exactly R.
    # This cannot be folded into the panel stores above — when a panel is
    # written, the trailing columns still hold live data for later panels.
    # Doing it here removes the host-side `torch.triu`, which matters twice
    # over: it is compute (banned in an op by the no-torch rule) and, inside
    # `use_gems()`, it re-dispatches to the FlagGems triu kernel, which HANGS
    # on Ascend.
    for q in range(0, NC, PANEL):
        qcols = q + pj
        qo = wb + row[:, None] * swi + qcols[None, :] * swj
        qm = (row[:, None] < ROWS) & (qcols[None, :] < NC)
        blk = tl.load(W_ptr + qo, mask=qm, other=0.0).to(tl.float32)
        blk = blk * (row[:, None] <= qcols[None, :]).to(tl.float32)
        tl.store(W_ptr + qo, blk, mask=qm)


@triton.jit
def _vec_solve(
    R_ptr,
    X_ptr,
    RES_ptr,
    N,
    NC,
    NRHS,
    srb,
    sri,
    srj,
    sxb,
    sxn,
    sxr,
    seb,
    sec,
    RCOND,
    BLOCK: tl.constexpr,
):
    """Back-substitution + residuals; the frame spans N, NOT NC = n + nrhs.

    Sizing the frame by NC doubles it whenever one rhs column crosses a power
    of two -- a 256x256 problem with a single rhs has NC=257, so the frame was
    512 and the kernel needed 208KB of the 192KB UB, pushing that shape onto
    the serial `_vec_solve_blk` (measured 0.064x end-to-end).

    Nothing actually needs a lane past N-1: x, the dot product, r_ii, the
    diagonal and the store all live in [0, N). Only the rhs entry and the
    residual rows sit at column N+k, and those are single values -- read them
    as SCALARS and the frame is next_pow2(N).

    Measured (probe_vecsolve_frame.py): fits at frame 512 for N=512 as well,
    i.e. the whole 256..512 band leaves the serial path. Trimming named
    temporaries did NOT help (four variants all reported exactly 208KB) -- what
    mattered was shrinking the LOADED EXTENT from NC to N.

    `bad` is gone too: the NaN goes straight into the scalar xi, landing in the
    same lane the vector accumulator used to mark.
    """
    pid = tl.program_id(0)
    cc = tl.arange(0, BLOCK)
    rb = pid * srb

    diag = tl.load(R_ptr + rb + cc * sri + cc * srj, mask=cc < N, other=0.0).to(
        tl.float32
    )
    tol = RCOND * tl.max(tl.abs(diag), axis=0)

    # residuals: sum of squares of rows [N, NC) in the rhs column, as scalars
    for k in range(NRHS):
        acc = tl.zeros((1,), dtype=tl.float32)
        for i in range(N, NC):
            v = tl.load(R_ptr + rb + i * sri + (N + k) * srj).to(tl.float32)
            acc += v * v
        tl.store(RES_ptr + pid * seb + (N + k) * sec, tl.sum(acc, axis=0))

    for k in range(NRHS):
        x = tl.zeros((BLOCK,), dtype=tl.float32)
        for t in range(N):
            i = N - 1 - t
            r_i = tl.load(R_ptr + rb + i * sri + cc * srj, mask=cc < N, other=0.0).to(
                tl.float32
            )
            mi = (cc == i).to(tl.float32)
            r_ii = tl.sum(mi * r_i, axis=0)
            # rhs entry is at column N+k, outside the frame -> scalar load
            c_i = tl.load(R_ptr + rb + i * sri + (N + k) * srj).to(tl.float32)
            dot = tl.sum(r_i * (cc > i).to(tl.float32) * x, axis=0)
            d = (tl.abs(r_ii) <= tol).to(tl.float32)
            safe = r_ii * (1.0 - d) + d
            xi = (c_i - dot) / safe + tl.sqrt(-d)
            x = x * (1.0 - mi) + xi * mi
        tl.store(X_ptr + pid * sxb + cc * sxn + k * sxr, x, mask=cc < N)


@triton.jit
def _vec_solve_blk(
    R_ptr,
    X_ptr,
    RES_ptr,
    N,
    NC,
    NRHS,
    srb,
    sri,
    srj,
    sxb,
    sxn,
    sxr,
    seb,
    sec,
    RCOND,
    BLOCK: tl.constexpr,
):
    """Back-substitution + residuals with a FIXED tile: N and NC are LOOP BOUNDS.

    `_vec_solve` above now frames by N and fits through N=512, so this path is
    only for N > 512. It used to frame by NC = n + nrhs, which needed 208KB of
    the 192KB UB at width 512 (`ub overflow, requires 1704960 bits while
    1572864 bits available`) and dragged the whole 256..512 band onto this
    serial kernel.

    Here nothing scales: ~4 tiles of BLOCK stay live whatever N and NC are. The
    partial solution lives in the X output buffer between steps rather than in
    registers, which costs O(N/BLOCK) extra loads per row and buys no ceiling.

    Ascend rules observed throughout: bounds-only load masks with row/column
    offsets applied ARITHMETICALLY (a mask carrying a runtime offset returns
    wrong data here), injective stores, and no in-loop tl.where.
    """
    pid = tl.program_id(0)
    rb = pid * srb

    # ---- tol = RCOND * max |R[i,i]| over i < N ----
    tacc = tl.zeros((1,), dtype=tl.float32)
    for cb in range(0, N, BLOCK):
        cc = cb + tl.arange(0, BLOCK)
        d = tl.load(R_ptr + rb + cc * sri + cc * srj, mask=cc < N, other=0.0).to(
            tl.float32
        )
        tacc = tl.maximum(tacc, tl.max(tl.abs(d), axis=0))
    tol = RCOND * tl.sum(tacc, axis=0)

    # ---- residuals: RES[N+k] = sum over rows i in [N, NC) of R[i, N+k]^2 ----
    for k in range(NRHS):
        acc = tl.zeros((1,), dtype=tl.float32)
        for ib in range(0, NC, BLOCK):
            ii = ib + tl.arange(0, BLOCK)
            r = tl.load(
                R_ptr + rb + ii * sri + (N + k) * srj, mask=ii < NC, other=0.0
            ).to(tl.float32)
            r = r * (ii >= N).to(tl.float32)  # offset applied arithmetically
            acc += tl.sum(r * r, axis=0)
        tl.store(RES_ptr + pid * seb + (N + k) * sec, tl.sum(acc, axis=0))

    # ---- back-substitution, one row at a time, x kept in X ----
    for k in range(NRHS):
        for t in range(N):
            i = N - 1 - t
            dot = tl.zeros((1,), dtype=tl.float32)
            for jb in range(0, N, BLOCK):
                jj = jb + tl.arange(0, BLOCK)
                rij = tl.load(
                    R_ptr + rb + i * sri + jj * srj, mask=jj < N, other=0.0
                ).to(tl.float32)
                xj = tl.load(
                    X_ptr + pid * sxb + jj * sxn + k * sxr, mask=jj < N, other=0.0
                ).to(tl.float32)
                dot += tl.sum(rij * xj * (jj > i).to(tl.float32), axis=0)
            r_ii = tl.load(R_ptr + rb + i * sri + i * srj).to(tl.float32)
            c_i = tl.load(R_ptr + rb + i * sri + (N + k) * srj).to(tl.float32)
            dd = (tl.abs(r_ii) <= tol).to(tl.float32)
            safe = r_ii * (1.0 - dd) + dd  # never zero, no Inf formed
            # sqrt(-1) = NaN on a dead pivot, matching the deficient-gels
            # contract; sqrt(-0) is 0 so a healthy pivot is untouched
            xi = (c_i - tl.sum(dot, axis=0)) / safe + tl.sqrt(-dd)
            tl.store(X_ptr + pid * sxb + i * sxn + k * sxr, xi)
            tl.debug_barrier()  # RAW: the next row reads x[i]


@triton.jit
def _wide_solve(
    R_ptr,
    B_ptr,
    W_ptr,
    M,
    NRHS,
    srb,
    sri,
    srj,
    sbb,
    sbi,
    sbj,
    swb,
    swi,
    swj,
    RCOND,
    BLOCK_M: tl.constexpr,
):
    """w = R^-1 R^-T b using VECTORS only — never an m x m tile.

    The earlier version kept all of R resident as a (BLOCK_M, BLOCK_M) tile and
    sliced rows/columns out of it with masks. That m^2 term is what capped the
    underdetermined path at m <= 64 (128^2 fp32 = 64KB, over the 36KB budget).

    Here R is read one column (forward solve) or one row (back solve) at a time
    with a strided load, exactly as the tall path's `_vec_solve` does, and the
    rhs are handled one at a time. Live data is ~7 vectors of BLOCK_M, so the
    footprint is LINEAR in m: 7KB at m=256 versus 272KB for the tile form.

    Mask-only selects, no `tl.where`; the store is injective.
    """
    pid = tl.program_id(0)
    cc = tl.arange(0, BLOCK_M)
    rb = pid * srb

    # diagonal via a strided vector load (no tile)
    diag = tl.load(R_ptr + rb + cc * sri + cc * srj, mask=cc < M, other=0.0).to(
        tl.float32
    )
    tol = RCOND * tl.max(tl.abs(diag) * (cc < M).to(tl.float32), axis=0)

    for k in range(NRHS):
        b_k = tl.load(
            B_ptr + pid * sbb + cc * sbi + k * sbj, mask=cc < M, other=0.0
        ).to(tl.float32)
        bad = tl.zeros((BLOCK_M,), dtype=tl.float32)

        # forward substitution R^T z = b  -> needs COLUMN i of R
        z = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for i in range(M):
            coli = tl.load(R_ptr + rb + cc * sri + i * srj, mask=cc < M, other=0.0).to(
                tl.float32
            )
            mi = (cc == i).to(tl.float32)
            r_ii = tl.sum(mi * coli, axis=0)
            b_i = tl.sum(mi * b_k, axis=0)
            s = tl.sum(coli * (cc < i).to(tl.float32) * z, axis=0)
            d = (tl.abs(r_ii) <= tol).to(tl.float32)
            safe = r_ii * (1.0 - d) + d
            z_i = (b_i - s) / safe
            z = z * (1.0 - mi) + z_i * mi
            bad = bad + mi * d

        # back substitution R w = z  -> needs ROW i of R
        w = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for t in range(M):
            i = M - 1 - t
            rowi = tl.load(R_ptr + rb + i * sri + cc * srj, mask=cc < M, other=0.0).to(
                tl.float32
            )
            mi = (cc == i).to(tl.float32)
            r_ii = tl.sum(mi * rowi, axis=0)
            z_i = tl.sum(mi * z, axis=0)
            s = tl.sum(rowi * (cc > i).to(tl.float32) * w, axis=0)
            d = (tl.abs(r_ii) <= tol).to(tl.float32)
            safe = r_ii * (1.0 - d) + d
            w_i = (z_i - s) / safe
            w = w * (1.0 - mi) + w_i * mi
            bad = bad + mi * d

        # NaN on deficient rows only (bad may be 2 — both loops flag it; sqrt of
        # any negative is NaN, and sqrt(-0) = -0 leaves good rows untouched)
        w = w + tl.sqrt(-bad)
        tl.store(W_ptr + pid * swb + cc * swi + k * swj, w, mask=cc < M)


@triton.jit
def _atw_gemv(
    A_ptr,
    W_ptr,
    X_ptr,
    M,
    N,
    NRHS,
    sab,
    sai,
    saj,
    swb,
    swi,
    swj,
    sxb,
    sxi,
    sxj,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """x = A^T w. Reads A through its strides, so no transpose is materialised.
    Grid is (batch, ceil(N / BLOCK_N)), so n is unbounded."""
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    j = tl.arange(0, BLOCK_M)  # over m
    i = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # over n
    k = tl.arange(0, BLOCK_R)  # over nrhs

    a = tl.load(
        A_ptr + pid_b * sab + j[:, None] * sai + i[None, :] * saj,
        mask=(j[:, None] < M) & (i[None, :] < N),
        other=0.0,
    ).to(tl.float32)
    wv = tl.load(
        W_ptr + pid_b * swb + j[:, None] * swi + k[None, :] * swj,
        mask=(j[:, None] < M) & (k[None, :] < NRHS),
        other=0.0,
    ).to(tl.float32)

    acc = tl.zeros((BLOCK_N, BLOCK_R), dtype=tl.float32)
    for kk in range(NRHS):
        mk = (k == kk).to(tl.float32)
        wk = tl.sum(mk[None, :] * wv, axis=1)  # column kk of w
        xk = tl.sum(a * wk[:, None], axis=0)  # (BLOCK_N,)
        acc = acc * (1.0 - mk[None, :]) + xk[:, None] * mk[None, :]

    # injective: (i, k) is unique per lane
    tl.store(
        X_ptr + pid_b * sxb + i[:, None] * sxi + k[None, :] * sxj,
        acc,
        mask=(i[:, None] < N) & (k[None, :] < NRHS),
    )


# ------------------------------------------------------------- compact-WY QR
# Panel-blocked QR keeping each panel's reflectors in compact-WY form
# (Q = I - V T V^T). Rows are STREAMED through global memory, so M and NC are
# both LOOP BOUNDS and no tile scales with the problem. That is what removes the
# n+nrhs ceiling: the fan-in TSQR path above needs block_m >= NC, so a large NC
# forces few huge row-chunks and eventually will not fit at all.
#
# Ascend-specific, and measured rather than assumed: a load mask containing a
# RUNTIME ROW OFFSET (e.g. `rows >= j`) returns WRONG DATA on this backend --
# reproduced with no QR involved in probe_ascend_rowmask.py, worst relative
# error 1.21e-01, with the failing offsets moving as BLOCK_R changes. Every load
# below therefore masks for BOUNDS ONLY and applies the row offset
# arithmetically. This is the same class as note (2) in the module docstring.
_WY_P = 16
# Panel rows that fit in registers: M*P*4 bytes must stay under the ~36KB
# mask-kernel budget, so 512*16*4 = 32KB is the largest measured-good size.
_WY_PANEL_RESIDENT_MAX = 512
# BLOCK_R=128, not 256: _wy_update needs 237KB of UB at 256 against 192KB
# available, and BLOCK_R dominates that cost (at 256 the requirement only falls
# 237 -> 221 -> 205KB as BLOCK_C goes 64 -> 32 -> 16, so it never fits).
_WY_BLOCK_R = 128
_WY_BLOCK_C = 64


@triton.jit
def _wy_panel(
    W_ptr,
    TAU_ptr,
    M,
    NC,
    J0,
    PW,
    swb,
    swi,
    swj,
    stb,
    BLOCK_R: tl.constexpr,
    P: tl.constexpr,
):
    """Factor W[:, J0:J0+PW] in place; emit tau. Rows streamed, M a loop bound."""
    b = tl.program_id(0)
    wb = b * swb
    kk = tl.arange(0, P)
    pcols = J0 + kk

    for k in range(PW):
        j = J0 + k

        # ---- ||x||^2 for x = W[j:, j] ----
        acc = tl.zeros((1,), dtype=tl.float32)
        for rb in range(0, M, BLOCK_R):
            rows = rb + tl.arange(0, BLOCK_R)
            # bounds-only LOAD mask. A load mask containing a RUNTIME ROW
            # OFFSET returns wrong data here -- reproduced with no QR in
            # probe_ascend_rowmask.py (worst 1.21e-01, and the failing
            # offsets move with BLOCK_R). Apply the offset arithmetically.
            x = tl.load(W_ptr + wb + rows * swi + j * swj, mask=rows < M, other=0.0).to(
                tl.float32
            )
            x = x * (rows >= j).to(tl.float32)
            acc += tl.sum(x * x, axis=0)

        x0 = tl.load(W_ptr + wb + j * swi + j * swj).to(tl.float32)
        nrm = tl.sqrt(tl.sum(acc, axis=0))
        sgn = 1.0 - 2.0 * (x0 < 0.0).to(tl.float32)
        alpha = -sgn * nrm
        den = x0 - alpha
        sf = (tl.abs(den) > 0.0).to(tl.float32)  # 1 if the reflector exists
        anz = (tl.abs(alpha) > 0.0).to(tl.float32)
        # tau = sf ? (alpha - x0) / (alpha or 1) : 0     -- no Inf formed
        tau = (alpha - x0) / (alpha * anz + (1.0 - anz)) * sf * anz
        tl.store(TAU_ptr + b * stb + k, tau)
        tl.debug_barrier()  # WAR: norm reads first

        # ---- write v (strict lower) and R[j,j] = alpha ----
        den_d = den * sf + (1.0 - sf)  # never zero
        for rb in range(0, M, BLOCK_R):
            rows = rb + tl.arange(0, BLOCK_R)
            rmv = rows < M
            x = tl.load(W_ptr + wb + rows * swi + j * swj, mask=rmv, other=0.0).to(
                tl.float32
            )
            # rows > j get v; rows <= j are rewritten with their own value,
            # so the store mask is bounds-only too. The diagonal is set to
            # alpha by the store just below, after this identity rewrite.
            gt = (rows > j).to(tl.float32)
            val = x * (gt / den_d * sf + (1.0 - gt))
            tl.store(W_ptr + wb + rows * swi + j * swj, val, mask=rmv)
        tl.store(W_ptr + wb + j * swi + j * swj, alpha)
        tl.debug_barrier()  # RAW: v visible

        # ---- w = v^T @ panel[:, k+1:PW] ----
        w = tl.zeros((P,), dtype=tl.float32)
        for rb in range(0, M, BLOCK_R):
            rows = rb + tl.arange(0, BLOCK_R)
            rm = rows < M
            vv = tl.load(W_ptr + wb + rows * swi + j * swj, mask=rm, other=0.0).to(
                tl.float32
            )
            # vv := 1 on row j, 0 above it, loaded value below  (mask, not where)
            meq = (rows == j).to(tl.float32)
            mlt = (rows < j).to(tl.float32)
            vv = vv * (1.0 - meq) * (1.0 - mlt) + meq
            off = wb + rows[:, None] * swi + pcols[None, :] * swj
            tm = rm[:, None] & (kk[None, :] < PW) & (kk[None, :] > k)
            A = tl.load(W_ptr + off, mask=tm, other=0.0).to(tl.float32)
            w += tl.sum(vv[:, None] * A, axis=0)
        tl.debug_barrier()  # WAR: w read before update

        # ---- panel[:, k+1:PW] -= tau * v w^T ----
        for rb in range(0, M, BLOCK_R):
            rows = rb + tl.arange(0, BLOCK_R)
            rm = rows < M
            vv = tl.load(W_ptr + wb + rows * swi + j * swj, mask=rm, other=0.0).to(
                tl.float32
            )
            meq = (rows == j).to(tl.float32)
            mlt = (rows < j).to(tl.float32)
            vv = vv * (1.0 - meq) * (1.0 - mlt) + meq
            off = wb + rows[:, None] * swi + pcols[None, :] * swj
            tm = rm[:, None] & (kk[None, :] < PW) & (kk[None, :] > k)
            A = tl.load(W_ptr + off, mask=tm, other=0.0).to(tl.float32)
            tl.store(W_ptr + off, A - tau * vv[:, None] * w[None, :], mask=tm)
        tl.debug_barrier()  # RAW: next column sees it


@triton.jit
def _wy_panel_res(
    W_ptr,
    TAU_ptr,
    T_ptr,
    M,
    NC,
    J0,
    PW,
    swb,
    swi,
    swj,
    stb,
    sTb,
    sTi,
    BLOCK_M: tl.constexpr,
    P: tl.constexpr,
):
    """Panel factorization with the whole panel RESIDENT: one load, one store.

    `_wy_panel` streams rows, so it re-reads the (BLOCK_R, P) trailing tile
    twice and writes it once PER COLUMN -- ~48 tile round-trips for a PW=16
    panel. Its cost fits time ~= 0.83ms + 6.1e-4 * (M * PW), i.e. the tile
    traffic dominates. The panel itself is only M*P*4 bytes (32KB at M=512,
    P=16), which fits the ~36KB budget, so it can stay in registers for all PW
    columns instead: 48 round-trips -> 2.

    Measured against the streaming kernel (input staged on device, launches
    only): 2.98x at M=128, 5.81x at M=256, 11.65x at M=512, and FLAT at
    ~0.45ms across all three -- the M*PW term collapses to fixed overhead.
    Accuracy 5.5e-08 vs a float64 CPU Householder panel.

    Only valid while M <= BLOCK_M, so this is a fast path; `_wy_panel` stays
    for taller inputs, where M must remain a loop bound.
    """
    b = tl.program_id(0)
    wb = b * swb
    rows = tl.arange(0, BLOCK_M)
    kk = tl.arange(0, P)
    rm = (rows < M).to(tl.float32)
    pcols = J0 + kk
    off = wb + rows[:, None] * swi + pcols[None, :] * swj
    keep = (rows[:, None] < M) & (kk[None, :] < PW)

    A = tl.load(W_ptr + off, mask=keep, other=0.0).to(tl.float32)
    T = tl.zeros((P, P), dtype=tl.float32)

    for k in range(PW):
        j = J0 + k
        csel = (kk == k).to(tl.float32)
        x = tl.sum(A * csel[None, :], axis=1)
        gt = (rows > j).to(tl.float32) * rm
        eq = (rows == j).to(tl.float32) * rm
        lt = (rows < j).to(tl.float32) * rm

        xm = x * (gt + eq)
        nrm = tl.sqrt(tl.sum(xm * xm, axis=0))
        x0 = tl.sum(x * eq, axis=0)
        sgn = 1.0 - 2.0 * (x0 < 0.0).to(tl.float32)
        alpha = -sgn * nrm
        den = x0 - alpha
        sf = (tl.abs(den) > 0.0).to(tl.float32)
        anz = (tl.abs(alpha) > 0.0).to(tl.float32)
        tau = (alpha - x0) / (alpha * anz + (1.0 - anz)) * sf * anz
        tl.store(TAU_ptr + b * stb + k, tau)
        den_d = den * sf + (1.0 - sf)

        v = x * gt / den_d * sf + eq
        colv = x * gt / den_d * sf + eq * alpha + x * lt
        A = A * (1.0 - csel[None, :]) + colv[:, None] * csel[None, :]

        w = tl.sum(v[:, None] * A, axis=0)

        # T built here from the w already in hand -- _wy_buildT folded in. Its
        # recurrence needs G[:, k] for rows i < k, and
        #   w_i    = A[j,i] + sum_{r>j} A[r,i]*A[r,k]
        #   G[i,k] = A[J0+k,i] + sum_{r>J0+k} A[r,i]*A[r,k]     (j == J0+k)
        # are the same expression: v's zero prefix below row j does the
        # restriction for free. T is (P, P), so this costs no extra tile --
        # which is what makes it fit. A version needing a second (BLOCK_M, P)
        # tile would be 64KB, measured at 348KB required against 192KB.
        # Removes one launch of three per panel; ~2x on panel+buildT.
        mlt = (kk < k).to(tl.float32)
        z = w * mlt
        sT = tl.sum(T * z[None, :], axis=1)
        col = mlt * (-tau * sT) + csel * tau
        T = T * (1.0 - csel[None, :]) + col[:, None] * csel[None, :]

        trail = (kk > k).to(tl.float32)
        A = A - tau * v[:, None] * (w * trail)[None, :]

    tl.store(W_ptr + off, A, mask=keep)
    tl.store(
        T_ptr + b * sTb + kk[:, None] * sTi + kk[None, :],
        T,
        mask=(kk[:, None] < PW) & (kk[None, :] < PW),
    )


@triton.jit
def _wy_buildT(
    W_ptr,
    TAU_ptr,
    T_ptr,
    M,
    J0,
    PW,
    swb,
    swi,
    swj,
    stb,
    sTb,
    sTi,
    BLOCK_R: tl.constexpr,
    P: tl.constexpr,
):
    """T (P x P upper) with Q = I - V T V^T, from the Gram matrix G = V^T V."""
    b = tl.program_id(0)
    wb = b * swb
    kk = tl.arange(0, P)
    piv = J0 + kk

    G = tl.zeros((P, P), dtype=tl.float32)
    for rb in range(0, M, BLOCK_R):
        rows = rb + tl.arange(0, BLOCK_R)
        rm = rows < M
        vo = wb + rows[:, None] * swi + piv[None, :] * swj
        vm = rm[:, None] & (kk[None, :] < PW)
        V = tl.load(W_ptr + vo, mask=vm, other=0.0).to(tl.float32)
        meq = ((rows[:, None] == piv[None, :]) & vm).to(tl.float32)
        mlt = (rows[:, None] < piv[None, :]).to(tl.float32)
        V = V * (1.0 - meq) * (1.0 - mlt) + meq
        G += tl.dot(tl.trans(V), V, input_precision="ieee")

    T = tl.zeros((P, P), dtype=tl.float32)
    for k in range(PW):
        tau = tl.load(TAU_ptr + b * stb + k).to(tl.float32)
        mk = (kk == k).to(tl.float32)
        mlt = (kk < k).to(tl.float32)
        z = tl.sum(mk[None, :] * G, axis=1) * mlt
        s = tl.sum(T * z[None, :], axis=1)
        col = mlt * (-tau * s) + mk * tau
        T = T * (1.0 - mk[None, :]) + col[:, None] * mk[None, :]

    tl.store(
        T_ptr + b * sTb + kk[:, None] * sTi + kk[None, :],
        T,
        mask=(kk[:, None] < PW) & (kk[None, :] < PW),
    )


@triton.jit
def _wy_update(
    W_ptr,
    T_ptr,
    M,
    NC,
    J0,
    PW,
    swb,
    swi,
    swj,
    sTb,
    sTi,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    P: tl.constexpr,
):
    """trailing -= V @ (T^T @ (V^T @ trailing)), gridded over column blocks."""
    b = tl.program_id(0)
    cb = tl.program_id(1)
    wb = b * swb
    kk = tl.arange(0, P)
    piv = J0 + kk
    cols = J0 + PW + cb * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = cols < NC

    Wacc = tl.zeros((P, BLOCK_C), dtype=tl.float32)
    # NOTE: lower bound MUST be the literal 0, not J0. bishengir-compile
    # SIGSEGVs on a loop with a runtime-valued lower bound (proven by
    # probe_wy_ladder.py: every kernel using range(0,..) compiles, every
    # one using range(J0,..) crashes). Rows < J0 have V == 0 after the
    # mask reconstruction below, so they contribute exactly nothing to
    # either V^T@trailing or V@Y -- starting at 0 is the same arithmetic.
    for rb in range(0, M, BLOCK_R):
        rows = rb + tl.arange(0, BLOCK_R)
        rmask = rows < M
        vo = wb + rows[:, None] * swi + piv[None, :] * swj
        vm = rmask[:, None] & (kk[None, :] < PW)
        Vb = tl.load(W_ptr + vo, mask=vm, other=0.0).to(tl.float32)
        meq = ((rows[:, None] == piv[None, :]) & vm).to(tl.float32)
        mlt = (rows[:, None] < piv[None, :]).to(tl.float32)
        Vb = Vb * (1.0 - meq) * (1.0 - mlt) + meq
        to = wb + rows[:, None] * swi + cols[None, :] * swj
        Tb = tl.load(W_ptr + to, mask=rmask[:, None] & cmask[None, :], other=0.0).to(
            tl.float32
        )
        Wacc += tl.dot(tl.trans(Vb), Tb, input_precision="ieee")
    tl.debug_barrier()  # WAR before overwrite

    tof = tl.load(
        T_ptr + b * sTb + kk[:, None] * sTi + kk[None, :],
        mask=(kk[:, None] < PW) & (kk[None, :] < PW),
        other=0.0,
    ).to(tl.float32)
    Y = tl.dot(tl.trans(tof), Wacc, input_precision="ieee")

    # literal 0 lower bound again -- see the note above
    for rb in range(0, M, BLOCK_R):
        rows = rb + tl.arange(0, BLOCK_R)
        rmask = rows < M
        vo = wb + rows[:, None] * swi + piv[None, :] * swj
        vm = rmask[:, None] & (kk[None, :] < PW)
        Vb = tl.load(W_ptr + vo, mask=vm, other=0.0).to(tl.float32)
        meq = ((rows[:, None] == piv[None, :]) & vm).to(tl.float32)
        mlt = (rows[:, None] < piv[None, :]).to(tl.float32)
        Vb = Vb * (1.0 - meq) * (1.0 - mlt) + meq
        to = wb + rows[:, None] * swi + cols[None, :] * swj
        tm = rmask[:, None] & cmask[None, :]
        Tb = tl.load(W_ptr + to, mask=tm, other=0.0).to(tl.float32)
        tl.store(W_ptr + to, Tb - tl.dot(Vb, Y, input_precision="ieee"), mask=tm)


@triton.jit
def _zero_subdiag(
    R_ptr,
    N,
    srb,
    sri,
    srj,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Zero R[i, j] for i > j.

    _wy_panel leaves each reflector v in the strict lower triangle, and
    _vec_solve sums WHOLE rows for the residuals, so the reflectors have to go
    before it runs. Bounds-only masks, arithmetic zeroing, injective addresses.
    """
    b = tl.program_id(0)
    rb = tl.program_id(1)
    cb = tl.program_id(2)
    rows = rb * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = cb * BLOCK_C + tl.arange(0, BLOCK_C)
    msk = (rows[:, None] < N) & (cols[None, :] < N)
    off = b * srb + rows[:, None] * sri + cols[None, :] * srj
    v = tl.load(R_ptr + off, mask=msk, other=0.0).to(tl.float32)
    v = v * (rows[:, None] <= cols[None, :]).to(tl.float32)
    tl.store(R_ptr + off, v, mask=msk)


def wy_R(aug, NC):
    """QR of a tall (batch, M, BNC) matrix -> R (batch, BNC, BNC). No NC limit.

    Validated on 910B against a float64 CPU reference at 8 shapes including
    square 1024x1024 and 2048x1024 (relerr <= 5.8e-07); every stage also checked
    in isolation and composed.
    """
    batch, M, BNC = aug.shape
    P = _WY_P
    if M < BNC:
        # The zero rows exist ONLY so the (BNC, BNC) slice at the end is square.
        # Keep M at the REAL row count: appending zeros changes neither R nor
        # any reflector (a zero row adds nothing to a norm and stays zero under
        # every update), so the kernels should not sweep them. This also decides
        # whether the resident panel is reachable -- a 512x512 problem has
        # BNC=1024, and padding M to 1024 used to push it past the 512-row
        # register ceiling and back onto the streaming kernel.
        aug = torch.cat(
            [aug, torch.zeros(batch, BNC - M, BNC, dtype=aug.dtype, device=aug.device)],
            dim=1,
        )
    W = aug.contiguous()
    TAU = torch.zeros((batch, P), dtype=torch.float32, device=W.device)
    T = torch.zeros((batch, P, P), dtype=torch.float32, device=W.device)

    for J0 in range(0, NC, P):
        PW = min(P, NC - J0)
        # resident panel while the rows fit in registers (11.65x at M=512);
        # the streaming kernel keeps M as a loop bound for taller inputs
        panel_k = _wy_panel_res if M <= _WY_PANEL_RESIDENT_MAX else _wy_panel
        panel_kw = (
            {"BLOCK_M": _next_pow2(M)}
            if M <= _WY_PANEL_RESIDENT_MAX
            else {"BLOCK_R": _WY_BLOCK_R}
        )
        resident = M <= _WY_PANEL_RESIDENT_MAX
        if resident:
            # the resident panel emits T itself: one launch instead of two
            panel_k[(batch,)](
                W,
                TAU,
                T,
                M,
                NC,
                J0,
                PW,
                W.stride(0),
                W.stride(1),
                W.stride(2),
                TAU.stride(0),
                T.stride(0),
                T.stride(1),
                P=P,
                **panel_kw,
            )
        else:
            panel_k[(batch,)](
                W,
                TAU,
                M,
                NC,
                J0,
                PW,
                W.stride(0),
                W.stride(1),
                W.stride(2),
                TAU.stride(0),
                P=P,
                **panel_kw,
            )
        ntrail = NC - J0 - PW
        if ntrail > 0 and not resident:
            _wy_buildT[(batch,)](
                W,
                TAU,
                T,
                M,
                J0,
                PW,
                W.stride(0),
                W.stride(1),
                W.stride(2),
                TAU.stride(0),
                T.stride(0),
                T.stride(1),
                BLOCK_R=_WY_BLOCK_R,
                P=P,
            )
        if ntrail > 0:
            _wy_update[(batch, triton.cdiv(ntrail, _WY_BLOCK_C))](
                W,
                T,
                M,
                NC,
                J0,
                PW,
                W.stride(0),
                W.stride(1),
                W.stride(2),
                T.stride(0),
                T.stride(1),
                BLOCK_R=_WY_BLOCK_R,
                BLOCK_C=_WY_BLOCK_C,
                P=P,
            )

    # may alias W when M == BNC; harmless, W is dead from here
    R = W[:, :BNC, :].contiguous()
    _zero_subdiag[(batch, triton.cdiv(BNC, 64), triton.cdiv(BNC, 64))](
        R,
        BNC,
        R.stride(0),
        R.stride(1),
        R.stride(2),
        BLOCK_R=64,
        BLOCK_C=64,
    )
    return R


def tsqr_R_panel(aug, NC):
    """QR of a tall (batch, M, BNC) matrix -> R (batch, BNC, BNC), any NC."""
    batch, M, BNC = aug.shape
    # each chunk needs >= NC rows for its R to be complete
    BR0 = max(_next_pow2(NC), 256)
    P0 = _panel_width(BR0)
    n_chunks = triton.cdiv(M, BR0)
    pad = n_chunks * BR0 - M
    if pad:
        aug = torch.cat(
            [aug, torch.zeros(batch, pad, BNC, dtype=aug.dtype, device=aug.device)],
            dim=1,
        )
    W = aug.reshape(batch * n_chunks, BR0, BNC).contiguous()
    _panel_qr[(W.shape[0],)](
        W,
        BR0,
        NC,
        W.stride(0),
        W.stride(1),
        W.stride(2),
        BLOCK_ROWS=BR0,
        PANEL=P0,
    )
    # _panel_qr already zeroed the sub-diagonal, so this slice IS R
    R = W[:, :BNC, :].contiguous().reshape(batch, n_chunks, BNC, BNC)

    live = n_chunks
    G = 2
    while live > 1:
        ng = triton.cdiv(live, G)
        padb = ng * G - live
        if padb:
            R = torch.cat(
                [R, torch.zeros(batch, padb, BNC, BNC, dtype=R.dtype, device=R.device)],
                dim=1,
            )
        rows = G * BNC
        BR = _next_pow2(rows)
        P = _panel_width(BR)
        S = R.reshape(batch, ng, rows, BNC)
        if BR > rows:
            S = torch.cat(
                [
                    S,
                    torch.zeros(
                        batch, ng, BR - rows, BNC, dtype=R.dtype, device=R.device
                    ),
                ],
                dim=2,
            )
        S = S.reshape(batch * ng, BR, BNC).contiguous()
        _panel_qr[(S.shape[0],)](
            S,
            rows,
            NC,
            S.stride(0),
            S.stride(1),
            S.stride(2),
            BLOCK_ROWS=BR,
            PANEL=P,
        )
        R = S[:, :BNC, :].contiguous().reshape(batch, ng, BNC, BNC)
        live = ng
    return R.reshape(batch, BNC, BNC).contiguous()


def lstsq_tall_panel(A, B, rcond=None):
    """m >= n, any m, any n+nrhs. -> (X, residuals)"""
    batch, m, n = A.shape
    nrhs = B.shape[-1]
    NC = n + nrhs
    BNC = _next_pow2(NC)
    if rcond is None:
        rcond = torch.finfo(torch.float32).eps * max(m, n)

    aug = torch.zeros((batch, m, BNC), dtype=torch.float32, device=A.device)
    aug[:, :, :n] = A
    aug[:, :, n:NC] = B
    # fan-in TSQR where it fits (proven path), compact-WY beyond it. WY has no
    # NC ceiling at all -- M and NC are loop bounds there, not tile dimensions.
    Rf = tsqr_R_panel(aug, NC) if BNC <= _MAX_BNC else wy_R(aug, NC)

    X = torch.zeros((batch, n, nrhs), dtype=torch.float32, device=A.device)
    RES = torch.zeros((batch, BNC), dtype=torch.float32, device=A.device)
    # The SOLVE is chosen independently of the QR: its frame spans N, so the
    # rhs columns no longer drag it over a power of two. Measured to fit at
    # frame 512, which covers the whole 256..512 band that previously fell onto
    # the serial blocked solve.
    BN = _next_pow2(max(n, 1))
    if BN <= _SOLVE_MAX_FRAME:
        _vec_solve[(batch,)](
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
            BLOCK=BN,
        )
    else:
        # beyond that the frame itself is the problem, so the solve is blocked
        # (serial over rows -- correct, but the slow path; see the doc)
        _vec_solve_blk[(batch,)](
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
            BLOCK=256,
        )
    return X, RES[:, n:NC].contiguous()


@triton.jit
def _wide_solve_blk(
    R_ptr,
    B_ptr,
    Z_ptr,
    W_ptr,
    M,
    NRHS,
    srb,
    sri,
    srj,
    sbb,
    sbi,
    sbj,
    szb,
    szi,
    szj,
    swb,
    swi,
    swj,
    RCOND,
    BLOCK: tl.constexpr,
):
    """w = R^-1 R^-T b with a FIXED tile, so m is a LOOP BOUND.

    `_wide_solve` above is linear in m -- ~7 live vectors of BLOCK_M -- which is
    why it capped the underdetermined path at m <= 256. Linear is still growth:
    the tall path's `_vec_solve` has the same shape and needed 208KB of the
    192KB UB at BLOCK_M=512. This keeps ~3 tiles of BLOCK live at any m.

    z (forward) and w (back) live in global buffers between rows instead of in
    registers. Bounds-only load masks with the triangular offsets applied
    ARITHMETICALLY, injective stores, one barrier per row for the RAW.
    """
    pid = tl.program_id(0)
    rb = pid * srb

    tacc = tl.zeros((1,), dtype=tl.float32)
    for cb in range(0, M, BLOCK):
        cc = cb + tl.arange(0, BLOCK)
        d = tl.load(R_ptr + rb + cc * sri + cc * srj, mask=cc < M, other=0.0).to(
            tl.float32
        )
        tacc = tl.maximum(tacc, tl.max(tl.abs(d), axis=0))
    tol = RCOND * tl.sum(tacc, axis=0)

    for k in range(NRHS):
        # ---- forward: R^T z = b, needs COLUMN i of R ----
        for i in range(M):
            acc = tl.zeros((1,), dtype=tl.float32)
            for jb in range(0, M, BLOCK):
                jj = jb + tl.arange(0, BLOCK)
                rji = tl.load(
                    R_ptr + rb + jj * sri + i * srj, mask=jj < M, other=0.0
                ).to(tl.float32)
                zj = tl.load(
                    Z_ptr + pid * szb + jj * szi + k * szj, mask=jj < M, other=0.0
                ).to(tl.float32)
                acc += tl.sum(rji * zj * (jj < i).to(tl.float32), axis=0)
            r_ii = tl.load(R_ptr + rb + i * sri + i * srj).to(tl.float32)
            b_i = tl.load(B_ptr + pid * sbb + i * sbi + k * sbj).to(tl.float32)
            dd = (tl.abs(r_ii) <= tol).to(tl.float32)
            safe = r_ii * (1.0 - dd) + dd
            zi = (b_i - tl.sum(acc, axis=0)) / safe + tl.sqrt(-dd)
            tl.store(Z_ptr + pid * szb + i * szi + k * szj, zi)
            tl.debug_barrier()

        # ---- back: R w = z, needs ROW i of R ----
        for t in range(M):
            i = M - 1 - t
            acc = tl.zeros((1,), dtype=tl.float32)
            for jb in range(0, M, BLOCK):
                jj = jb + tl.arange(0, BLOCK)
                rij = tl.load(
                    R_ptr + rb + i * sri + jj * srj, mask=jj < M, other=0.0
                ).to(tl.float32)
                wj = tl.load(
                    W_ptr + pid * swb + jj * swi + k * swj, mask=jj < M, other=0.0
                ).to(tl.float32)
                acc += tl.sum(rij * wj * (jj > i).to(tl.float32), axis=0)
            r_ii = tl.load(R_ptr + rb + i * sri + i * srj).to(tl.float32)
            z_i = tl.load(Z_ptr + pid * szb + i * szi + k * szj).to(tl.float32)
            dd = (tl.abs(r_ii) <= tol).to(tl.float32)
            safe = r_ii * (1.0 - dd) + dd
            wi = (z_i - tl.sum(acc, axis=0)) / safe + tl.sqrt(-dd)
            tl.store(W_ptr + pid * swb + i * swi + k * swj, wi)
            tl.debug_barrier()


@triton.jit
def _atw_gemv_blk(
    A_ptr,
    W_ptr,
    X_ptr,
    M,
    N,
    NRHS,
    sab,
    sai,
    saj,
    swb,
    swi,
    swj,
    sxb,
    sxn,
    sxr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """X = A^T w, blocked over BOTH dimensions so m is a loop bound.

    `_atw_gemv` sizes its A tile as BLOCK_N x BLOCK_M with BLOCK_M = next_pow2(m);
    at m=1024 its own budget formula clamps BLOCK_N to the floor of 16 and still
    asks for a 64KB tile. Here both extents are fixed and m is accumulated over.
    """
    b = tl.program_id(0)
    nb = tl.program_id(1)
    cols = nb * BLOCK_N + tl.arange(0, BLOCK_N)
    cmask = cols < N
    for k in range(NRHS):
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for ib in range(0, M, BLOCK_M):
            ii = ib + tl.arange(0, BLOCK_M)
            rmask = ii < M
            a = tl.load(
                A_ptr + b * sab + ii[:, None] * sai + cols[None, :] * saj,
                mask=rmask[:, None] & cmask[None, :],
                other=0.0,
            ).to(tl.float32)
            wv = tl.load(
                W_ptr + b * swb + ii * swi + k * swj, mask=rmask, other=0.0
            ).to(tl.float32)
            acc += tl.sum(a * wv[:, None], axis=0)
        tl.store(X_ptr + b * sxb + cols * sxn + k * sxr, acc, mask=cmask)


def _empty_rank_sv(A):
    return (
        torch.empty(0, dtype=torch.int64, device=A.device),
        torch.empty(0, dtype=A.dtype, device=A.device),
    )


def _lstsq_wide(A, B, rcond):
    """Minimum-norm solve for m < n: x = A^T R^-1 R^-T b, A^T = QR."""
    batch, m, n = A.shape
    nrhs = B.shape[-1]
    BM = _next_pow2(m)
    BR = _next_pow2(nrhs)

    big = BM > _MAX_BNC
    # MEASURED: unlike the tall solve, _wide_solve does NOT survive frame 512.
    # Raising this gate to _SOLVE_MAX_FRAME broke underdetermined_wy at
    # (512,1024), (300,700) and batched (512,1024) -- all BM=512 -- while
    # (1024,2048) kept passing because BM=1024 stayed on the blocked path. The
    # tall solve's win did not transfer: it came from shrinking the LOADED
    # EXTENT from NC to N, and _wide_solve was already framed by M, so there
    # was nothing analogous left to shrink. Keep the QR ceiling here.
    big_solve = BM > _MAX_BNC

    At = A.transpose(-1, -2).contiguous()
    aug = torch.zeros((batch, n, BM), dtype=torch.float32, device=A.device)
    aug[:, :, :m] = At
    # the QR of A^T has short dimension m, so a large m collapses the fan-in
    # tree exactly as square does on the tall path -- compact-WY has no such
    # ceiling, since m and n are loop bounds there
    R = wy_R(aug, m) if big else tsqr_R_panel(aug, m)

    Bp = torch.zeros((batch, BM, BR), dtype=torch.float32, device=A.device)
    Bp[:, :m, :nrhs] = B
    w = torch.zeros((batch, BM, BR), dtype=torch.float32, device=A.device)
    if big_solve:
        Z = torch.zeros((batch, BM, BR), dtype=torch.float32, device=A.device)
        _wide_solve_blk[(batch,)](
            R,
            Bp,
            Z,
            w,
            m,
            nrhs,
            R.stride(0),
            R.stride(1),
            R.stride(2),
            Bp.stride(0),
            Bp.stride(1),
            Bp.stride(2),
            Z.stride(0),
            Z.stride(1),
            Z.stride(2),
            w.stride(0),
            w.stride(1),
            w.stride(2),
            rcond,
            BLOCK=256,
        )
    else:
        _wide_solve[(batch,)](
            R,
            Bp,
            w,
            m,
            nrhs,
            R.stride(0),
            R.stride(1),
            R.stride(2),
            Bp.stride(0),
            Bp.stride(1),
            Bp.stride(2),
            w.stride(0),
            w.stride(1),
            w.stride(2),
            rcond,
            BLOCK_M=BM,
        )

    # reserve room for wv (BM x BR) and acc, not just the A tile
    BLOCK_N = max(16, min(128, (_BUDGET_ELEMS - BM * BR) // max(BM, 1)))
    BLOCK_N = 1 << (BLOCK_N.bit_length() - 1)
    X = torch.zeros((batch, n, nrhs), dtype=torch.float32, device=A.device)
    if big:
        _atw_gemv_blk[(batch, triton.cdiv(n, 64))](
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
            BLOCK_M=64,
            BLOCK_N=64,
        )
    else:
        _atw_gemv[(batch, triton.cdiv(n, BLOCK_N))](
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
            BLOCK_M=BM,
            BLOCK_N=BLOCK_N,
            BLOCK_R=BR,
        )
    return X


def linalg_lstsq(A, b, rcond=None, driver=None):
    logger.debug("GEMS_ASCEND LINALG_LSTSQ")

    if driver not in (None, "gels"):
        raise RuntimeError(
            "torch.linalg.lstsq: `driver` other than `gels` is not " "supported on CUDA"
        )
    if A.dtype != torch.float32:
        raise NotImplementedError(
            f"linalg_lstsq on Ascend supports float32 only (got {A.dtype}); "
            "this device has no float64"
        )
    if A.dim() < 2:
        raise RuntimeError("torch.linalg.lstsq: input must have at least 2 dimensions.")
    if b.dim() < 1:
        raise RuntimeError("torch.linalg.lstsq: other must have at least 1 dimension.")

    m, n = A.shape[-2], A.shape[-1]

    # VECTOR rhs iff b has one fewer dim AND matches A.shape[:-1] exactly
    # (torch does not broadcast the vector case); MATRIX rhs iff equal ndim,
    # where batch dims DO broadcast.
    dim_diff = A.dim() - b.dim()
    if dim_diff == 1 and tuple(b.shape) == tuple(A.shape[:-1]):
        vector_rhs, b2 = True, b.unsqueeze(-1)
    elif dim_diff == 0:
        vector_rhs, b2 = False, b
    else:
        raise RuntimeError(
            "torch.linalg.lstsq: input.dim() must be greater or equal to "
            "other.dim() and (input.dim() - other.dim()) <= 1"
        )
    if b2.shape[-2] != m:
        raise RuntimeError(
            "torch.linalg.lstsq: input.size(-2) should match other.size(-2)"
        )
    nrhs = b2.shape[-1]

    batch_shape = torch.broadcast_shapes(A.shape[:-2], b2.shape[:-2])
    rank, singular_values = _empty_rank_sv(A)

    # degenerate dims: LAPACK ?gels quick-returns having zeroed its buffer
    if m == 0 or n == 0 or nrhs == 0:
        solution = torch.zeros((*batch_shape, n, nrhs), dtype=A.dtype, device=A.device)
        if vector_rhs:
            solution = solution.squeeze(-1)
        residuals = (
            torch.zeros((*batch_shape, nrhs), dtype=A.dtype, device=A.device)
            if m > n
            else torch.empty(0, dtype=A.dtype, device=A.device)
        )
        return solution, residuals, rank, singular_values

    if rcond is None:
        rcond = torch.finfo(A.dtype).eps * max(m, n)

    Af = A.expand(*batch_shape, m, n).reshape(-1, m, n).contiguous()
    Bf = b2.expand(*batch_shape, m, nrhs).reshape(-1, m, nrhs).contiguous()

    if m < n:
        X = _lstsq_wide(Af, Bf, rcond)
        residuals = torch.empty(0, dtype=A.dtype, device=A.device)
    else:
        X, RES = lstsq_tall_panel(Af, Bf, rcond)
        # torch returns residuals only for m > n, and keeps them at (*, nrhs)
        # even when b was a vector -- do NOT squeeze them.
        residuals = (
            RES.reshape(*batch_shape, nrhs)
            if m > n
            else torch.empty(0, dtype=A.dtype, device=A.device)
        )

    solution = X.reshape(*batch_shape, n, nrhs)
    if vector_rhs:
        solution = solution.squeeze(-1)
    return solution, residuals, rank, singular_values

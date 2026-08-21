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
"""Ascend private implementation of ``torch.linalg.qr``.

Same algorithm family as the generic implementation (plain/blocked
Householder QR + TSQR for tall-skinny inputs), with kernels written against
the patterns the Ascend 910B vector backend compiles correctly:

* no ``tl.dot``: all linear algebra is done with 1-D contiguous vectors and
  2-D affine tiles reduced with ``tl.sum`` (the generic implementation's
  ``tl.dot``/SRAM-resident kernels overflow the 192 KB unified buffer here);
* no vector ``tl.where`` stores: masks are arithmetic (``(mask) * value``);
* large matrices are staged in padded column-major buffers (``ld`` is a
  multiple of the row-block size) so every vectorised load/store is
  contiguous and aligned.

Routing (per input shape, mirroring the generic implementation):

* tall-skinny (m >> n, n <= 64): multi-level TSQR -- 128-row chunks factored
  in parallel, the stacked chunk R factors reduced recursively, Q assembled
  through the level chain;
* few rows (m <= 128): a single register-blocked kernel does the whole
  factorisation with the column tiles register-resident;
* otherwise: blocked panel factorisation (multi-CTA panels with an
  atomic-barrier row split when the matrix is tall enough) + per-panel
  trailing updates + a single-launch Q assembly.

Engineering constraints on this backend (all measured, see
``linalg_qr_ascend_notes.md``):

* a single launch with more blocks than the 910B vector-core count (40)
  corrupts the first block's output for the multi-loop kernels, so grids are
  capped at ``_MAX_BLOCKS`` via batch/column chunking;
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.linalg_qr import _validate_mode
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

_RM = 128
_TN = 64
# Panel width for the blocked factorization.  Also the column-chunk width of
# the in-panel reflector application, so a whole panel is one (RM, TN) tile.
_NB = 64
# Vector-core count of the 910B: launching one of the multi-loop kernels with
# more blocks than this corrupts the first block's output, so oversized grids
# are processed in chunks (see ``_batch_chunks`` / ``_axis1_chunks``).
_MAX_BLOCKS = 40
# Cap on the row/column-tile grid axis of the streaming copy kernels (they
# grid-stride any remaining tiles).  Beyond a few tiles per row/column the
# extra CTAs cost more in launch scheduling than they buy in parallelism.
_CPY_TILES = 8


def _cpy_tiles(axis1, numel, block):
    """Tile-axis size for the streaming copy kernels: split tiles across CTAs
    for parallelism, but cap the total CTA count -- past ~16k one-tile CTAs
    the scheduling overhead outweighs the parallelism (huge copies such as
    the 8192x8192 complete-mode Q regressed ~5% without this)."""
    return min(_grid_elem(numel, block), _CPY_TILES, max(1, 16384 // axis1))


def _grid_elem(numel, block):
    return (numel + block - 1) // block


def _tile_width(cols):
    """Adaptive work-tile width: the panel / trailing / Q kernels move
    (RM, width) tiles, and a fixed 64-wide tile wastes 16x the bandwidth on
    narrow matrices such as 4096x4."""
    return max(1, min(_TN, triton.next_power_of_2(cols)))


def _par_tile_width(cols, batches):
    """Tile width for the column-parallel kernels (trailing update, Q
    assembly).  A wide 64-column tile leaves most vector cores idle on small
    matrices (e.g. 256x256: 3 CTAs of 40), so narrow the tile until the
    column chunks roughly fill the free cores; never below 16 lanes, and
    never narrower than what a many-batch launch can afford (oversubscribed
    grids are serialized by ``_axis1_chunks`` into extra launches)."""
    w = _tile_width(cols)
    want = max(1, _MAX_BLOCKS // batches)
    while w > 16 and (cols + w - 1) // w < want:
        w //= 2
    return w


def _batch_chunks(B):
    """(offset, size) chunks of the batch axis, capped at _MAX_BLOCKS."""
    if B <= _MAX_BLOCKS:
        return ((0, B),)
    return tuple((b0, min(_MAX_BLOCKS, B - b0)) for b0 in range(0, B, _MAX_BLOCKS))


def _bslice(args, b0, bn):
    if b0 == 0 and args[0].shape[0] == bn:
        return args
    return tuple(t[b0 : b0 + bn] if isinstance(t, torch.Tensor) else t for t in args)


def _axis1_chunks(units, blocks0):
    """(offset, count) chunks of a grid's second axis such that the total
    block count of each launch stays within _MAX_BLOCKS."""
    cap = max(1, _MAX_BLOCKS // blocks0)
    if units <= cap:
        return ((0, units),)
    return tuple((u0, min(cap, units - u0)) for u0 in range(0, units, cap))


# ---------------------------------------------------------------------------
# Kernel 1: staged copy A -> Wc.  One CTA per (batch, column, row tile): the
# Wc side is a contiguous vector store, the A side is a strided gather with
# arbitrary user strides.  (The previous scalar version needed ~34 ms for a
# 512x512 fp32 matrix on a single CTA; the serial per-CTA tile loop was
# latency-bound, hence the row-tile axis -- streaming copies are unaffected
# by the _MAX_BLOCKS grid cap.)
# ---------------------------------------------------------------------------
@triton.jit
def _copy_a_to_wc(A, Wc, m, n, ld, sAb, sAm, sAn, sWb, RM: tl.constexpr):
    pid_b = tl.program_id(0)
    j = tl.program_id(1)
    # grid-stride over row tiles: the host caps axis 2 so huge copies do not
    # spawn hundreds of thousands of one-tile CTAs (scheduling overhead
    # outweighs the extra parallelism there)
    for t in range(tl.program_id(2), tl.cdiv(m, RM), tl.num_programs(2)):
        rows = t * RM + tl.arange(0, RM)
        rmask = rows < m
        v = tl.load(A + pid_b * sAb + j * sAn + rows * sAm, mask=rmask, other=0.0)
        tl.store(Wc + pid_b * sWb + j * ld + rows, v, mask=rmask)


# ---------------------------------------------------------------------------
# Kernel 2: panel factorization.  One CTA per batch factors the nb columns
# j0..j0+nb-1 of Wc (unblocked Householder) and applies each reflector to the
# remaining panel columns.  Only scalars and (TN,) accumulators are carried
# across the reflector loop; the (RM, TN) work tiles are reloaded from global
# memory every pass, which is the pattern the Ascend backend compiles
# correctly.  With WRITE_R (TSQR final level only: j0 == 0, nb == n <= 64)
# the n x n upper triangle is also written straight to R, saving a separate
# copy-out launch.
# ---------------------------------------------------------------------------
@triton.jit
def _panel_kernel(
    Wc,
    Vc,
    TAU,
    R,
    m,
    j0,
    nb,
    ld,
    sWb,
    sVb,
    sTauB,
    sTauN,
    sRb,
    sRm,
    sRn,
    RM: tl.constexpr,
    PW: tl.constexpr,
    WRITE_R: tl.constexpr,
):
    pid_b = tl.program_id(0)
    Wb = Wc + pid_b * sWb
    Vb = Vc + pid_b * sVb
    TAUb = TAU + pid_b * sTauB
    dt = Wc.dtype.element_ty
    zero = tl.full((), 0.0, dtype=dt)
    one = tl.full((), 1.0, dtype=dt)
    rows0 = tl.arange(0, RM)
    cols = tl.arange(0, PW)
    cmask = cols < nb
    num_tiles = (m + RM - 1) // RM

    for jj in range(nb):
        j = j0 + jj
        Wcol = Wb + j * ld
        Vcol = Vb + j * ld

        alpha = zero
        xnorm_sq = zero
        for t in range(num_tiles):
            rows = t * RM + rows0
            rmask = rows < m
            col = tl.load(Wcol + rows, mask=rmask, other=zero)
            alpha += tl.sum((rmask & (rows == j)) * col)
            xnorm_sq += tl.sum((rmask & (rows > j)) * col * col)

        norm = tl.sqrt(alpha * alpha + xnorm_sq)
        beta = tl.where(alpha >= zero, -norm, norm)
        reflect = xnorm_sq > zero
        reflect_f = tl.where(reflect, one, zero)
        beta_eff = tl.where(reflect, beta, alpha)
        tau = tl.where(reflect, (beta - alpha) / beta, zero)
        denom = alpha - beta
        denom_safe = tl.where(reflect, denom, one)

        tl.store(TAUb + j * sTauN, tau)

        for t in range(num_tiles):
            rows = t * RM + rows0
            rmask = rows < m
            col = tl.load(Wcol + rows, mask=rmask, other=zero)
            vtail = (col / denom_safe) * reflect_f
            v = (rows > j) * vtail + (rows == j) * one
            tl.store(Vcol + rows, v, mask=rmask)
            wval = col + (rows == j) * (beta_eff - col)
            tl.store(Wcol + rows, wval, mask=rmask)

        # apply the new reflector to the panel columns right of j (relative
        # indices jj+1..nb-1); columns <= jj keep w == 0 and store back
        # unchanged
        w = tl.zeros((PW,), dtype=dt)
        for t in range(num_tiles):
            rows = t * RM + rows0
            rmask = rows < m
            v = tl.load(Vcol + rows, mask=rmask, other=zero)
            x = tl.load(
                Wb + (j0 + cols)[None, :] * ld + rows[:, None],
                mask=rmask[:, None] & cmask[None, :],
                other=zero,
            )
            w += tl.sum(v[:, None] * x, axis=0)

        w = tau * w * cmask * (cols > jj)

        for t in range(num_tiles):
            rows = t * RM + rows0
            rmask = rows < m
            v = tl.load(Vcol + rows, mask=rmask, other=zero)
            x = tl.load(
                Wb + (j0 + cols)[None, :] * ld + rows[:, None],
                mask=rmask[:, None] & cmask[None, :],
                other=zero,
            )
            x = x - v[:, None] * w[None, :]
            tl.store(
                Wb + (j0 + cols)[None, :] * ld + rows[:, None],
                x,
                mask=rmask[:, None] & cmask[None, :],
            )

    if WRITE_R:
        # TSQR final level (j0 == 0, nb == n <= 64): the panel is the whole
        # stack, so the n x n upper triangle goes straight to R here, saving
        # a separate copy-out launch.  Same arithmetic-mask store pattern as
        # _triu_copy_cm.
        rr = tl.arange(0, 64)
        rc = tl.arange(0, 64)
        rmask2 = (rr < nb)[:, None] & (rc < nb)[None, :]
        rv = tl.load(Wb + rc[None, :] * ld + rr[:, None], mask=rmask2, other=zero)
        tl.store(
            R + pid_b * sRb + rr[:, None] * sRm + rc[None, :] * sRn,
            rv * (rr[:, None] <= rc[None, :]),
            mask=rmask2,
        )


# ---------------------------------------------------------------------------
# Kernel 2b: multi-CTA panel factorisation.  The single-CTA panel kernel is
# bandwidth-bound applying each reflector across the whole panel; this variant
# gives one 64-row band to each CTA.  Per reflector the band partials
# (alpha / xnorm / w) go to fixed slots, every CTA sums them after an atomic
# spin barrier, so the work and the bandwidth scale with the band count.
# Slot-based partials keep the result deterministic (no floating-point
# atomics).  Three barriers per reflector: partials written -> sums/apply
# done -> apply visible for the next reduce.
#
# The barrier counter must start at zero for every call and live in its own
# allocation (a counter embedded in a shape-dependent workspace gets
# overwritten by other shapes' data): the caller passes a fresh zeroed CTR
# per QR call and mirrors the running total in ``base``.
# ---------------------------------------------------------------------------
@triton.jit
def _panel_mcta_kernel(
    Wc,
    Vc,
    TAU,
    ABUF,
    XBUF,
    WBUF,
    CTR,
    m,
    j0,
    nb,
    base,
    nc,
    ld,
    sWb,
    sVb,
    sTauB,
    sTauN,
    RM: tl.constexpr,
    PW: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    Wb = Wc + pid_b * sWb
    Vb = Vc + pid_b * sVb
    TAUb = TAU + pid_b * sTauB
    abuf = ABUF + pid_b * sTauB * nc  # (k, nc) partial slots per batch
    xbuf = XBUF + pid_b * sTauB * nc
    wbuf = WBUF + pid_b * (64 * nc * PW)  # reflector slots, nb <= 64
    ctr = CTR + pid_b
    dt = Wc.dtype.element_ty
    zero = tl.full((), 0.0, dtype=dt)
    one = tl.full((), 1.0, dtype=dt)
    rows0 = tl.arange(0, RM)
    rows = pid_c * RM + rows0
    rmask = rows < m
    cols = tl.arange(0, PW)
    cmask = cols < nb
    cnt = base
    # dep carries a dependency on the last barrier's spin result into the
    # panel loads (always 0, see the barrier comment below); initialised
    # here so it is loop-carried across reflectors.
    dep = tl.minimum(base, 0)

    for jj in range(nb):
        j = j0 + jj
        Wcol = Wb + j * ld
        Vcol = Vb + j * ld

        # band partials for alpha / xnorm of column j
        col = tl.load(Wcol + rows + dep, mask=rmask, other=zero)
        pa = tl.sum((rmask & (rows == j)) * col)
        px = tl.sum((rmask & (rows > j)) * col * col)
        tl.store(abuf + j * nc + pid_c, pa)
        tl.store(xbuf + j * nc + pid_c, px)
        tl.atomic_add(ctr, 1, sem="release")
        # Barrier spin: an atomic inside the spin loop segfaults CANN 9.0.0's
        # bishengir-compile, so poll with a volatile load; the loop-free
        # acquire atomic below restores the read-side ordering.  The loads
        # of barrier-protected buffers must depend on the spin result (via
        # dep, always 0): otherwise the 9.0.0 compiler hoists them above
        # the spin loop and reads stale partials.
        cw = tl.load(ctr, volatile=True)
        while cw < cnt + nc:
            cw = tl.load(ctr, volatile=True)
        tl.atomic_add(ctr, 0, sem="acquire")
        cnt += nc
        dep = tl.minimum(cw - cnt, 0)

        alpha = zero
        xnorm_sq = zero
        for q in range(nc):
            alpha += tl.load(abuf + j * nc + q + dep)
            xnorm_sq += tl.load(xbuf + j * nc + q + dep)

        norm = tl.sqrt(alpha * alpha + xnorm_sq)
        beta = tl.where(alpha >= zero, -norm, norm)
        reflect = xnorm_sq > zero
        reflect_f = tl.where(reflect, one, zero)
        beta_eff = tl.where(reflect, beta, alpha)
        tau = tl.where(reflect, (beta - alpha) / beta, zero)
        denom = alpha - beta
        denom_safe = tl.where(reflect, denom, one)
        if pid_c == 0:
            tl.store(TAUb + j * sTauN, tau)

        v = (rows == j) * one + (rows > j) * (col / denom_safe) * reflect_f
        tl.store(Vcol + rows, v, mask=rmask)
        # full-column store (rewrites the tail with identical values): a
        # single-lane masked store of a broadcast scalar crashes the vector
        # core on this backend
        wval = col + (rows == j) * (beta_eff - col)
        tl.store(Wcol + rows, wval, mask=rmask)

        # within-panel apply: band partial of w = v^T X_panel
        x = tl.load(
            Wb + (j0 + cols)[None, :] * ld + rows[:, None] + dep,
            mask=rmask[:, None] & cmask[None, :],
            other=zero,
        )
        pw = tl.sum(v[:, None] * x, axis=0)
        tl.store(wbuf + (jj * nc + pid_c) * PW + cols, pw, mask=cmask)
        tl.atomic_add(ctr, 1, sem="release")
        cw = tl.load(ctr, volatile=True)
        while cw < cnt + nc:
            cw = tl.load(ctr, volatile=True)
        tl.atomic_add(ctr, 0, sem="acquire")
        cnt += nc
        dep = tl.minimum(cw - cnt, 0)

        w = tl.zeros((PW,), dtype=dt)
        for q in range(nc):
            w += tl.load(wbuf + (jj * nc + q) * PW + cols + dep, mask=cmask, other=zero)
        w = tau * w * cmask * (cols > jj)
        x = x - v[:, None] * w[None, :]
        tl.store(
            Wb + (j0 + cols)[None, :] * ld + rows[:, None],
            x,
            mask=rmask[:, None] & cmask[None, :],
        )
        tl.atomic_add(ctr, 1, sem="release")
        cw = tl.load(ctr, volatile=True)
        while cw < cnt + nc:
            cw = tl.load(ctr, volatile=True)
        tl.atomic_add(ctr, 0, sem="acquire")
        cnt += nc
        dep = tl.minimum(cw - cnt, 0)


# ---------------------------------------------------------------------------
# Kernel 3: trailing update.  One CTA per (batch, column chunk) applies all
# nb reflectors of panel j0 to the trailing columns c0..c0+p-1.  Replaces nb
# separate launches with one.
# ---------------------------------------------------------------------------
@triton.jit
def _trailing_apply_kernel(
    Vc,
    TAU,
    Wc,
    m,
    j0,
    nb,
    c0,
    p,
    coff,
    ld,
    sVb,
    sTauB,
    sTauN,
    sWb,
    RM: tl.constexpr,
    PW: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = coff + tl.program_id(1)
    Vb = Vc + pid_b * sVb
    TAUb = TAU + pid_b * sTauB
    Wb = Wc + pid_b * sWb
    dt = Wc.dtype.element_ty
    zero = tl.full((), 0.0, dtype=dt)
    rows0 = tl.arange(0, RM)
    cols = pid_c * PW + tl.arange(0, PW)
    cmask = cols < p
    num_tiles = (m + RM - 1) // RM

    for r in range(nb):
        j = j0 + r
        tau = tl.load(TAUb + j * sTauN)
        w = tl.zeros((PW,), dtype=dt)
        for t in range(num_tiles):
            rows = t * RM + rows0
            rmask = rows < m
            v = tl.load(Vb + j * ld + rows, mask=rmask, other=zero)
            x = tl.load(
                Wb + (c0 + cols)[None, :] * ld + rows[:, None],
                mask=rmask[:, None] & cmask[None, :],
                other=zero,
            )
            w += tl.sum(v[:, None] * x, axis=0)

        w = tau * w * cmask

        for t in range(num_tiles):
            rows = t * RM + rows0
            rmask = rows < m
            v = tl.load(Vb + j * ld + rows, mask=rmask, other=zero)
            x = tl.load(
                Wb + (c0 + cols)[None, :] * ld + rows[:, None],
                mask=rmask[:, None] & cmask[None, :],
                other=zero,
            )
            x = x - v[:, None] * w[None, :]
            tl.store(
                Wb + (c0 + cols)[None, :] * ld + rows[:, None],
                x,
                mask=rmask[:, None] & cmask[None, :],
            )


# ---------------------------------------------------------------------------
# Kernel 4: Q assembly in a single launch.  One CTA per (batch, column chunk)
# writes the identity block of Qc, then applies every reflector k-1..0 (the
# two-pass dot/update structure, with the reflector loop inside the kernel).
# No 2-D tile is carried across reflector iterations.
# ---------------------------------------------------------------------------
@triton.jit
def _q_apply_kernel(
    Vc,
    TAU,
    Qc,
    m,
    k,
    qcols,
    coff,
    ld,
    sVb,
    sTauB,
    sTauN,
    sQcb,
    sQcm,
    sQcn,
    RM: tl.constexpr,
    PW: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = coff + tl.program_id(1)
    Vb = Vc + pid_b * sVb
    TAUb = TAU + pid_b * sTauB
    Qb = Qc + pid_b * sQcb
    dt = Qc.dtype.element_ty
    zero = tl.full((), 0.0, dtype=dt)
    rows0 = tl.arange(0, RM)
    cols = pid_c * PW + tl.arange(0, PW)
    cmask = cols < qcols
    num_tiles = (m + RM - 1) // RM

    for t in range(num_tiles):
        rows = t * RM + rows0
        rmask = rows < m
        val = (rows[:, None] == cols[None, :]) * 1.0
        tl.store(
            Qb + rows[:, None] * sQcm + cols[None, :] * sQcn,
            val,
            mask=rmask[:, None] & cmask[None, :],
        )

    for jj in range(k):
        j = k - 1 - jj
        tau = tl.load(TAUb + j * sTauN)
        w = tl.zeros((PW,), dtype=dt)
        for t in range(num_tiles):
            rows = t * RM + rows0
            rmask = rows < m
            v = tl.load(Vb + j * ld + rows, mask=rmask, other=zero)
            x = tl.load(
                Qb + rows[:, None] * sQcm + cols[None, :] * sQcn,
                mask=rmask[:, None] & cmask[None, :],
                other=zero,
            )
            w += tl.sum(v[:, None] * x, axis=0)

        w = tau * w * cmask

        for t in range(num_tiles):
            rows = t * RM + rows0
            rmask = rows < m
            v = tl.load(Vb + j * ld + rows, mask=rmask, other=zero)
            x = tl.load(
                Qb + rows[:, None] * sQcm + cols[None, :] * sQcn,
                mask=rmask[:, None] & cmask[None, :],
                other=zero,
            )
            x = x - v[:, None] * w[None, :]
            tl.store(
                Qb + rows[:, None] * sQcm + cols[None, :] * sQcn,
                x,
                mask=rmask[:, None] & cmask[None, :],
            )


# ---------------------------------------------------------------------------
# Kernel 5/6: copy-out helpers, one CTA per (batch, TM x TN) tile.  Both the
# column-major source (contiguous along rows) and the row-major destination
# (contiguous along columns) are accessed as 2-D tiles, so both sides
# vectorise -- a row-per-CTA version gathers the source one element per
# cache line and is ~100x slower on large complete-mode Q.  Arithmetic
# (i <= j) replaces tl.where for the triangular mask.
# ---------------------------------------------------------------------------
@triton.jit
def _copy_qc_to_q(
    Qc, Q, m, qcols, ld, sQcb, sQb, sQm, sQn, TM: tl.constexpr, TN: tl.constexpr
):
    pid_b = tl.program_id(0)
    bi = tl.program_id(1)
    bj = tl.program_id(2)
    rows = bi * TM + tl.arange(0, TM)
    cols = bj * TN + tl.arange(0, TN)
    mask = (rows < m)[:, None] & (cols < qcols)[None, :]
    v = tl.load(
        Qc + pid_b * sQcb + rows[:, None] + cols[None, :] * ld, mask=mask, other=0.0
    )
    tl.store(Q + pid_b * sQb + rows[:, None] * sQm + cols[None, :] * sQn, v, mask=mask)


@triton.jit
def _triu_copy_cm(
    Wc, R, rrows, n, ld, sWb, sRb, sRm, sRn, TM: tl.constexpr, TN: tl.constexpr
):
    pid_b = tl.program_id(0)
    bi = tl.program_id(1)
    bj = tl.program_id(2)
    rows = bi * TM + tl.arange(0, TM)
    cols = bj * TN + tl.arange(0, TN)
    mask = (rows < rrows)[:, None] & (cols < n)[None, :]
    v = tl.load(
        Wc + pid_b * sWb + rows[:, None] + cols[None, :] * ld, mask=mask, other=0.0
    )
    tl.store(
        R + pid_b * sRb + rows[:, None] * sRm + cols[None, :] * sRn,
        v * (rows[:, None] <= cols[None, :]),
        mask=mask,
    )


# Scalar fallbacks of the copy-outs for outputs whose last dimension is not
# contiguous (strided views as out=): vector stores on arbitrary user strides
# are one of the patterns the Ascend backend miscompiles.
@triton.jit
def _copy_qc_to_q_scalar(Qc, Q, m, qcols, ld, sQcb, sQb, sQm, sQn):
    pid_b = tl.program_id(0)
    for j in range(qcols):
        for i in range(m):
            v = tl.load(Qc + pid_b * sQcb + j * ld + i)
            tl.store(Q + pid_b * sQb + i * sQm + j * sQn, v)


@triton.jit
def _triu_copy_scalar(Wc, R, rrows, n, ld, sWb, sRb, sRm, sRn):
    pid_b = tl.program_id(0)
    for j in range(n):
        for i in range(rrows):
            v = tl.load(Wc + pid_b * sWb + j * ld + i)
            tl.store(R + pid_b * sRb + i * sRm + j * sRn, v * (i <= j))


# ---------------------------------------------------------------------------
# Kernel 7: register-blocked QR for matrices with few rows (m <= RM, any n).
# The active (RM, TN) column block lives in registers across its whole
# factorisation: reflectors of previous blocks are applied in registers (one
# 1-D V load each) and the block's own reflectors never round-trip W through
# global memory.  R is written once per block; V/TAU go to global for the Q
# assembly, which applies every reflector to a register-resident Qt tile.
# This is what makes wide inputs (n >> m) competitive: the unblocked scheme
# re-read the trailing matrix twice per reflector.  The classic "loop-carried
# 2-D register tile" pattern compiles here because the selects are arithmetic
# (no tl.where stores) and tiles stay within ~64x64 -- bigger ones overflow
# the 192 KB unified buffer once the compiler multi-buffers the loop nests.
# ---------------------------------------------------------------------------
@triton.jit
def _qr_reg_kernel(
    A,
    Vc,
    Q,
    R,
    TAU,
    m,
    n,
    k,
    qcols,
    rrows,
    ld,
    sAb,
    sAm,
    sAn,
    sVb,
    sQb,
    sQm,
    sQn,
    sRb,
    sRm,
    sRn,
    RM: tl.constexpr,
    TN: tl.constexpr,
    TQ: tl.constexpr,
    PUT_Q: tl.constexpr,
):
    pid_b = tl.program_id(0)
    Ab = A + pid_b * sAb
    Vb = Vc + pid_b * sVb
    Qb = Q + pid_b * sQb
    Rb = R + pid_b * sRb
    TAUb = TAU + pid_b * k
    dt = A.dtype.element_ty
    zero = tl.full((), 0.0, dtype=dt)
    one = tl.full((), 1.0, dtype=dt)
    rows0 = tl.arange(0, RM)
    cols0 = tl.arange(0, TN)
    rmask = rows0 < m

    for c0 in range(0, n, TN):
        cols = c0 + cols0
        cmask = cols < n
        T = tl.load(
            Ab + rows0[:, None] * sAm + cols[None, :] * sAn,
            mask=rmask[:, None] & cmask[None, :],
            other=zero,
        )

        # reflectors of all previous column blocks, applied in registers
        for j in range(0, min(k, c0)):
            tau = tl.load(TAUb + j)
            v = tl.load(Vb + j * ld + rows0, mask=rmask, other=zero)
            w = tau * tl.sum(v[:, None] * T, axis=0)
            T = T - v[:, None] * w[None, :]

        # factor this block's columns in registers
        for jj in range(min(TN, k - c0)):
            col = tl.sum(T * (cols0[None, :] == jj), axis=1)
            alpha = tl.sum((rows0 == jj) * col)
            xsq = tl.sum((rows0 > jj) * col * col)
            norm = tl.sqrt(alpha * alpha + xsq)
            beta = tl.where(alpha >= zero, -norm, norm)
            reflect = xsq > zero
            reflect_f = tl.where(reflect, one, zero)
            tau = tl.where(reflect, (beta - alpha) / beta, zero)
            denom = alpha - beta
            denom_safe = tl.where(reflect, denom, one)
            tl.store(TAUb + c0 + jj, tau)
            vtail = (col / denom_safe) * reflect_f * (rows0 > jj)
            v = vtail + (rows0 == jj) * one
            tl.store(Vb + (c0 + jj) * ld + rows0, v, mask=rmask)
            w = tau * tl.sum(v[:, None] * T, axis=0)
            T = T - v[:, None] * w[None, :]

        tl.store(
            Rb + rows0[:, None] * sRm + cols[None, :] * sRn,
            T * (rows0[:, None] <= cols[None, :]),
            mask=rmask[:, None] & cmask[None, :] & (rows0[:, None] < rrows),
        )

    if PUT_Q:
        qcols0 = tl.arange(0, TQ)
        qmask = qcols0 < qcols
        Qt = (rows0[:, None] == qcols0[None, :]) * one
        for rj in range(k):
            j = k - 1 - rj
            tau = tl.load(TAUb + j)
            v = tl.load(Vb + j * ld + rows0, mask=rmask, other=zero)
            w = tau * tl.sum(v[:, None] * Qt, axis=0)
            Qt = Qt - v[:, None] * w[None, :]
        tl.store(
            Qb + rows0[:, None] * sQm + qcols0[None, :] * sQn,
            Qt,
            mask=rmask[:, None] & qmask[None, :],
        )


# ---------------------------------------------------------------------------
# Kernel 8 (TSQR level 1): factor one mp-row chunk of A independently, one
# CTA per (batch, chunk).  Stores the chunk's n x n R into the stacked
# buffer, plus its reflectors V/tau for the level-3 Q assembly.  The chunk
# tile is register-resident, same as the wide kernel.
# ---------------------------------------------------------------------------
@triton.jit
def _tsqr_l1_kernel(
    A,
    R1,
    V1,
    TAU1,
    m,
    n,
    mp,
    coff,
    sAb,
    sAm,
    sAn,
    sRb,
    sRrb,
    sRrm,
    sRrn,
    sVb,
    sTb,
    RM: tl.constexpr,
    TN: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = coff + tl.program_id(1)
    roff = pid_c * mp
    Ab = A + pid_b * sAb
    Rb = R1 + pid_b * sRb + pid_c * sRrb
    Vb = V1 + pid_b * sVb + pid_c * n * RM
    TAUb = TAU1 + pid_b * sTb + pid_c * n
    dt = A.dtype.element_ty
    zero = tl.full((), 0.0, dtype=dt)
    one = tl.full((), 1.0, dtype=dt)
    rows0 = tl.arange(0, RM)
    cols0 = tl.arange(0, TN)
    rmask = (roff + rows0) < m
    cmask = cols0 < n

    T = tl.load(
        Ab + (roff + rows0)[:, None] * sAm + cols0[None, :] * sAn,
        mask=rmask[:, None] & cmask[None, :],
        other=zero,
    )

    for jj in range(n):
        col = tl.sum(T * (cols0[None, :] == jj), axis=1)
        alpha = tl.sum((rows0 == jj) * col)
        xsq = tl.sum((rows0 > jj) * col * col)
        norm = tl.sqrt(alpha * alpha + xsq)
        beta = tl.where(alpha >= zero, -norm, norm)
        reflect = xsq > zero
        reflect_f = tl.where(reflect, one, zero)
        tau = tl.where(reflect, (beta - alpha) / beta, zero)
        denom = alpha - beta
        denom_safe = tl.where(reflect, denom, one)
        tl.store(TAUb + jj, tau)
        vtail = (col / denom_safe) * reflect_f * (rows0 > jj)
        v = vtail + (rows0 == jj) * one
        tl.store(Vb + jj * RM + rows0, v, mask=rmask)
        w = tau * tl.sum(v[:, None] * T, axis=0)
        T = T - v[:, None] * w[None, :]

    tl.store(
        Rb + rows0[:, None] * sRrm + cols0[None, :] * sRrn,
        T * (rows0[:, None] <= cols0[None, :]),
        mask=(rows0[:, None] < n) & cmask[None, :],
    )


# ---------------------------------------------------------------------------
# Kernel 9 (TSQR level 3): the reduced Q is blockdiag(H_0..H_{P-1}) * Q2.
# Each row chunk takes Q2's n-row block for that chunk, pads it to the chunk
# height with zeros and applies its own level-1 reflectors in registers.
# ---------------------------------------------------------------------------
@triton.jit
def _tsqr_q_kernel(
    Q2,
    V1,
    TAU1,
    Q,
    m,
    n,
    mp,
    coff,
    sQ2b,
    sVb,
    sTb,
    sQb,
    sQm,
    sQn,
    RM: tl.constexpr,
    TQ: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = coff + tl.program_id(1)
    roff = pid_c * mp
    Q2b = Q2 + pid_b * sQ2b + pid_c * n * n
    Vb = V1 + pid_b * sVb + pid_c * n * RM
    TAUb = TAU1 + pid_b * sTb + pid_c * n
    Qb = Q + pid_b * sQb + roff * sQm
    dt = Q2.dtype.element_ty
    zero = tl.full((), 0.0, dtype=dt)
    rows0 = tl.arange(0, RM)
    cols0 = tl.arange(0, TQ)
    rmask = (roff + rows0) < m
    cmask = cols0 < n

    # Qt = pad(Q2 block): the top n rows carry this chunk's level-2 Q rows
    Qt = tl.load(
        Q2b + rows0[:, None] * n + cols0[None, :],
        mask=(rows0[:, None] < n) & cmask[None, :],
        other=zero,
    )

    for rj in range(n):
        j = n - 1 - rj
        tau = tl.load(TAUb + j)
        v = tl.load(Vb + j * RM + rows0, mask=rmask, other=zero)
        w = tau * tl.sum(v[:, None] * Qt, axis=0)
        Qt = Qt - v[:, None] * w[None, :]

    tl.store(
        Qb + rows0[:, None] * sQm + cols0[None, :] * sQn,
        Qt,
        mask=rmask[:, None] & cmask[None, :],
    )


def _reg_tile_cfg(m, n, qcols):
    """Pick (RM, TN, TQ) register-tile dimensions for the register-blocked
    kernel, or None when the rows do not fit a tile the unified buffer can
    hold.  TN is the column-chunk width (any n: wider matrices just run more
    chunks); the quantised dimensions keep the number of kernel variants --
    and therefore cold-start compiles -- bounded."""

    def q2(x):
        p = 8
        while p < x and p < 64:
            p *= 2
        return p

    qc = q2(qcols) if qcols else 8
    # column-chunk width: wider chunks mean fewer of them, and the
    # cross-chunk reflector application is quadratic in the chunk count.
    # Capped by next_pow2(n) so narrow matrices do not drag masked-out
    # lanes through every tile op.  The per-row caps are the widest
    # loop-carried register tiles the unified buffer accepts (64x128 and
    # 32x256 overflow it).
    tn_n = min(256, triton.next_power_of_2(max(n, 1)))
    if m <= 16:
        return (q2(m), min(256, tn_n), min(qc, 64))
    if m <= 32:
        return (q2(m), min(128, tn_n), min(qc, 64))
    if m <= 64:
        return (q2(m), min(64, tn_n), min(qc, 64))
    # m <= 128: single-column-chunk configs only.  The cross-chunk reflector
    # application is broken for RM=128 with more than one chunk (wrong
    # results for e.g. 128x64), so wider matrices fall through to the
    # staged paths.
    if m <= 128 and tn_n <= 32:
        tq = min(qc, 32)
        # complete mode needs all qcols Q columns in the register tile;
        # wider than the TQ cap they would never be written (a 128x128
        # loop-carried tile overflows the UB), so use the staged paths
        if qcols > tq:
            return None
        return (128, min(32, tn_n), tq)
    return None


def _ascend_linalg_qr(A, mode, out=None):
    batch_shape = A.shape[:-2]
    m, n = A.shape[-2], A.shape[-1]
    k = min(m, n)
    B = 1
    for d in batch_shape:
        B *= d

    if m == 0 or n == 0:
        # Degenerate input: no factorisation to run.  torch.linalg.qr returns
        # empty factors, except complete mode with zero columns where Q = I.
        # Same construction as the generic implementation.
        if mode == "r":
            q_shape, r_shape = (0,), (*batch_shape, k, n)
        elif mode == "reduced":
            q_shape, r_shape = (*batch_shape, m, k), (*batch_shape, k, n)
        else:
            q_shape, r_shape = (*batch_shape, m, m), (*batch_shape, m, n)
        if out is not None:
            Q, R = out
        else:
            Q = A.new_empty(q_shape)
            R = A.new_empty(r_shape)
        if mode == "complete" and n == 0 and m > 0:
            eye = torch.eye(m, dtype=A.dtype, device=A.device)
            Q.copy_(eye.expand(*batch_shape, m, m))
        return Q, R

    qcols = 0 if mode == "r" else (k if mode == "reduced" else m)
    rrows = k if mode in ("reduced", "r") else m

    out_Q = out_R = None
    if out is not None:
        out_Q, out_R = out
        out_Q = out_Q.reshape(B, m, qcols) if qcols else out_Q.reshape(0)
        out_R = out_R.reshape(B, rrows, n)

    # Padded column-major working matrices.  ld is always a multiple of the
    # vector block size (_RM), so every column starts at an aligned address.
    ld = max(_RM, triton.next_power_of_2(m))
    A_view = A.reshape(B, m, n)
    sAb, sAm, sAn = A_view.stride()

    with torch_device_fn.device(A.device):

        # Tall-skinny (m >> n, n <= 64): multi-level TSQR.  Rows are split
        # into 128-row chunks factored in parallel (level 1); each level's
        # stacked chunk R's become the next level's input (recursively) until
        # the stack is small enough for the staged panel machinery; Q is then
        # assembled through the level chain, innermost factors first.
        if (
            m > 128
            and n <= 64
            and m >= 4 * n
            and B <= _MAX_BLOCKS  # TSQR launches grid axis 0 = B unchunked
            and mode in ("reduced", "r")
            and A_view.stride(2) == 1
            and (out_Q is None or out_Q.dim() < 3 or out_Q.stride(2) == 1)
            and (out_R is None or out_R.stride(2) == 1)
        ):
            mp = 128
            lvls = []  # (input rows, chunks) per level
            rows = m
            while True:
                P = (rows + mp - 1) // mp
                lvls.append((rows, P))
                rows = P * n
                if rows <= 512:
                    break
            m2 = rows  # final stack rows, factored by the panel kernel
            ld2 = max(_RM, triton.next_power_of_2(m2))
            bands2 = (m2 + 63) // 64
            nc2 = bands2
            # n >= 16: for very narrow stacks the panel is a handful of
            # reflectors and the barrier overhead outweighs the parallelism
            l2_mcta = 2 <= bands2 and B * bands2 <= _MAX_BLOCKS and n >= 16
            # Small final stacks (<= 128 rows, e.g. 4096x4 -> a 128x4 stack)
            # are factored by the register kernel in a single launch that
            # writes Q2 / R straight to their final buffers, saving the
            # panel + Q2-assembly + R-copy launches (host-launch bound at
            # these sizes).
            reg2 = (
                _reg_tile_cfg(m2, n, n if mode == "reduced" else 0)
                if m2 <= 128
                else None
            )
            l2_reg = reg2 is not None
            if l2_reg:
                l2_mcta = False
            # workspace: per level R-stack (B, P, n, n), V (B, P, n, mp),
            # tau (B, P, n); plus the final staging block and, for Q, one
            # (B, rows, n) block per level.
            offs, tot = [], 0
            for rows_l, P_l in lvls:
                r1 = B * P_l * n * n
                v1 = B * P_l * n * mp
                t1 = B * P_l * n
                xq = B * P_l * n * n  # level output block (P_l*n rows, n)
                offs.append(
                    (
                        tot,
                        tot + r1,
                        tot + r1 + v1,
                        tot + r1 + v1 + t1,
                        tot + r1 + v1 + t1 + xq,
                    )
                )
                tot += r1 + v1 + t1 + xq
            oW2 = tot
            tot += B * (2 * n * ld2 + n)
            oL2 = tot
            if l2_mcta:
                tot += B * (2 * n * nc2 + 64 * nc2 * 64)
            tws = torch.empty(tot, dtype=A.dtype, device=A.device)
            R1s, V1s, T1s, Xs = [], [], [], []
            for o, oR1, oV1, oT1, oX in offs:
                P_l = lvls[len(R1s)][1]
                R1s.append(tws[o:oR1].view(B, P_l, n, n))
                V1s.append(tws[oR1:oV1].view(B, P_l, n, mp))
                T1s.append(tws[oV1:oT1].view(B, P_l, n))
                Xs.append(tws[oT1:oX].view(B, P_l * n, n))
            if not l2_reg:
                Wc2 = tws[oW2 : oW2 + B * n * ld2].view(B, n, ld2)
                Vc2 = tws[oW2 + B * n * ld2 : oW2 + B * 2 * n * ld2].view(B, n, ld2)
                tau2 = tws[oW2 + B * 2 * n * ld2 : oW2 + B * (2 * n * ld2 + n)].view(
                    B, n
                )
            R = (
                out_R
                if out_R is not None
                else torch.empty(B, n, n, dtype=A.dtype, device=A.device)
            )
            sRb, sRm, sRn = R.stride()
            tn1 = _tile_width(n)

            # level-1 chain: factor each level's chunks in parallel; the
            # stack of chunk R's feeds the next level
            src_t = A_view
            sSb, sSm, sSn = sAb, sAm, sAn
            last = len(lvls) - 1
            for li, (rows_l, P_l) in enumerate(lvls):
                # the last level's R blocks go straight into the column-major
                # panel staging buffer (or stay row-major when the register
                # kernel factors the final stack); earlier levels stack
                # row-major
                if li == last and not l2_reg:
                    dstR, sRrb, sRrm, sRrn = Wc2, n, 1, ld2
                    sRb_l = n * ld2
                else:
                    dstR, sRrb, sRrm, sRrn = R1s[li], n * n, n, 1
                    sRb_l = P_l * n * n
                for u0, un in _axis1_chunks(P_l, B):
                    _tsqr_l1_kernel[(B, un)](
                        src_t,
                        dstR,
                        V1s[li],
                        T1s[li],
                        rows_l,
                        n,
                        mp,
                        u0,
                        sSb,
                        sSm,
                        sSn,
                        sRb_l,
                        sRrb,
                        sRrm,
                        sRrn,
                        P_l * n * mp,
                        P_l * n,
                        RM=mp,
                        TN=tn1,
                    )
                if li != last:
                    src_t = R1s[li].view(B, P_l * n, n)
                    sSb, sSm, sSn = P_l * n * n, n, 1

            # final level: the last stack (already in Wc2, or row-major in
            # R1s[last] when l2_reg) is factored here
            if l2_reg:
                RM2, TN2, TQ2 = reg2
                stack = R1s[last].view(B, m2, n)
                sSb2, sSm2, sSn2 = stack.stride()
                ldv2 = 128
                Vc2r = torch.empty(B, n, ldv2, dtype=A.dtype, device=A.device)
                tau2r = torch.empty(B, n, dtype=A.dtype, device=A.device)
                qc2 = n if mode == "reduced" else 0
                if qc2:
                    Q2 = Xs[-1]
                    sQ2b, sQ2m, sQ2n = Q2.stride()
                else:
                    Q2 = torch.empty((0,), dtype=A.dtype, device=A.device)
                    sQ2b = sQ2m = sQ2n = 0
                for b0, bn in _batch_chunks(B):
                    Ss, V2s, T2s, Q2s, Rs = _bslice((stack, Vc2r, tau2r, Q2, R), b0, bn)
                    _qr_reg_kernel[(bn,)](
                        Ss,
                        V2s,
                        Q2s,
                        Rs,
                        T2s,
                        m2,
                        n,
                        n,
                        qc2,
                        n,
                        ldv2,
                        sSb2,
                        sSm2,
                        sSn2,
                        n * ldv2,
                        sQ2b,
                        sQ2m,
                        sQ2n,
                        sRb,
                        sRm,
                        sRn,
                        RM=RM2,
                        TN=TN2,
                        TQ=TQ2,
                        PUT_Q=bool(qc2),
                    )
                if mode == "r":
                    Q = (
                        out_Q
                        if out_Q is not None
                        else torch.empty((0,), dtype=A.dtype, device=A.device)
                    )
                    return Q, R.reshape(*batch_shape, n, n)
            elif l2_mcta:
                ab2 = tws[oL2 : oL2 + B * n * nc2].view(B, n, nc2)
                xb2 = tws[oL2 + B * n * nc2 : oL2 + B * 2 * n * nc2].view(B, n, nc2)
                wb2 = tws[
                    oL2 + B * 2 * n * nc2 : oL2 + B * (2 * n * nc2 + 64 * nc2 * 64)
                ].view(B, 64 * nc2, 64)
                # fresh zeroed barrier counters per call (start at 0)
                ct2 = torch.zeros(B, dtype=torch.int32, device=A.device)
                # no in-kernel R write here: an R-writer tail on this kernel
                # segfaults bishengir-compile on CANN 9.0.0 (CI), so this
                # branch keeps the separate triu copy-out below
                _panel_mcta_kernel[(B, nc2)](
                    Wc2,
                    Vc2,
                    tau2,
                    ab2,
                    xb2,
                    wb2,
                    ct2,
                    m2,
                    0,
                    n,
                    0,
                    nc2,
                    ld2,
                    n * ld2,
                    n * ld2,
                    n,
                    1,
                    RM=64,
                    PW=64,
                )
            else:
                _panel_kernel[(B,)](
                    Wc2,
                    Vc2,
                    tau2,
                    R,
                    m2,
                    0,
                    n,
                    ld2,
                    n * ld2,
                    n * ld2,
                    n,
                    1,
                    sRb,
                    sRm,
                    sRn,
                    RM=_RM,
                    PW=tn1,
                    WRITE_R=True,
                )
            if mode == "r":
                if l2_mcta:
                    for b0, bn in _batch_chunks(B):
                        W2s, Rs = _bslice((Wc2, R), b0, bn)
                        _triu_copy_cm[(bn, _grid_elem(n, 64), _grid_elem(n, _TN))](
                            W2s, Rs, n, n, ld2, n * ld2, sRb, sRm, sRn, TM=64, TN=_TN
                        )
                Q = (
                    out_Q
                    if out_Q is not None
                    else torch.empty((0,), dtype=A.dtype, device=A.device)
                )
                return Q, R.reshape(*batch_shape, n, n)

            # Q assembly through the chain, innermost first: with l2_reg the
            # register kernel has already written Q2 (Xs[-1]) and R; the
            # single-CTA panel has written R via WRITE_R; the mcta panel
            # still needs the separate triu copy-out below.
            Q = (
                out_Q
                if out_Q is not None
                else torch.empty(B, m, n, dtype=A.dtype, device=A.device)
            )
            sQb, sQm, sQn = Q.stride()
            if not l2_reg:
                _q_apply_kernel[(B, _grid_elem(n, tn1))](
                    Vc2,
                    tau2,
                    Xs[-1],
                    m2,
                    n,
                    n,
                    0,
                    ld2,
                    n * ld2,
                    n,
                    1,
                    m2 * n,
                    n,
                    1,
                    RM=_RM,
                    PW=tn1,
                )
            for li in range(len(lvls) - 1, -1, -1):
                rows_l, P_l = lvls[li]
                dst = Q if li == 0 else Xs[li - 1]
                dSb, dSm, dSn = dst.stride()
                for u0, un in _axis1_chunks(P_l, B):
                    _tsqr_q_kernel[(B, un)](
                        Xs[li],
                        V1s[li],
                        T1s[li],
                        dst,
                        rows_l,
                        n,
                        mp,
                        u0,
                        P_l * n * n,
                        P_l * n * mp,
                        P_l * n,
                        dSb,
                        dSm,
                        dSn,
                        RM=mp,
                        TQ=tn1,
                    )
            if l2_mcta:
                for b0, bn in _batch_chunks(B):
                    W2s, Rs = _bslice((Wc2, R), b0, bn)
                    _triu_copy_cm[(bn, _grid_elem(n, 64), _grid_elem(n, _TN))](
                        W2s, Rs, n, n, ld2, n * ld2, sRb, sRm, sRn, TM=64, TN=_TN
                    )
            return (Q.reshape(*batch_shape, m, n), R.reshape(*batch_shape, n, n))

        reg = None
        # the register tile is loaded straight from A with user strides; a
        # non-column-contiguous A turns that load into a full 2-D gather,
        # which the compiler buffers past the unified-buffer limit.  Such
        # inputs go through the staged (column-major copy-in) paths instead.
        if (
            A_view.stride(2) == 1
            and (out_Q is None or out_Q.dim() < 3 or out_Q.stride(2) == 1)
            and (out_R is None or out_R.stride(2) == 1)
        ):
            reg = _reg_tile_cfg(m, n, qcols)
        if reg is not None:
            RMv, TNv, TQv = reg
            R = (
                out_R
                if out_R is not None
                else torch.empty(B, rrows, n, dtype=A.dtype, device=A.device)
            )
            sRb, sRm, sRn = R.stride()
            if qcols:
                Q = (
                    out_Q
                    if out_Q is not None
                    else torch.empty(B, m, qcols, dtype=A.dtype, device=A.device)
                )
                sQb, sQm, sQn = Q.stride()
            else:
                if out_Q is not None:
                    Q = out_Q
                else:
                    Q = torch.empty((0,), dtype=A.dtype, device=A.device)
                sQb, sQm, sQn = 0, 0, 0
            ldv = 128
            Vc = torch.empty(B, k, ldv, dtype=A.dtype, device=A.device)
            tau = torch.empty(B, k, dtype=A.dtype, device=A.device)
            for b0, bn in _batch_chunks(B):
                As, Vs, Ts, Qs, Rs = _bslice((A_view, Vc, tau, Q, R), b0, bn)
                _qr_reg_kernel[(bn,)](
                    As,
                    Vs,
                    Qs,
                    Rs,
                    Ts,
                    m,
                    n,
                    k,
                    qcols,
                    rrows,
                    ldv,
                    sAb,
                    sAm,
                    sAn,
                    k * ldv,
                    sQb,
                    sQm,
                    sQn,
                    sRb,
                    sRm,
                    sRn,
                    RM=RMv,
                    TN=TNv,
                    TQ=TQv,
                    PUT_Q=bool(qcols),
                )
            if mode == "reduced":
                return (Q.reshape(*batch_shape, m, k), R.reshape(*batch_shape, k, n))
            if mode == "r":
                return (Q, R.reshape(*batch_shape, k, n))
            return (Q.reshape(*batch_shape, m, m), R.reshape(*batch_shape, m, n))

        bands = (m + 63) // 64
        use_mcta = 2 <= bands <= 16 and B * bands <= _MAX_BLOCKS
        nc = bands  # one band per CTA
        mnc = nc  # scratch slot stride factor
        extra = B * (2 * k * nc + 64 * nc * 64) if use_mcta else 0
        ws = torch.empty(
            B * (n * ld + k * ld + k + qcols * ld) + extra,
            dtype=A.dtype,
            device=A.device,
        )
        Wc = ws[: B * n * ld].view(B, n, ld)
        Vc = ws[B * n * ld : B * (n + k) * ld].view(B, k, ld)
        tau = ws[B * (n + k) * ld : B * (n + k) * ld + B * k].view(B, k)
        Qc = ws[B * (n + k) * ld + B * k : B * ((n + k) * ld + k + qcols * ld)].view(
            B, qcols, ld
        )
        if use_mcta:
            oX = B * ((n + k) * ld + k + qcols * ld)
            abuf = ws[oX : oX + B * k * mnc].view(B, k, mnc)
            xbuf = ws[oX + B * k * mnc : oX + B * 2 * k * mnc].view(B, k, mnc)
            wbuf = ws[
                oX + B * 2 * k * mnc : oX + B * (2 * k * mnc + 64 * mnc * 64)
            ].view(B, 64 * mnc, 64)
            # fresh zeroed barrier counters per call (start at 0); the host
            # mirrors the running total in cbase across the panel loop
            ctr = torch.zeros(B, dtype=torch.int32, device=A.device)
            cbase = 0

        sWb = n * ld
        sVb = k * ld
        sTauB, sTauN = tau.stride()
        sQcb = qcols * ld

        for b0, bn in _batch_chunks(B):
            As, Ws, Vs, Ts = _bslice((A_view, Wc, Vc, tau), b0, bn)
            _copy_a_to_wc[(bn, n, _cpy_tiles(n, m, _RM))](
                As, Ws, m, n, ld, sAb, sAm, sAn, sWb, RM=_RM
            )
            for j0 in range(0, k, _NB):
                nb = min(_NB, k - j0)
                if use_mcta:
                    As_, Ws_, Vs_, Ts_, abs_, xbs_, wbs_, cs_ = _bslice(
                        (A_view, Wc, Vc, tau, abuf, xbuf, wbuf, ctr), b0, bn
                    )
                    _panel_mcta_kernel[(bn, nc)](
                        Ws_,
                        Vs_,
                        Ts_,
                        abs_,
                        xbs_,
                        wbs_,
                        cs_,
                        m,
                        j0,
                        nb,
                        cbase,
                        nc,
                        ld,
                        sWb,
                        sVb,
                        sTauB,
                        sTauN,
                        RM=64,
                        PW=64,
                    )
                    cbase += 3 * nb * nc
                else:
                    _panel_kernel[(bn,)](
                        Ws,
                        Vs,
                        Ts,
                        Ws,
                        m,
                        j0,
                        nb,
                        ld,
                        sWb,
                        sVb,
                        sTauB,
                        sTauN,
                        0,
                        0,
                        0,
                        RM=_RM,
                        PW=_tile_width(nb),
                        WRITE_R=False,
                    )
                c0 = j0 + nb
                p = n - c0
                if p > 0:
                    pw_t = _par_tile_width(p, bn)
                    for u0, un in _axis1_chunks(_grid_elem(p, pw_t), bn):
                        _trailing_apply_kernel[(bn, un)](
                            Vs,
                            Ts,
                            Ws,
                            m,
                            j0,
                            nb,
                            c0,
                            p,
                            u0,
                            ld,
                            sVb,
                            sTauB,
                            sTauN,
                            sWb,
                            RM=_RM,
                            PW=pw_t,
                        )

    if mode == "r":
        R = (
            out_R
            if out_R is not None
            else torch.empty(B, k, n, dtype=A.dtype, device=A.device)
        )
        sRb, sRm, sRn = R.stride()
        with torch_device_fn.device(A.device):
            for b0, bn in _batch_chunks(B):
                Ws, Rs = _bslice((Wc, R), b0, bn)
                if sRn == 1:
                    _triu_copy_cm[(bn, _grid_elem(k, 64), _grid_elem(n, _TN))](
                        Ws, Rs, k, n, ld, sWb, sRb, sRm, sRn, TM=64, TN=_TN
                    )
                else:
                    _triu_copy_scalar[(bn,)](Ws, Rs, k, n, ld, sWb, sRb, sRm, sRn)
        Q = (
            out_Q
            if out_Q is not None
            else torch.empty((0,), dtype=A.dtype, device=A.device)
        )
        return Q, R.reshape(*batch_shape, k, n)

    # Q assembly: build Q in padded column-major space in a single launch,
    # then copy out to the caller's layout.  The vectorised copy needs a
    # contiguous last output dimension; strided views fall back to the scalar
    # kernel (vector stores on arbitrary user strides miscompile).
    Q = (
        out_Q
        if out_Q is not None
        else torch.empty(B, m, qcols, dtype=A.dtype, device=A.device)
    )
    sQb, sQm, sQn = Q.stride()

    with torch_device_fn.device(A.device):
        for b0, bn in _batch_chunks(B):
            Vs, Ts, Qcs, Qs = _bslice((Vc, tau, Qc, Q), b0, bn)
            pw_q = _par_tile_width(qcols, bn)
            for u0, un in _axis1_chunks(_grid_elem(qcols, pw_q), bn):
                _q_apply_kernel[(bn, un)](
                    Vs,
                    Ts,
                    Qcs,
                    m,
                    k,
                    qcols,
                    u0,
                    ld,
                    sVb,
                    sTauB,
                    sTauN,
                    sQcb,
                    1,
                    ld,
                    RM=_RM,
                    PW=pw_q,
                )
            if sQn == 1:
                _copy_qc_to_q[(bn, _grid_elem(m, 64), _grid_elem(qcols, _TN))](
                    Qcs, Qs, m, qcols, ld, sQcb, sQb, sQm, sQn, TM=64, TN=_TN
                )
            else:
                _copy_qc_to_q_scalar[(bn,)](Qcs, Qs, m, qcols, ld, sQcb, sQb, sQm, sQn)

    R = (
        out_R
        if out_R is not None
        else torch.empty(B, rrows, n, dtype=A.dtype, device=A.device)
    )
    sRb, sRm, sRn = R.stride()
    with torch_device_fn.device(A.device):
        for b0, bn in _batch_chunks(B):
            Ws, Rs = _bslice((Wc, R), b0, bn)
            if sRn == 1:
                _triu_copy_cm[(bn, _grid_elem(rrows, 64), _grid_elem(n, _TN))](
                    Ws, Rs, rrows, n, ld, sWb, sRb, sRm, sRn, TM=64, TN=_TN
                )
            else:
                _triu_copy_scalar[(bn,)](Ws, Rs, rrows, n, ld, sWb, sRb, sRm, sRn)

    if mode == "reduced":
        return (Q.reshape(*batch_shape, m, k), R.reshape(*batch_shape, k, n))
    return (Q.reshape(*batch_shape, m, m), R.reshape(*batch_shape, m, n))


def _linalg_qr(A, mode="reduced", *, out=None):
    _validate_mode(mode)
    if A.dim() < 2:
        raise RuntimeError("linalg_qr: input must have at least 2 dimensions")
    if A.dtype not in (torch.float32, torch.float64):
        raise NotImplementedError(
            "FlagGems linalg_qr currently supports float32 and float64 inputs; "
            f"got dtype={A.dtype}"
        )
    return _ascend_linalg_qr(A, mode, out=out)


def linalg_qr(A, mode="reduced", *, out=None):
    logger.debug("GEMS_ASCEND LINALG_QR")
    return _linalg_qr(A, mode=mode, out=out)


def linalg_qr_out(A, mode="reduced", *, Q, R):
    logger.debug("GEMS_ASCEND LINALG_QR_OUT")
    return _linalg_qr(A, mode, out=(Q, R))

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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device as runtime_device
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_DET_PACK_BUDGET = 2048
_DET_PACK_MAX = 64
# When n falls in (16, 32], each matrix alone occupies 1K lanes; relaxing the budget is needed to fill the batch across
# 48 CTAs
# (measured: 512 matrices of 32x32 with PACK=4 are 1.3x faster than PACK=2); if the batch is too small,
# packing instead leaves too few CTAs, hence the threshold.
_DET_PACK_BUDGET_WIDE = 4096
_DET_PACK_MIN_BATCH_WIDE = 64
# For large batches with n <= 16, one more step: at batch=1024, n=16, PACK 8→16 reduces 2.67 waves to
# 1.33 waves (measured +9%); but halving the CTA count is a net loss for small batches (measured -10% at batch=128),
# and the (16,32] band for 32x32 falls back similarly at PACK=8, so relaxation is limited to this narrow band.
_DET_PACK_BUDGET_BIG = 4096
_DET_PACK_MIN_BATCH_BIG = 1024


@triton.jit
def _reduce_mul(a, b):
    return a * b


@libentry()
@triton.jit
def _det_blocked_kernel(
    A,
    out,
    N,
    BLOCK: tl.constexpr,
):
    pid = tle.program_id(0)
    base = pid * N * N
    swap_count = tl.zeros((), dtype=tl.int32)

    for k in range(N):
        best_val = tl.full((), -1.0, dtype=A.dtype.element_ty)
        best_row = tl.full((), k, dtype=tl.int32)
        for i0 in range(k, N, BLOCK):
            rows = i0 + tl.arange(0, BLOCK)
            col = tl.load(A + base + rows * N + k, mask=rows < N, other=0.0)
            abs_col = tl.where((rows >= k) & (rows < N), tl.abs(col), -1.0)
            tile_max = tl.max(abs_col, axis=0)
            tile_row = tl.min(tl.where(abs_col == tile_max, rows, N), axis=0)
            is_better = tile_max > best_val
            best_row = tl.where(is_better, tile_row, best_row)
            best_val = tl.where(is_better, tile_max, best_val)

        for j0 in range(k, N, BLOCK):
            cols = j0 + tl.arange(0, BLOCK)
            cmask = cols < N
            row_k = tl.load(A + base + k * N + cols, mask=cmask, other=0.0)
            row_p = tl.load(A + base + best_row * N + cols, mask=cmask, other=0.0)
            tl.store(A + base + k * N + cols, row_p, mask=cmask)
            tl.store(A + base + best_row * N + cols, row_k, mask=cmask)
        swap_count = tl.where(best_row != k, swap_count + 1, swap_count)

        tl.debug_barrier()

        pivot = tl.load(A + base + k * N + k)
        safe_pivot = tl.where(pivot == 0.0, 1.0, pivot)

        for i0 in range(k + 1, N, BLOCK):
            rows = i0 + tl.arange(0, BLOCK)
            rmask = rows < N
            col = tl.load(A + base + rows * N + k, mask=rmask, other=0.0)
            tl.store(A + base + rows * N + k, col / safe_pivot, mask=rmask)

        tl.debug_barrier()

        for i0 in range(k + 1, N, BLOCK):
            rows = i0 + tl.arange(0, BLOCK)
            rmask = rows < N
            l_col = tl.load(A + base + rows * N + k, mask=rmask, other=0.0)
            for j0 in range(k + 1, N, BLOCK):
                cols = j0 + tl.arange(0, BLOCK)
                cmask = cols < N
                u_row = tl.load(A + base + k * N + cols, mask=cmask, other=0.0)
                tmask = rmask[:, None] & cmask[None, :]
                tile = tl.load(
                    A + base + rows[:, None] * N + cols[None, :], mask=tmask, other=0.0
                )
                tile = tile - l_col[:, None] * u_row[None, :]
                tl.store(A + base + rows[:, None] * N + cols[None, :], tile, mask=tmask)

        tl.debug_barrier()

    det = tl.full((), 1.0, dtype=A.dtype.element_ty)
    for i0 in range(0, N, BLOCK):
        d = i0 + tl.arange(0, BLOCK)
        diag = tl.load(A + base + d * N + d, mask=d < N, other=1.0)
        det = det * tl.reduce(diag, 0, combine_fn=_reduce_mul)
    det = tl.where(swap_count % 2 == 0, det, -det)
    tl.store(out + pid, det)


@libentry()
@triton.jit
def _det_panel_kernel(
    A,
    out,
    N,
    PANEL: tl.constexpr,
    BLOCK: tl.constexpr,
    SINGLE: tl.constexpr,
    ONE_PANEL: tl.constexpr,
):
    pid = tle.program_id(0)
    base = pid * N * N
    swap_count = tl.zeros((), dtype=tl.int32)

    if ONE_PANEL:
        # Det-only single-panel fast path (host guarantees N <= PANEL and
        # BLOCK >= N): the panel loop has exactly one iteration and every
        # inner chunk loop covers its whole range in one shot, so the
        # elimination step flattens to straight-line code (each range()
        # costs an integer division on this backend). There is no trailing
        # submatrix and this kernel exports nothing, so the scaled
        # multiplier column is never read back: it stays in registers,
        # dropping the per-step strided scatter of the L factors and the
        # barrier that only ordered its reload.
        for k in range(0, N):
            rows = k + tl.arange(0, BLOCK)
            # The load pads rows >= N with 0 and abs(0) = 0 never wins the
            # max against a nonzero pivot column; an all-zero (singular)
            # column falls through to the best_val <= 0 guard below.
            col = tl.load(A + base + rows * N + k, mask=rows < N, other=0.0)
            abs_col = tl.abs(col)
            best_val = tl.max(abs_col, axis=0)
            best_row = tl.min(tl.where(abs_col == best_val, rows, N), axis=0)
            best_row = tl.where(best_val <= 0.0, k, best_row)

            cols = tl.arange(0, BLOCK)
            cmask = cols < N
            row_k = tl.load(A + base + k * N + cols, mask=cmask, other=0.0)
            row_p = tl.load(A + base + best_row * N + cols, mask=cmask, other=0.0)
            tl.store(A + base + k * N + cols, row_p, mask=cmask)
            tl.store(A + base + best_row * N + cols, row_k, mask=cmask)
            swap_count = tl.where(best_row != k, swap_count + 1, swap_count)

            tl.debug_barrier()

            pivot = tl.load(A + base + k * N + k)
            safe_pivot = tl.where(pivot == 0.0, 1.0, pivot)
            rows = k + 1 + tl.arange(0, BLOCK)
            rmask = rows < N
            l_col = tl.load(A + base + rows * N + k, mask=rmask, other=0.0)
            l_col = l_col / safe_pivot

            tl.debug_barrier()

            pcols = k + 1 + tl.arange(0, PANEL)
            pcmask = pcols < N
            u_row = tl.load(A + base + k * N + pcols, mask=pcmask, other=0.0)
            tmask = rmask[:, None] & pcmask[None, :]
            tile = tl.load(
                A + base + rows[:, None] * N + pcols[None, :], mask=tmask, other=0.0
            )
            tile = tile - l_col[:, None] * u_row[None, :]
            tl.store(A + base + rows[:, None] * N + pcols[None, :], tile, mask=tmask)

            tl.debug_barrier()
    else:
        for k0 in range(0, N, PANEL):
            kend = tl.minimum(k0 + PANEL, N)

            if SINGLE:
                # Single-block path (host guarantees BLOCK >= N): every
                # inner chunk loop of the elimination step has exactly one
                # iteration. Flatten the step and reuse the scaled
                # multiplier column from registers; the reload was the only
                # read of column k between the scaling store and the rank-1
                # update, so the barrier separating them goes away too (the
                # u-row/tile reads touch the swap-written rows, already
                # ordered by the post-swap barrier).
                for k in range(k0, kend):
                    rows = k + tl.arange(0, BLOCK)
                    col = tl.load(A + base + rows * N + k, mask=rows < N, other=0.0)
                    abs_col = tl.abs(col)
                    best_val = tl.max(abs_col, axis=0)
                    best_row = tl.min(tl.where(abs_col == best_val, rows, N), axis=0)
                    best_row = tl.where(best_val <= 0.0, k, best_row)

                    cols = k0 + tl.arange(0, BLOCK)
                    cmask = cols < N
                    row_k = tl.load(A + base + k * N + cols, mask=cmask, other=0.0)
                    row_p = tl.load(
                        A + base + best_row * N + cols, mask=cmask, other=0.0
                    )
                    tl.store(A + base + k * N + cols, row_p, mask=cmask)
                    tl.store(A + base + best_row * N + cols, row_k, mask=cmask)
                    swap_count = tl.where(best_row != k, swap_count + 1, swap_count)

                    tl.debug_barrier()

                    pivot = tl.load(A + base + k * N + k)
                    safe_pivot = tl.where(pivot == 0.0, 1.0, pivot)
                    rows = k + 1 + tl.arange(0, BLOCK)
                    rmask = rows < N
                    l_col = tl.load(A + base + rows * N + k, mask=rmask, other=0.0)
                    l_col = l_col / safe_pivot
                    # The scaled L factors are re-read by the trailing
                    # strip/dot phases only while this panel leaves a
                    # trailing submatrix (kend < N); on the last panel the
                    # store is dead work on the serial path.
                    if kend < N:
                        tl.store(A + base + rows * N + k, l_col, mask=rmask)

                    pcols = k + 1 + tl.arange(0, PANEL)
                    pcmask = pcols < kend
                    u_row = tl.load(A + base + k * N + pcols, mask=pcmask, other=0.0)
                    tmask = rmask[:, None] & pcmask[None, :]
                    tile = tl.load(
                        A + base + rows[:, None] * N + pcols[None, :],
                        mask=tmask,
                        other=0.0,
                    )
                    tile = tile - l_col[:, None] * u_row[None, :]
                    tl.store(
                        A + base + rows[:, None] * N + pcols[None, :], tile, mask=tmask
                    )

                    tl.debug_barrier()
            else:
                for k in range(k0, kend):
                    best_val = tl.full((), -1.0, dtype=A.dtype.element_ty)
                    best_row = tl.full((), k, dtype=tl.int32)
                    for i0 in range(k, N, BLOCK):
                        rows = i0 + tl.arange(0, BLOCK)
                        col = tl.load(A + base + rows * N + k, mask=rows < N, other=0.0)
                        abs_col = tl.where((rows >= k) & (rows < N), tl.abs(col), -1.0)
                        tile_max = tl.max(abs_col, axis=0)
                        tile_row = tl.min(
                            tl.where(abs_col == tile_max, rows, N), axis=0
                        )
                        is_better = tile_max > best_val
                        best_row = tl.where(is_better, tile_row, best_row)
                        best_val = tl.where(is_better, tile_max, best_val)
                    best_row = tl.where(best_val <= 0.0, k, best_row)

                    for j0 in range(k0, N, BLOCK):
                        cols = j0 + tl.arange(0, BLOCK)
                        cmask = cols < N
                        row_k = tl.load(A + base + k * N + cols, mask=cmask, other=0.0)
                        row_p = tl.load(
                            A + base + best_row * N + cols, mask=cmask, other=0.0
                        )
                        tl.store(A + base + k * N + cols, row_p, mask=cmask)
                        tl.store(A + base + best_row * N + cols, row_k, mask=cmask)
                    swap_count = tl.where(best_row != k, swap_count + 1, swap_count)

                    tl.debug_barrier()

                    pivot = tl.load(A + base + k * N + k)
                    safe_pivot = tl.where(pivot == 0.0, 1.0, pivot)
                    for i0 in range(k + 1, N, BLOCK):
                        rows = i0 + tl.arange(0, BLOCK)
                        rmask = rows < N
                        col = tl.load(A + base + rows * N + k, mask=rmask, other=0.0)
                        tl.store(A + base + rows * N + k, col / safe_pivot, mask=rmask)

                    tl.debug_barrier()

                    pcols = k + 1 + tl.arange(0, PANEL)
                    pcmask = pcols < kend
                    u_row = tl.load(A + base + k * N + pcols, mask=pcmask, other=0.0)
                    for i0 in range(k + 1, N, BLOCK):
                        rows = i0 + tl.arange(0, BLOCK)
                        rmask = rows < N
                        l_col = tl.load(A + base + rows * N + k, mask=rmask, other=0.0)
                        tmask = rmask[:, None] & pcmask[None, :]
                        tile = tl.load(
                            A + base + rows[:, None] * N + pcols[None, :],
                            mask=tmask,
                            other=0.0,
                        )
                        tile = tile - l_col[:, None] * u_row[None, :]
                        tl.store(
                            A + base + rows[:, None] * N + pcols[None, :],
                            tile,
                            mask=tmask,
                        )

                    tl.debug_barrier()

            # When the panel reaches the end of the matrix (kend == N) there
            # is no trailing submatrix and every j-loop of the strip phase is
            # empty, leaving only a strip load + barrier per column. Skip the
            # dead phase entirely: for the last panel this removes up to
            # PANEL dead barrier iterations from the serial path.
            if kend < N:
                for c in range(k0, kend):
                    prows = c + 1 + tl.arange(0, PANEL)
                    prmask = prows < kend
                    l_strip = tl.load(A + base + prows * N + c, mask=prmask, other=0.0)
                    for j0 in range(kend, N, BLOCK):
                        cols = j0 + tl.arange(0, BLOCK)
                        cmask = cols < N
                        u_strip = tl.load(
                            A + base + c * N + cols, mask=cmask, other=0.0
                        )
                        tmask = prmask[:, None] & cmask[None, :]
                        tile = tl.load(
                            A + base + prows[:, None] * N + cols[None, :],
                            mask=tmask,
                            other=0.0,
                        )
                        tile = tile - l_strip[:, None] * u_strip[None, :]
                        tl.store(
                            A + base + prows[:, None] * N + cols[None, :],
                            tile,
                            mask=tmask,
                        )

                    tl.debug_barrier()

            pcols = k0 + tl.arange(0, PANEL)
            pmask = pcols < kend
            for i0 in range(kend, N, BLOCK):
                rows = i0 + tl.arange(0, BLOCK)
                rmask = rows < N
                l_tile = tl.load(
                    A + base + rows[:, None] * N + pcols[None, :],
                    mask=rmask[:, None] & pmask[None, :],
                    other=0.0,
                )
                for j0 in range(kend, N, BLOCK):
                    cols = j0 + tl.arange(0, BLOCK)
                    cmask = cols < N
                    u_tile = tl.load(
                        A + base + pcols[:, None] * N + cols[None, :],
                        mask=pmask[:, None] & cmask[None, :],
                        other=0.0,
                    )
                    tmask = rmask[:, None] & cmask[None, :]
                    tile = tl.load(
                        A + base + rows[:, None] * N + cols[None, :],
                        mask=tmask,
                        other=0.0,
                    )
                    tile = tile - tl.dot(l_tile, u_tile, input_precision="ieee")
                    tl.store(
                        A + base + rows[:, None] * N + cols[None, :], tile, mask=tmask
                    )

            tl.debug_barrier()

    det = tl.full((), 1.0, dtype=A.dtype.element_ty)
    for i0 in range(0, N, BLOCK):
        d = i0 + tl.arange(0, BLOCK)
        diag = tl.load(A + base + d * N + d, mask=d < N, other=1.0)
        det = det * tl.reduce(diag, 0, combine_fn=_reduce_mul)
    det = tl.where(swap_count % 2 == 0, det, -det)
    tl.store(out + pid, det)


# GCU400 fast register kernel for float32 det (n <= 16 and 32 < n <= 64).
#
# Measured on L600, a register-tile where+sum extraction costs ~1.6-5us (DSM
# round trip) while a 1D reduction over a [<=256] vector costs ~0.3-0.5us.
# The upstream register kernel spends 7 tile reductions per elimination step
# (column extract x2, pivot row, current row, plus 1D max/argmax/pivot) and
# pads tiny matrices to BLOCK_N=16, which dominates its runtime here. This
# variant keeps only 3 tile extractions per step (column, pivot row, current
# row), rebuilds the post-swap column by vector arithmetic instead of
# re-extraction, applies the rank-1 update unmasked (multipliers are zeroed
# above the diagonal row and the u-row is zeroed at columns <= k, so the FMA
# is a no-op outside the active submatrix), and pads only to BLOCK_N=8.
# Kept in registers end to end, so batch dispatch scales the same as upstream.
# Dispatch: n <= 16 and 32 < n <= 64 use this kernel (upstream wins at
# n in (16, 32]); n > 64 keeps the upstream panel kernel with a wider
# PANEL/BLOCK config (see _linalg_det_impl).


@libentry()
@triton.jit
def _det_register2_kernel(A, out, N, BLOCK_N: tl.constexpr):
    pid = tle.program_id(0)
    rows = tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_N)
    offsets = pid * N * N + rows[:, None] * N + cols[None, :]
    load_mask = (rows[:, None] < N) & (cols[None, :] < N)
    work = tl.load(A + offsets, mask=load_mask, other=0.0)
    swap_parity = tl.zeros((), dtype=tl.int32)
    for k in range(N):
        col_k = tl.sum(tl.where(cols[None, :] == k, work, 0.0), axis=1)
        abs_col = tl.abs(col_k)
        abs_col = tl.where((rows < k) | (rows >= N), -1.0, abs_col)
        pivot_val = tl.max(abs_col, axis=0)
        pivot_row = tl.min(tl.where(abs_col == pivot_val, rows, BLOCK_N), axis=0)
        row_k = tl.sum(tl.where(rows[:, None] == k, work, 0.0), axis=0)
        row_p = tl.sum(tl.where(rows[:, None] == pivot_row, work, 0.0), axis=0)
        work = tl.where(rows[:, None] == k, row_p[None, :], work)
        work = tl.where(rows[:, None] == pivot_row, row_k[None, :], work)
        swap_parity = swap_parity + (pivot_row != k).to(tl.int32)
        ck = tl.sum(tl.where(rows == k, col_k, 0.0), axis=0)
        pv = tl.sum(tl.where(rows == pivot_row, col_k, 0.0), axis=0)
        col_new = tl.where(rows == k, pv, tl.where(rows == pivot_row, ck, col_k))
        safe = tl.where(pv == 0.0, 1.0, pv)
        inv = 1.0 / safe
        mult = tl.where(rows > k, col_new * inv, 0.0)
        mult = tl.where(rows < N, mult, 0.0)
        u_row = tl.where(cols > k, row_p, 0.0)
        work = work - mult[:, None] * u_row[None, :]
    diag = tl.sum(tl.where(rows[:, None] == cols[None, :], work, 0.0), axis=0)
    diag = tl.where(cols < N, diag, 1.0)
    det = tl.reduce(diag, 0, combine_fn=_reduce_mul)
    det = tl.where(swap_parity % 2 == 0, det, -det)
    tl.store(out + pid, det)


@libentry()
@triton.jit
def _det_pack_kernel(A, out, BATCH, N, BLOCK_N: tl.constexpr, PACK: tl.constexpr):
    """Handles PACK n x n matrices per call (n <= BLOCK_N), packing the batch axis into the CTA.

    A small-matrix LU is only O(n^3) of very short work, while CTA launch/drain is fixed
    overhead, so one CTA per matrix makes a 4096-case of 4x4 blocks limited by CTA count
    rather than compute. Packing PACK matrices into a 3-D block [PACK, BLOCK_N, BLOCK_N] of
    one CTA raises matrix- and instruction-level parallelism, cutting the CTA count by PACK.
    """
    pid = tle.program_id(0)
    rows = tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_N)
    bids = pid * PACK + tl.arange(0, PACK)
    bmask = bids < BATCH

    offsets = (
        bids[:, None, None] * N * N + rows[None, :, None] * N + cols[None, None, :]
    )
    load_mask = (
        bmask[:, None, None] & (rows[None, :, None] < N) & (cols[None, None, :] < N)
    )
    work = tl.load(A + offsets, mask=load_mask, other=0.0)

    # Explicitly broadcast to [PACK, BLOCK_N] to avoid relying on Triton's implicit broadcasting of [1, BN] and [PACK,
    # 1]
    row_idx = tl.zeros((PACK, BLOCK_N), dtype=tl.int32) + rows[None, :]
    parity = tl.zeros((PACK,), dtype=tl.int32)

    for k in range(N):
        # The k-th column (one [BLOCK_N] vector per matrix)
        col_k = tl.sum(tl.where(cols[None, None, :] == k, work, 0.0), axis=2)
        abs_col = tl.abs(col_k)
        abs_col = tl.where((row_idx < k) | (row_idx >= N), -1.0, abs_col)
        pivot_val = tl.max(abs_col, axis=1)
        pivot_row = tl.min(
            tl.where(abs_col == pivot_val[:, None], row_idx, BLOCK_N), axis=1
        )

        row_k = tl.sum(tl.where(row_idx[:, :, None] == k, work, 0.0), axis=1)
        row_p = tl.sum(
            tl.where(row_idx[:, :, None] == pivot_row[:, None, None], work, 0.0), axis=1
        )
        work = tl.where(row_idx[:, :, None] == k, row_p[:, None, :], work)
        work = tl.where(
            row_idx[:, :, None] == pivot_row[:, None, None], row_k[:, None, :], work
        )
        parity = parity + (pivot_row != k).to(tl.int32)

        # Reconstruct the k-th column after the swap with vector arithmetic, avoiding another full-tile reduction
        ck = tl.sum(tl.where(row_idx == k, col_k, 0.0), axis=1)
        pv = tl.sum(tl.where(row_idx == pivot_row[:, None], col_k, 0.0), axis=1)
        col_new = tl.where(
            row_idx == k,
            pv[:, None],
            tl.where(row_idx == pivot_row[:, None], ck[:, None], col_k),
        )
        safe = tl.where(pv == 0.0, 1.0, pv)
        inv = 1.0 / safe
        mult = tl.where(row_idx > k, col_new * inv[:, None], 0.0)
        mult = tl.where(row_idx < N, mult, 0.0)
        u_row = tl.where(cols[None, :] > k, row_p, 0.0)
        work = work - mult[:, :, None] * u_row[:, None, :]

    diag = tl.sum(
        tl.where(row_idx[:, :, None] == cols[None, None, :], work, 0.0), axis=1
    )
    diag = tl.where(cols[None, :] < N, diag, 1.0)
    det = tl.reduce(diag, 1, combine_fn=_reduce_mul)
    det = tl.where(parity % 2 == 0, det, -det)
    tl.store(out + bids, det, mask=bmask)


def linalg_det(A):
    logger.debug("GEMS_ENFLAME LINALG_DET")
    return _linalg_det_impl(A)


def linalg_det_out(A, *, out=None):
    logger.debug("GEMS_ENFLAME LINALG_DET_OUT")
    if out is None:
        raise TypeError("linalg_det(): out must be provided for out variant")
    if out.dtype != A.dtype:
        raise RuntimeError(
            f"linalg_det: dtype of out ({out.dtype}) does not match "
            f"dtype of input ({A.dtype})"
        )
    if out.device != A.device:
        raise RuntimeError(
            f"linalg_det: device of out ({out.device}) does not match "
            f"device of input ({A.device})"
        )
    if out.shape != A.shape[:-2]:
        raise RuntimeError(
            f"linalg_det: shape of out {tuple(out.shape)} does not match "
            f"expected shape {tuple(A.shape[:-2])}"
        )
    out.copy_(_linalg_det_impl(A))
    return out


def _linalg_det_impl(A):
    if A.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"linalg_det only supports float32 and float64, got {A.dtype}")

    if A.dim() < 2:
        raise ValueError(
            f"linalg_det: input tensor must be at least 2D, got {A.dim()}D"
        )

    m, n = A.shape[-2], A.shape[-1]
    if m != n:
        raise ValueError(
            f"linalg_det: input tensor must be a square matrix, got {m}x{n}"
        )

    batch_shape = A.shape[:-2]
    if n == 0:
        return torch.ones(batch_shape, dtype=A.dtype, device=A.device)

    batch_count = math.prod(batch_shape)
    if batch_count == 0:
        return torch.empty(batch_shape, dtype=A.dtype, device=A.device)

    is_f32 = A.dtype == torch.float32
    use_pack = is_f32 and (
        (n <= 16 and batch_count > 1)
        or (16 < n <= 32 and batch_count >= _DET_PACK_MIN_BATCH_WIDE)
    )
    use_reg2 = is_f32 and not use_pack and n <= 16

    # The register/pack kernels never write to the working matrix, so a
    # contiguous input is passed directly as a view (no scratch copy); the
    # blocked/panel kernels eliminate in place and need the private clone.
    # Skipping the clone on the read-only bands also removes an extra aten
    # copy-kernel launch from the op latency.
    if (use_pack or use_reg2) and A.is_contiguous():
        A_work = A.reshape(batch_count, n, n)
    else:
        A_work = A.clone(memory_format=torch.contiguous_format).reshape(
            batch_count, n, n
        )
    out = torch.empty(batch_count, dtype=A.dtype, device=A.device)

    grid = (batch_count,)
    with torch_device_fn.device(A.device):
        if use_pack:
            # Small-matrix batches: pack PACK matrices per CTA within a
            # ~2K-4K element lane budget so the CTA count drops by PACK.
            block_n = max(4, triton.next_power_of_2(n))
            if n <= 16 and batch_count >= _DET_PACK_MIN_BATCH_BIG:
                budget = _DET_PACK_BUDGET_BIG
            else:
                budget = _DET_PACK_BUDGET if n <= 16 else _DET_PACK_BUDGET_WIDE
            pack = max(1, min(_DET_PACK_MAX, budget // (block_n * block_n)))
            # tl.arange needs a power of two; the tail is masked by bmask.
            pack = min(pack, 1 << (batch_count - 1).bit_length())
            num_warps = min(8, max(1, (pack * block_n * block_n) // 512))
            grid = (triton.cdiv(batch_count, pack),)
            _det_pack_kernel[grid](
                A_work,
                out,
                batch_count,
                n,
                BLOCK_N=block_n,
                PACK=pack,
                num_warps=num_warps,
            )
            return out.reshape(batch_shape)

        if use_reg2:
            # GCU400 fast register path (n <= 16): 3 tile extractions per
            # elimination step with the post-swap column rebuilt by vector
            # arithmetic; see the kernel comment above.
            block_n = max(8, triton.next_power_of_2(n))
            num_warps = min(8, max(1, (block_n * block_n * 4) // 4096))
            _det_register2_kernel[grid](
                A_work, out, n, BLOCK_N=block_n, num_warps=num_warps
            )
            return out.reshape(batch_shape)

        # LU shape tuning is dtype dependent. A float64 element is twice as
        # wide and the per-element arithmetic intensity of the trailing
        # update is higher, so the panel kernel wants a narrower
        # panel/block than float32: measured on H20, PANEL=16/BLOCK=32
        # beats PANEL=32/BLOCK=64 by 1.1-1.4x for float64 at n >= 64 and is
        # the better choice for 32 < n <= 64 as well. float32 instead uses
        # PANEL=128 with BLOCK equal to the matrix size (<=256), which
        # amortizes the per-op round-trip tax of the serial panel phase
        # (measured 1.5-1.8x over upstream PANEL=32/BLOCK=64 for
        # n in (32, 256]) and enables the single-block flattening below.
        # The metax float64 special case is preserved.
        is_f64 = A.dtype == torch.float64
        use_panel = (n > 32 if is_f64 else n > 16) and not (
            is_f64 and runtime_device.vendor_name == "metax"
        )
        if use_panel:
            if is_f64:
                # f64 panels are always multi-block here (n > 32 > BLOCK=32).
                panel_kwargs = {
                    "PANEL": 16,
                    "BLOCK": 32,
                    "SINGLE": 0,
                    "ONE_PANEL": 0,
                    "num_warps": 4,
                }
            else:
                # GCU400 f32: the single-block panel path (plain 1D loads +
                # fixed-cost 2D tile ops, no cross-lane tile reductions)
                # measures ~5us/step vs ~8.3us/step for the register-tile
                # register2 kernel, so it also serves (16, 64] (register2
                # keeps n <= 16, where its in-register extraction chains
                # win). n <= 128 puts the whole matrix in one panel: the
                # ONE_PANEL arm drops the dead L-factor stores and the
                # trailing phases outright.
                block_val = min(256, triton.next_power_of_2(max(n, 32)))
                single = 1 if n <= block_val else 0
                # Narrow the rank-1 tile to the matrix width inside the
                # one-panel band (n <= 128): the masked-off lanes of a
                # BLOCK/PANEL tile are pure fixed-cost tile-op waste
                # (measured +4-8% at n in (16, 32] shrinking BLOCK 64->32,
                # on top of the +5-6% from narrowing PANEL at n <= 64).
                panel_val = 128 if n > 128 else triton.next_power_of_2(max(n, 16))
                panel_kwargs = {
                    "PANEL": panel_val,
                    "BLOCK": block_val,
                    "SINGLE": single,
                    "ONE_PANEL": 1 if (single == 1 and n <= panel_val) else 0,
                    # Triton GCU 4.1.6 (2026-11-17 stack): nw1 wins the whole
                    # one-panel band -- the intra-warp reduction tree of the
                    # per-step pivot max/argmin and the single-warp tile ops
                    # beat every cross-warp shape (measured n=128: -30% vs
                    # nw4, n=64: -10% vs nw2, n=32: -11% vs nw2; nw8/nw16
                    # measure flat, silently downshifted). n > 128 stays at
                    # nw4: the SINGLE arm's [BLOCK=256, PANEL=128] tiles
                    # exceed the 917504B DSM budget at nw<4 (nw2 needs
                    # 1318912B). num_stages measures flat -- the per-step
                    # debug_barriers already prevent cross-step pipelining.
                    "num_warps": 1 if block_val <= 128 else 4,
                }
            _det_panel_kernel[grid](A_work, out, n, **panel_kwargs)
        else:
            block = min(64, max(8, triton.next_power_of_2(n)))
            num_warps = min(4, max(1, (block * block * 8) // 4096))
            _det_blocked_kernel[grid](A_work, out, n, BLOCK=block, num_warps=num_warps)

    return out.reshape(batch_shape)

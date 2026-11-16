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

"""Kunlunxin backend override for ``inverse`` (batched Gauss-Jordan on XPU3).

Why a dedicated implementation exists
-------------------------------------
The generic kernel (``flag_gems/ops/inverse.py``) keeps the working matrix and
the inverse accumulator as register-resident ``BLOCK x BLOCK`` tiles and
extracts rows / columns with ``tl.gather``.  TritonXPU cannot lower
``tt.gather`` at all (constant indices included), so every call fails in the
PassManager:

    failed to legalize operation 'tt.gather' that was explicitly marked illegal

The previous backend override delegated to ``torch.linalg.solve(A, I)``.  That
is a vendor/ATen compute fallback (red line: no CPU/ATen/native/composite
fallback), so this module replaces it with a backend-local Triton path built
from the patterns already proven on this stack by ``_kunlunxin/ops/
linalg_det.py``:

* elimination steps are separate launches (an in-kernel ``for k`` loop around a
  global store/load round trip is miscompiled by this backend unless the
  working tile is >= 1024 lanes and the batch is tiny);
* partial pivoting with plain ``tl.max`` + ``tl.min(where(...))`` -- never
  ``tl.argmax`` (unreliable on XPU3);
* every reduction runs over a load-rooted 1-D or ``[BB, ROWS]`` column tile,
  never over an arithmetic 2-D tile;
* the working matrix ``W`` and the inverse accumulator ``X`` live in padded
  power-of-two ``ROWS x AW`` buffers; the padding is filled with the identity
  so the padded system is block diagonal and the leading ``n x n`` block of
  ``X`` is the inverse.

Algorithm (Gauss-Jordan with partial pivoting on ``[W | X]``)::

    for k in 0 .. n-1:
        p = first argmax_{i >= k, i < n} |W[i, k]|
        swap rows k and p of W and X
        normalise row k of both by the swapped pivot
        for every row i != k:
            row_i -= W[i, k] * row_k           # both W and X
        pin W[:, k] to e_k                    # keeps the dead column exact

Dispatch of the step loop
-------------------------
Per-launch and per-program dispatch are the two costs on this backend
(~1.3 us per program).  Three regimes are used:

* ``tot <= _FUSED_MAX_TOT`` and a small batch: the whole step (pivot search,
  swap, normalisation, elimination) is fused into one program per matrix
  (``_inv_step_kernel``, the ``_det_step_kernel`` regime);
* several matrices share a program group (``_inv_scan_group_kernel`` +
  ``_inv_update_group_kernel``) so the per-program cost is amortised; the two
  work buffers are ping-ponged because a grouped update must not read a buffer
  it is rewriting;
* very large ``n``: a per-matrix pivot/swap/normalise launch plus a chunked
  eliminate launch (``_inv_pivot_swap_norm_kernel`` + ``_inv_elim_kernel``),
  where the post-swap elimination factors are passed through a side buffer so
  the eliminate launch never reads the column it zeroes.

The pack / unpadded-to-padded scatter and the inverse unpack are flat kernels
that span matrix boundaries, so a batch of thousands of tiny matrices costs a
handful of programs rather than one per matrix.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_MAX_BLK = 4096
# Default block width for the packed-tile kernels.  It must divide the padded
# tile exactly where divisibility is required (pack / eliminate); the unpack
# kernel clamps its tail index instead because masked stores are unreliable.
_BLK = 1024
# Both tiles (W and X) live in one program only while their combined footprint
# stays inside a single block, i.e. the ``_det_step_kernel`` regime.
_FUSED_MAX_TOT = _MAX_BLK // 2
# Grouped step path: lanes per program (two tiles, so ~2x this many values are
# live) and the smallest batch worth grouping.
_GROUP_LANES = 2048
_GROUP_MIN_BATCH = 16
# Small-iteration-tile in-kernel loop floor (``linalg_det._ELIM_MIN_TOT``).
_ELIM_MIN_TOT = 1024
# Batch bound for the single-launch in-kernel loop: padded small-n layouts are
# only reproducible for a handful of matrices (linalg_det's
# ``_ELIM_SMALL_MAX_BATCH == 4``).  Raised to 8 after a forced-path sweep
# (n in 2..32 x batch 1..16 x 3 seeds) measured worst abs error 8.9e-8 with
# zero failures, so the benchmark's batch=6 small matrices can use one launch
# instead of the 2-launches-per-step fused path.
_ELIM_MAX_BATCH = 8


def _plan(n):
    """Padded working layout for an ``n x n`` matrix.

    Mirrors ``_kunlunxin/ops/linalg_det._plan``: square rows for the sizes that
    fit one block, otherwise a 64-wide minimum row stride.  ``rows == aw == n``
    for the common powers of two, so no padding is introduced there.
    """
    rows = triton.next_power_of_2(n)
    # Narrow (unpadded) row strides are exact from 8 wide on; below that the
    # tiny flat tiles pick up the backend's small-tile miscompile (a batched
    # n = 3 matrix came out 3.9e-2 off), so those sizes keep a 64-wide stride.
    aw = rows if rows >= 8 else 64
    tot = rows * aw
    blk = min(_BLK, tot)
    return rows, aw, tot, blk, tot // blk


def _plan_pad(n):
    """Layout for the single-launch in-kernel Gauss-Jordan path.

    ``_inv_gj_kernel`` repeats an in-kernel elimination loop, which this
    backend only lowers exactly for a >= 1024-lane iteration tile (the same
    constraint as ``linalg_det._det_elim_kernel``), so the row stride is padded
    up until the tile reaches that width.
    """
    rows = triton.next_power_of_2(n)
    aw = max(64, rows)
    while rows * aw < _ELIM_MIN_TOT:
        aw *= 2
    tot = rows * aw
    return rows, aw, tot, tot, 1


def _pick_group(batch, tot):
    """Largest power-of-two group of matrices handled by one program."""
    bb = 1
    while bb * 2 * tot <= _GROUP_LANES and batch % (bb * 2) == 0 and bb * 2 <= batch:
        bb *= 2
    return bb


@libentry()
@triton.jit
def _inv_pack_kernel(
    A, W, X, N, AW: tl.constexpr, TOT: tl.constexpr, BLK: tl.constexpr
):
    """Scatter the contiguous input into padded ``W`` and initialise ``X=I``.

    ``BLK`` divides ``TOT`` exactly, so the tile arithmetic stays int32 and no
    masked store is needed.  Padding lanes are filled with the identity rather
    than left dangling: the padded system is then block diagonal and the
    leading ``n x n`` block of the accumulated inverse is exactly ``A^-1``.
    Out-of-range source indices are clamped and the loaded value discarded
    (``other=`` pollutes live lanes on this backend).
    """
    b = tle.program_id(0).to(tl.int64)
    # Keep the tile arithmetic in int32: int64 ``//``/``%`` is emulated and
    # dominated the whole inverse when the tile was sized to the padded buffer.
    blk = tl.program_id(1)
    e = blk * BLK + tl.arange(0, BLK)
    row = e // AW
    col = e % AW
    live = (row < N) & (col < N)
    idx = tl.where(live, row * N + col, 0)
    a = tl.load(A + b * N * N + idx)
    ident = (row == col).to(W.dtype.element_ty)
    tl.store(W + b * TOT + e, tl.where(live, a, ident))
    tl.store(X + b * TOT + e, ident)


@libentry()
@triton.jit
def _inv_unpack_kernel(
    X, OUT, N, AW: tl.constexpr, TOT: tl.constexpr, BLK: tl.constexpr
):
    """Copy the leading ``n x n`` block of ``X`` back to the output.

    ``BLK`` need not divide ``n * n``; both the source and the destination
    index of the tail lanes are clamped to the last element, so the redundant
    writes are idempotent and no masked store is required.
    """
    b = tle.program_id(0).to(tl.int64)
    # int32 tile arithmetic: see the note in ``_inv_pack_kernel``.
    blk = tl.program_id(1)
    total = N * N
    e = blk * BLK + tl.arange(0, BLK)
    src = tl.where(e < total, e, total - 1)
    row = src // N
    col = src % N
    val = tl.load(X + b * TOT + row * AW + col)
    tl.store(OUT + b * total + src, val)


@libentry()
@triton.jit
def _inv_unpack_flat_kernel(
    X,
    OUT,
    NUM,
    NC: tl.constexpr,
    NN: tl.constexpr,
    AW: tl.constexpr,
    TOT: tl.constexpr,
    BLK: tl.constexpr,
):
    """Whole-batch unpack in one program for small ``batch * n * n``.

    The per-matrix unpack launches ``batch`` programs, and small programs on
    this backend are nearly serial: batch=3 (n=3) measured 34 us vs 10 us for
    batch=1.  When ``batch * n * n`` fits a single tile, flatten the whole
    copy into one program (one flat index -> matrix / row / col).  Tail lanes
    clamp to the last element so redundant writes are idempotent.  ``n`` and
    ``n * n`` are constexpr so the three integer divisions strength-reduce
    (runtime divisors are emulated and measured significant at this size).
    """
    blk = tl.program_id(0)
    nflat = NUM * NN
    e = blk * BLK + tl.arange(0, BLK)
    src = tl.where(e < nflat, e, nflat - 1)
    bi = src // NN
    li = src - bi * NN
    row = li // NC
    col = li % NC
    val = tl.load(X + bi.to(tl.int64) * TOT + row * AW + col)
    tl.store(OUT + src, val)


@libentry()
@triton.jit
def _inv_step_kernel(
    W, X, N, K, AW: tl.constexpr, ROWS: tl.constexpr, TOT: tl.constexpr
):
    """One complete Gauss-Jordan step for a matrix that fits one program.

    Pivot search, row swap, row normalisation and the trailing rank-1 update of
    both ``W`` and ``X`` are fused.  Exactly one program owns the matrix, so the
    in-place store (the ``_det_step_kernel`` regime) cannot race: every read is
    issued before the store of the same program.
    """
    b = tl.program_id(0).to(tl.int64)
    base = b * TOT
    e = tl.arange(0, TOT)
    row = e // AW
    col = e % AW
    w = tl.load(W + base + e)
    x = tl.load(X + base + e)

    # Pivot search over the load-rooted column vector (never tl.argmax).
    ridx = tl.arange(0, ROWS)
    live_rows = ridx < N
    col_k = tl.load(W + base + ridx * AW + K)
    cand = tl.where(live_rows & (ridx >= K), tl.abs(col_k), -1.0)
    best = tl.max(cand, axis=0)
    prow = tl.min(tl.where(cand == best, ridx, ROWS - 1), axis=0)
    akk = tl.sum(tl.where(ridx == K, col_k, 0.0), axis=0)
    apk = tl.sum(tl.where(ridx == prow, col_k, 0.0), axis=0)
    safe = tl.where(apk == 0.0, 1.0, apk)

    row_k_w = tl.load(W + base + K * AW + col)
    row_p_w = tl.load(W + base + prow * AW + col)
    row_k_x = tl.load(X + base + K * AW + col)
    row_p_x = tl.load(X + base + prow * AW + col)
    # Physical swap first: row p takes the old row k, in both W and X.  Column
    # K of the swapped matrix is the old column K except that row p carries
    # the old diagonal ``akk``.
    col_k_elems = tl.load(W + base + row * AW + K)
    swapped_w = tl.where(row == prow, row_k_w, w)
    swapped_x = tl.where(row == prow, row_k_x, x)
    lcol = tl.where(row == prow, akk, col_k_elems)
    mult = tl.where(row == K, 0.0, lcol / safe)

    new_w = swapped_w - mult * row_p_w
    new_x = swapped_x - mult * row_p_x
    # Row K becomes the normalised old pivot row.
    new_w = tl.where(row == K, row_p_w / safe, new_w)
    new_x = tl.where(row == K, row_p_x / safe, new_x)
    # Pin the eliminated column of W to e_K (the dead column stays exact).
    new_w = tl.where(col == K, tl.where(row == K, 1.0, 0.0), new_w)
    tl.store(W + base + e, new_w)
    tl.store(X + base + e, new_x)


@libentry()
@triton.jit
def _inv_gj_kernel(
    W0,
    W1,
    X0,
    X1,
    N,
    AW: tl.constexpr,
    ROWS: tl.constexpr,
    TOT: tl.constexpr,
    BLK: tl.constexpr,
):
    """Whole Gauss-Jordan elimination in a single launch, one matrix/program.

    The per-step host loop costs a ~11 us dispatch floor per launch on this
    backend, which dominates a single small matrix.  This kernel runs the whole
    ``for k`` chain in-kernel instead, ping-ponging ``W``/``X`` pairs so no
    iteration reads the buffer it writes (a single-buffer in-kernel loop is
    silently reordered by this backend -- see ``linalg_det._det_elim_kernel``).

    Only used for a tiny batch and a >= 1024-lane iteration tile, the regime in
    which that pattern was measured exact.
    """
    b = tl.program_id(0).to(tl.int64)
    base = b * TOT
    ridx = tl.arange(0, ROWS)
    live_rows = ridx < N
    nchunk = TOT // BLK
    for k in range(0, N):
        is_even = (k % 2) == 0
        src_w = tl.where(is_even, W0, W1)
        dst_w = tl.where(is_even, W1, W0)
        src_x = tl.where(is_even, X0, X1)
        dst_x = tl.where(is_even, X1, X0)

        col_k_sp = tl.load(src_w + base + ridx * AW + k)
        cand = tl.where(live_rows & (ridx >= k), tl.abs(col_k_sp), -1.0)
        best = tl.max(cand, axis=0)
        prow = tl.min(tl.where(cand == best, ridx, ROWS - 1), axis=0)
        akk = tl.sum(tl.where(ridx == k, col_k_sp, 0.0), axis=0)
        apk = tl.sum(tl.where(ridx == prow, col_k_sp, 0.0), axis=0)
        safe = tl.where(apk == 0.0, 1.0, apk)

        for c in range(0, nchunk):
            e = c * BLK + tl.arange(0, BLK)
            row = e // AW
            col = e % AW
            w = tl.load(src_w + base + e)
            x = tl.load(src_x + base + e)
            row_k_w = tl.load(src_w + base + k * AW + col)
            row_p_w = tl.load(src_w + base + prow * AW + col)
            row_k_x = tl.load(src_x + base + k * AW + col)
            row_p_x = tl.load(src_x + base + prow * AW + col)
            col_k = tl.load(src_w + base + e - col + k)
            swapped_w = tl.where(row == prow, row_k_w, w)
            swapped_x = tl.where(row == prow, row_k_x, x)
            lcol = tl.where(row == prow, akk, col_k)
            mult = tl.where(row == k, 0.0, lcol / safe)
            new_w = swapped_w - mult * row_p_w
            new_x = swapped_x - mult * row_p_x
            new_w = tl.where(row == k, row_p_w / safe, new_w)
            new_x = tl.where(row == k, row_p_x / safe, new_x)
            new_w = tl.where(col == k, tl.where(row == k, 1.0, 0.0), new_w)
            tl.store(dst_w + base + e, new_w)
            tl.store(dst_x + base + e, new_x)


@libentry()
@triton.jit
def _inv_scan_group_kernel(
    W,
    PR,
    AP,
    AK,
    N,
    K,
    AW: tl.constexpr,
    ROWS: tl.constexpr,
    TOT: tl.constexpr,
    BB: tl.constexpr,
):
    """Pivot search for ``BB`` matrices at once (one program per group).

    Only the pivot column of each matrix is touched, as a ``[BB, ROWS]`` tile,
    so the reductions stay on a single axis -- reducing a second axis of the
    same tile is not tileable on this backend.  The pivot row, the signed pivot
    and the old diagonal are handed to the update kernel through side buffers.
    """
    g = tl.program_id(0).to(tl.int64)
    m = tl.arange(0, BB).to(tl.int64)
    ridx = tl.arange(0, ROWS)[None, :]
    idx = g * BB + m
    mbase = idx[:, None] * TOT
    cks = tl.load(W + mbase + ridx * AW + K)
    cand = tl.where((ridx >= K) & (ridx < N), tl.abs(cks), -1.0)
    best = tl.max(cand, axis=1)[:, None]
    prow = tl.min(tl.where(cand == best, ridx, ROWS - 1), axis=1)
    apk = tl.sum(tl.where(ridx == prow[:, None], cks, 0.0), axis=1)
    akk = tl.sum(tl.where(ridx == K, cks, 0.0), axis=1)
    tl.store(PR + idx, prow)
    tl.store(AP + idx, apk)
    tl.store(AK + idx, akk)


@libentry()
@triton.jit
def _inv_update_group_kernel(
    SRC_W,
    DST_W,
    SRC_X,
    DST_X,
    PR,
    AP,
    AK,
    K,
    AW: tl.constexpr,
    ROWS: tl.constexpr,
    TOT: tl.constexpr,
    BB: tl.constexpr,
    NLANE: tl.constexpr,
):
    """Swap + normalise + eliminate for ``BB`` matrices (flat ``NLANE`` tile).

    The update must not write a buffer it reads: with ``BB`` matrices per
    program the flat tile is chunked and a later chunk would gather a row an
    earlier chunk has already overwritten.  The caller therefore ping-pongs
    ``W`` and ``X`` across steps.  The pivot column is addressed as
    ``off - col + K`` (the ``mbase + row * AW + K`` form is miscompiled for
    wide tiles on this backend).
    """
    g = tl.program_id(0).to(tl.int64)
    e = tl.arange(0, NLANE)
    gb = g * BB
    mm = e // TOT
    inner = e % TOT
    row = inner // AW
    col = inner % AW
    off = gb * TOT + e
    mbase = gb * TOT + mm * TOT
    midx = gb + mm
    prow = tl.load(PR + midx)
    apk = tl.load(AP + midx)
    akk = tl.load(AK + midx)
    safe = tl.where(apk == 0.0, 1.0, apk)

    w = tl.load(SRC_W + off)
    x = tl.load(SRC_X + off)
    row_k_w = tl.load(SRC_W + mbase + K * AW + col)
    row_p_w = tl.load(SRC_W + mbase + prow * AW + col)
    row_k_x = tl.load(SRC_X + mbase + K * AW + col)
    row_p_x = tl.load(SRC_X + mbase + prow * AW + col)
    col_k = tl.load(SRC_W + off - col + K)

    swapped_w = tl.where(row == prow, row_k_w, w)
    swapped_x = tl.where(row == prow, row_k_x, x)
    lcol = tl.where(row == prow, akk, col_k)
    mult = tl.where(row == K, 0.0, lcol / safe)

    new_w = swapped_w - mult * row_p_w
    new_x = swapped_x - mult * row_p_x
    new_w = tl.where(row == K, row_p_w / safe, new_w)
    new_x = tl.where(row == K, row_p_x / safe, new_x)
    new_w = tl.where(col == K, tl.where(row == K, 1.0, 0.0), new_w)
    tl.store(DST_W + off, new_w)
    tl.store(DST_X + off, new_x)


@libentry()
@triton.jit
def _inv_pivot_swap_norm_kernel(
    W,
    X,
    FACT,
    N,
    K,
    AW: tl.constexpr,
    ROWS: tl.constexpr,
    TOT: tl.constexpr,
):
    """Pivot search + physical swap + row-K normalisation (one matrix/program).

    Runs as its own launch so the following eliminate launch never writes the
    pivot row it reads.  The post-swap elimination factors are saved to
    ``FACT`` here: recovering them in the eliminate kernel from ``W`` would
    race against that kernel's own writes to column ``K``.
    """
    b = tl.program_id(0).to(tl.int64)
    base = b * TOT

    ridx = tl.arange(0, ROWS)
    live_rows = ridx < N
    col_k = tl.load(W + base + ridx * AW + K)
    cand = tl.where(live_rows & (ridx >= K), tl.abs(col_k), -1.0)
    best = tl.max(cand, axis=0)
    prow = tl.min(tl.where(cand == best, ridx, ROWS - 1), axis=0)
    akk = tl.sum(tl.where(ridx == K, col_k, 0.0), axis=0)
    apk = tl.sum(tl.where(ridx == prow, col_k, 0.0), axis=0)
    safe = tl.where(apk == 0.0, 1.0, apk)

    cols = tl.arange(0, AW)
    row_k_w = tl.load(W + base + K * AW + cols)
    row_p_w = tl.load(W + base + prow * AW + cols)
    row_k_x = tl.load(X + base + K * AW + cols)
    row_p_x = tl.load(X + base + prow * AW + cols)

    same = prow == K
    norm_w = row_p_w / safe
    norm_x = row_p_x / safe
    # Row K carries the normalised old pivot row; row p carries the old row K.
    # When p == K both stores hit the same row with the same value.
    tl.store(W + base + K * AW + cols, norm_w)
    tl.store(X + base + K * AW + cols, norm_x)
    tl.store(W + base + prow * AW + cols, tl.where(same, norm_w, row_k_w))
    tl.store(X + base + prow * AW + cols, tl.where(same, norm_x, row_k_x))

    fact = tl.where(ridx == K, 0.0, tl.where(ridx == prow, akk, col_k))
    tl.store(FACT + b * ROWS + ridx, fact)


@libentry()
@triton.jit
def _inv_elim_kernel(
    W,
    X,
    FACT,
    K,
    AW: tl.constexpr,
    ROWS: tl.constexpr,
    TOT: tl.constexpr,
    BLK: tl.constexpr,
):
    """Rank-1 elimination of the remaining rows for one step.

    ``BLK`` divides ``TOT`` exactly, so no masked store is needed.  Reads the
    (stable) pivot row K and the per-row factors; writes every row ``i != K``
    of both tiles.  ``W``'s eliminated column is pinned to e_K.
    """
    b = tl.program_id(0).to(tl.int64)
    blk = tl.program_id(1).to(tl.int64)
    base = b * TOT
    e = blk * BLK + tl.arange(0, BLK)
    row = e // AW
    col = e % AW
    is_k = row == K
    factor = tl.load(FACT + b * ROWS + row)
    row_k_w = tl.load(W + base + K * AW + col)
    row_k_x = tl.load(X + base + K * AW + col)
    w = tl.load(W + base + e)
    x = tl.load(X + base + e)
    new_w = tl.where(col == K, 0.0, w - factor * row_k_w)
    new_x = x - factor * row_k_x
    tl.store(W + base + e, tl.where(is_k, w, new_w))
    tl.store(X + base + e, tl.where(is_k, x, new_x))


def _run_fused(w, x, batch, n, rows, aw, tot):
    for k in range(n):
        _inv_step_kernel[(batch,)](w, x, n, k, AW=aw, ROWS=rows, TOT=tot, num_warps=1)


def _run_gj(w, x, batch, n, rows, aw, tot, blk):
    w1 = torch.empty_like(w)
    x1 = torch.empty_like(x)
    _inv_gj_kernel[(batch,)](
        w, w1, x, x1, n, AW=aw, ROWS=rows, TOT=tot, BLK=blk, num_warps=2
    )
    # Each iteration writes the buffer pair it did not read, so the last
    # written pair is the (n - 1) parity one.
    return x1 if (n % 2 == 1) else x


def _run_grouped(w0, x0, w1, x1, batch, n, rows, aw, tot, group):
    grid = (batch // group,)
    pr = torch.empty(batch, dtype=torch.int32, device=w0.device)
    ap = torch.empty(batch, dtype=w0.dtype, device=w0.device)
    ak = torch.empty(batch, dtype=w0.dtype, device=w0.device)
    nlanes = group * tot
    src_w, dst_w, src_x, dst_x = w0, w1, x0, x1
    for k in range(n):
        _inv_scan_group_kernel[grid](
            src_w, pr, ap, ak, n, k, AW=aw, ROWS=rows, TOT=tot, BB=group, num_warps=1
        )
        _inv_update_group_kernel[grid](
            src_w,
            dst_w,
            src_x,
            dst_x,
            pr,
            ap,
            ak,
            k,
            AW=aw,
            ROWS=rows,
            TOT=tot,
            BB=group,
            NLANE=nlanes,
            num_warps=4,
        )
        src_w, dst_w = dst_w, src_w
        src_x, dst_x = dst_x, src_x
    return src_x


def _run_chunked(w, x, batch, n, rows, aw, tot, blk):
    nblk = tot // blk
    fact = torch.empty(batch * rows, dtype=w.dtype, device=w.device)
    for k in range(n):
        _inv_pivot_swap_norm_kernel[(batch,)](
            w, x, fact, n, k, AW=aw, ROWS=rows, TOT=tot, num_warps=1
        )
        _inv_elim_kernel[(batch, nblk)](
            w, x, fact, k, AW=aw, ROWS=rows, TOT=tot, BLK=blk, num_warps=4
        )


def _inverse_impl(A):
    n = A.shape[-1]
    batch_shape = A.shape[:-2]
    batch = 1
    for dim in batch_shape:
        batch *= dim

    rows, aw, tot, blk, _ = _plan(n)
    # A single small batch is launch-bound: run the whole elimination chain in
    # one kernel.  Restricted to the tile/batch regime where the in-kernel
    # loop was measured exact (see ``_inv_gj_kernel``).
    use_gj = batch <= _ELIM_MAX_BATCH and rows <= 32
    if use_gj:
        rows, aw, tot, blk, _ = _plan_pad(n)
    A_work = A.contiguous().reshape(batch, n, n)
    dtype = A.dtype
    device = A.device
    w = torch.empty(batch * tot, dtype=dtype, device=device)
    x = torch.empty(batch * tot, dtype=dtype, device=device)
    pack_blk = blk
    pack_grid = (batch, tot // pack_blk)

    # Contiguous output explicitly: ``empty_like`` would preserve a transposed
    # input's strides while the kernels address the result as a flat buffer.
    out = torch.empty(A.shape, dtype=dtype, device=device)
    out_flat = out.reshape(batch, n, n)

    # The per-step chunked path (per-matrix pivot/swap/normalise) is only exact
    # with a >= 64-wide row stride on this backend; every layout narrower than
    # that (tot <= 2048) goes through the fused or grouped kernels instead.
    group = _pick_group(batch, tot) if tot <= _FUSED_MAX_TOT else 1
    use_grouped = batch >= _GROUP_MIN_BATCH and group > 1

    with torch_device_fn.device(device):
        # Pack stays a separate launch: an in-kernel prologue that builds
        # ``W0``/``X0`` from ``A`` was measured to be reordered by this backend
        # (same-kernel global write-then-read, wrong values at batch=4).
        _inv_pack_kernel[pack_grid](
            A_work, w, x, n, AW=aw, TOT=tot, BLK=pack_blk, num_warps=1
        )
        if use_gj:
            out_x = _run_gj(w, x, batch, n, rows, aw, tot, blk)
        elif use_grouped:
            w1 = torch.empty_like(w)
            x1 = torch.empty_like(x)
            out_x = _run_grouped(w, x, w1, x1, batch, n, rows, aw, tot, group)
        elif tot <= _FUSED_MAX_TOT:
            _run_fused(w, x, batch, n, rows, aw, tot)
            out_x = x
        else:
            _run_chunked(w, x, batch, n, rows, aw, tot, blk)
            out_x = x

        # Contiguous output explicitly: ``empty_like`` would preserve a
        # transposed input's strides while the unpack addresses the result as
        # a flat ``batch * n * n`` buffer.
        #
        # The unpack tile only ever touches ``n * n`` elements, but ``pack_blk``
        # is sized for the padded working tile (up to 1024 lanes).  Launching
        # that many lanes for a 2x2 matrix makes the per-lane int64 ``//``/``%``
        # dominate the whole inverse on this backend (measured 337 us/call for
        # the small benchmark shapes, 91% of the total).  Clamp the unpack tile
        # to the output size; it is left unchanged for large ``n`` where
        # ``pack_blk`` already divides ``n * n``.
        #
        # Small programs are nearly serial on this backend, so for small
        # ``batch * n * n`` the per-matrix grid's ``batch`` programs cost more
        # than the copy itself (batch=3/n=3: 34 us vs 10 us at batch=1).  Flatten
        # the whole batch into a single program in that regime.
        nflat = batch * n * n
        if nflat <= 1024:
            flat_blk = max(1, triton.next_power_of_2(nflat))
            _inv_unpack_flat_kernel[(1,)](
                out_x,
                out_flat,
                batch,
                NC=n,
                NN=n * n,
                AW=aw,
                TOT=tot,
                BLK=flat_blk,
                num_warps=1,
            )
        else:
            unpack_blk = min(pack_blk, triton.next_power_of_2(max(n * n, 1)))
            _inv_unpack_kernel[(batch, triton.cdiv(n * n, unpack_blk))](
                out_x, out_flat, n, AW=aw, TOT=tot, BLK=unpack_blk, num_warps=1
            )
    return out


def inverse(A):
    """Invert a square matrix (batched), equivalent to ``torch.inverse``."""
    logger.debug("GEMS_KUNLUNXIN INVERSE")

    assert A.dim() >= 2, "inverse: input must be at least 2D"
    assert A.shape[-1] == A.shape[-2], "inverse: input must be a square matrix"
    assert A.dtype in (
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    ), f"inverse: unsupported dtype {A.dtype}"

    if A.dtype in (torch.complex64, torch.complex128):
        # The XPU3 stack has no complex linear-algebra path at all
        # (torch.eye(complex64) is UNIMPLEMENTED and the vendor linalg_inv
        # rejects kcomplex64), so the previous vendor delegation could not
        # handle complex inputs either.  Fail explicitly instead of silently
        # producing a real-valued result.
        raise NotImplementedError(
            "Kunlunxin inverse does not support complex dtypes: the XPU3 "
            "stack provides no complex linear-algebra primitive"
        )

    if A.numel() == 0:
        return torch.empty_like(A)

    return _inverse_impl(A)

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_MIN_LDA = 64
_MAX_BLK = 4096
_MULTI_MIN_BATCH = 128
_MIN_DET4_GROUP = 64
# A grouped elimination tile may use twice a single block as long as the
# per-matrix tile is itself block sized (n >= 32): measured 2.26x at
# batch 512 / n 32 for 8192 lanes against 2.06x for 4096 and 1.82x for 16384.
# Small-n layouts (TOT <= 512) lose from the wider tile and keep _MAX_BLK.
_MAX_GROUP_LANES = 8192
# Grouping pays off while the per-matrix tile is small enough for the group to
# hold several of them.  At TOT = 4096 only BB = 2 fits and the second launch
# per step then costs more than the grouping saves (n = 64 / batch 256:
# grouped 14.3 ms vs 13.3 ms for the launch-per-step loop), so those shapes
# take the plain per-step loop instead.
_MAX_GROUP_TOT = 2048
# Single-launch elimination needs a >= 1024 lane iteration tile, and for the
# padded small-n layouts it is only reproducible up to a handful of matrices
# (bit-exact and stable for batch <= 8, deterministically corrupt from 16).
_ELIM_MIN_TOT = 1024
_ELIM_SMALL_MAX_BATCH = 4


@libentry()
@triton.jit
def _det4_kernel(A, OUT, TOT: tl.constexpr):
    """Closed-form 4x4 determinant.

    The cofactor expansion is evaluated directly in registers: the matrix is
    fetched with 16 scalar loads (one per entry; TOT == 16) and the formula
    only does multiplies/subtracts.  There is no working-buffer store/load
    round trip, so this path is immune to the backend's unsafe in-kernel
    store->load reordering (the reason every other kernel is launched once per
    step).
    """
    b = tle.program_id(0).to(tl.int64)
    base = b * TOT
    a00 = tl.load(A + base + 0)
    a01 = tl.load(A + base + 1)
    a02 = tl.load(A + base + 2)
    a03 = tl.load(A + base + 3)
    a10 = tl.load(A + base + 4)
    a11 = tl.load(A + base + 5)
    a12 = tl.load(A + base + 6)
    a13 = tl.load(A + base + 7)
    a20 = tl.load(A + base + 8)
    a21 = tl.load(A + base + 9)
    a22 = tl.load(A + base + 10)
    a23 = tl.load(A + base + 11)
    a30 = tl.load(A + base + 12)
    a31 = tl.load(A + base + 13)
    a32 = tl.load(A + base + 14)
    a33 = tl.load(A + base + 15)
    det = (
        a00
        * (
            a11 * (a22 * a33 - a23 * a32)
            - a12 * (a21 * a33 - a23 * a31)
            + a13 * (a21 * a32 - a22 * a31)
        )
        - a01
        * (
            a10 * (a22 * a33 - a23 * a32)
            - a12 * (a20 * a33 - a23 * a30)
            + a13 * (a20 * a32 - a22 * a30)
        )
        + a02
        * (
            a10 * (a21 * a33 - a23 * a31)
            - a11 * (a20 * a33 - a23 * a30)
            + a13 * (a20 * a31 - a21 * a30)
        )
        - a03
        * (
            a10 * (a21 * a32 - a22 * a31)
            - a11 * (a20 * a32 - a22 * a30)
            + a12 * (a20 * a31 - a21 * a30)
        )
    )
    tl.store(OUT + b, det)


@libentry()
@triton.jit
def _det4_group_kernel(A, OUT, BB: tl.constexpr):
    """Closed-form 4x4 determinant for BB matrices per program.

    Identical arithmetic to ``_det4_kernel`` (and bit-exact against it), but the
    16 scalar loads become 16 BB-lane stride-16 loads, so a batch of 4096 costs
    16 programs instead of 4096.  One matrix per program was spending ~1.3us of
    per-program dispatch on 64 bytes of data.

    BB must be >= 64: below that the [BB]-lane result store is not reliably
    bit-reproducible on this backend.  At BB == 16 it disagreed with the
    scalar kernel and with itself across runs; BB == 32 looked exact in some
    probes but at batch 224 (grid 7) a single process saw BB == 8 disagree
    across four runs and BB == 16/32 disagree with the scalar kernel, while a
    second process saw all of them agree -- intermittent, so the gate stays
    where every probe was stable.
    """
    g = tle.program_id(0).to(tl.int64)
    idx = g * BB + tl.arange(0, BB).to(tl.int64)
    base = idx * 16
    a00 = tl.load(A + base + 0)
    a01 = tl.load(A + base + 1)
    a02 = tl.load(A + base + 2)
    a03 = tl.load(A + base + 3)
    a10 = tl.load(A + base + 4)
    a11 = tl.load(A + base + 5)
    a12 = tl.load(A + base + 6)
    a13 = tl.load(A + base + 7)
    a20 = tl.load(A + base + 8)
    a21 = tl.load(A + base + 9)
    a22 = tl.load(A + base + 10)
    a23 = tl.load(A + base + 11)
    a30 = tl.load(A + base + 12)
    a31 = tl.load(A + base + 13)
    a32 = tl.load(A + base + 14)
    a33 = tl.load(A + base + 15)
    det = (
        a00
        * (
            a11 * (a22 * a33 - a23 * a32)
            - a12 * (a21 * a33 - a23 * a31)
            + a13 * (a21 * a32 - a22 * a31)
        )
        - a01
        * (
            a10 * (a22 * a33 - a23 * a32)
            - a12 * (a20 * a33 - a23 * a30)
            + a13 * (a20 * a32 - a22 * a30)
        )
        + a02
        * (
            a10 * (a21 * a33 - a23 * a31)
            - a11 * (a20 * a33 - a23 * a30)
            + a13 * (a20 * a31 - a21 * a30)
        )
        - a03
        * (
            a10 * (a21 * a32 - a22 * a31)
            - a11 * (a20 * a32 - a22 * a30)
            + a12 * (a20 * a31 - a21 * a30)
        )
    )
    tl.store(OUT + idx, det)


@triton.jit
def _reduce_mul(a, b):
    return a * b


def _pick_group(batch_count, tot):
    """Matrices per program for the batched elimination path.

    Per-launch pricing on this backend puts the elimination step at
    ~0.46-0.56us *per program* with only a weak dependence on the tile size
    (TOT 256 -> 512 costs ~20% more), i.e. one matrix per program spends
    almost all of its time on per-program dispatch rather than on the 1-2 KB
    it touches.  Grouping BB matrices into one program is the only lever on
    that term, so take the largest power of two whose flat tile still fits a
    single _MAX_BLK block and that divides the batch exactly -- a partial
    group would need masked loads, which this backend does not honour.

    Matrices whose own tile already fills a block (TOT >= 1024, i.e. n >= 32)
    are allowed twice that budget; wider tiles were measured to help there
    (n = 32: 2.26x at 8192 lanes vs 2.06x at 4096) and to hurt for the small-n
    layouts (n = 8: 0.72x at 8192), which stay at one block.
    """
    cap = _MAX_GROUP_LANES if tot >= _ELIM_MIN_TOT else _MAX_BLK
    bb = 1
    while bb * 2 * tot <= cap and batch_count % (bb * 2) == 0:
        bb *= 2
    return bb


def _plan(n):
    rows = triton.next_power_of_2(n)
    if n >= 16 and n % 8 == 0 and n * n <= _MAX_BLK:
        lda = n
    else:
        lda = max(_MIN_LDA, rows)
    tot = rows * lda
    blk = min(_MAX_BLK, tot)
    return rows, lda, tot, blk, tot // blk


def _plan_pad(n):
    """Layout for n < 32 on the single-launch elimination path.

    ``_det_elim_kernel`` is only exact when every iteration works on at least
    _ELIM_MIN_TOT lanes, which the natural small-n layouts do not reach
    (n = 8 -> 512, n = 16 -> 256), so the row stride is padded up until the
    tile does.  Padding lanes are zeroed by ``_det_pack_kernel`` and masked by
    the kernel, exactly as on the launch-per-step path.
    """
    rows = triton.next_power_of_2(n)
    lda = max(_MIN_LDA, rows)
    while rows * lda < _ELIM_MIN_TOT:
        lda *= 2
    tot = rows * lda
    return rows, lda, tot, tot, 1


@libentry()
@triton.jit
def _det_pack_kernel(
    SRC, DST, N, LDA: tl.constexpr, BLK: tl.constexpr, TOT: tl.constexpr
):
    """Scatter a contiguous (batch, N, N) buffer into (batch, ROWS, LDA).

    Padding lanes are zeroed rather than masked away: ``other=`` silently
    pollutes live lanes here, and a masked store is not honoured at all.
    """
    b = tle.program_id(0).to(tl.int64)
    blk = tle.program_id(1).to(tl.int64)
    e = blk * BLK + tl.arange(0, BLK)
    row = e // LDA
    col = e % LDA
    live = (row < N) & (col < N)
    idx = tl.where(live, row * N + col, 0)
    val = tl.load(SRC + b * N * N + idx)
    tl.store(DST + b * TOT + e, tl.where(live, val, 0.0))


@libentry()
@triton.jit
def _det_step0_kernel(SRC, W, DG, N, LDA: tl.constexpr, TOT: tl.constexpr):
    """Elimination step K=0 with the pack fused in.

    Only valid when W is a separate (padded) buffer: every read comes from SRC
    (contiguous N*N) and W is written only, so this one launch replaces the
    pack kernel plus step 0.  The sum-based ``akk``/``apk`` extraction is kept
    (a scalar load of the runtime ``prow`` address races against the stores of
    the previous launch on this backend).
    """
    b = tle.program_id(0).to(tl.int64)
    base = b * TOT
    sbase = b * N * N
    e = tl.arange(0, TOT)
    row = e // LDA
    col = e % LDA
    live = (row < N) & (col < N)
    idx = tl.where(live, row * N + col, 0)
    w = tl.load(SRC + sbase + idx)
    cand = tl.where((col == 0) & (row < N), tl.abs(w), -1.0)
    best = tl.max(cand, axis=0)
    prow = tl.min(tl.where(cand == best, row, TOT), axis=0)
    akk = tl.sum(tl.where((row == 0) & (col == 0), w, 0.0), axis=0)
    apk = tl.sum(tl.where((row == prow) & (col == 0), w, 0.0), axis=0)
    cidx = tl.where(col < N, col, 0)
    ridx = tl.where(row < N, row, 0)
    row_k = tl.load(SRC + sbase + cidx)
    row_p = tl.load(SRC + sbase + prow * N + cidx)
    col_k = tl.load(SRC + sbase + ridx * N)
    swapped = tl.where(row == 0, row_p, tl.where(row == prow, row_k, w))
    lcol = tl.where(row == 0, apk, tl.where(row == prow, akk, col_k))
    safe = tl.where(apk == 0.0, 1.0, apk)
    mult = tl.where((row > 0) & (row < N), lcol / safe, 0.0)
    urow = tl.where(col > 0, row_p, 0.0)
    tl.store(W + base + e, swapped - mult * urow)
    tl.store(DG + b * LDA, tl.where(prow != 0, -apk, apk))


@libentry()
@triton.jit
def _det_step_kernel(W, DG, N, K, LDA: tl.constexpr, TOT: tl.constexpr):
    """One complete elimination step for a matrix that fits in a single program.

    Pivot search, row swap and the trailing rank-1 update are fused.  Because
    exactly one program owns the whole matrix there is no cross-program race,
    and because ``K`` comes from the host there is no runtime loop around the
    global store/load round trip (an in-kernel loop over K silently corrupted
    ~4% of the matrices from 48 matrices upward even with debug barriers).

    The pivot search operates on a [ROWS]-shaped column gather instead of the
    [TOT]-shaped working tile: the masked 1-D reductions over the full tile
    were measured as the dominant cost of every step (a 512-lane ``tl.max``
    costs ~3x a 512-lane ``tl.sum`` on this backend), so the three per-step
    reductions run over ROWS <= 64 lanes.  Every ``tl.where``/``tl.select`` is
    still [TOT]-shaped; only the reduction inputs are [ROWS].
    """
    b = tle.program_id(0).to(tl.int64)
    base = b * TOT
    e = tl.arange(0, TOT)
    row = e // LDA
    col = e % LDA
    w = tl.load(W + base + e)
    ridx = tl.arange(0, LDA)
    col_k_sp = tl.load(W + base + ridx * LDA + K)
    cand = tl.where((ridx >= K) & (ridx < N), tl.abs(col_k_sp), -1.0)
    best = tl.max(cand, axis=0)
    prow = tl.min(tl.where(cand == best, ridx, LDA), axis=0)
    akk = tl.sum(tl.where(ridx == K, col_k_sp, 0.0), axis=0)
    apk = tl.sum(tl.where(ridx == prow, col_k_sp, 0.0), axis=0)
    row_k = tl.load(W + base + K * LDA + col)
    row_p = tl.load(W + base + prow * LDA + col)
    col_k = tl.load(W + base + row * LDA + K)
    swapped = tl.where(row == K, row_p, tl.where(row == prow, row_k, w))
    lcol = tl.where(row == K, apk, tl.where(row == prow, akk, col_k))
    safe = tl.where(apk == 0.0, 1.0, apk)
    mult = tl.where(row > K, lcol / safe, 0.0)
    urow = tl.where(col > K, row_p, 0.0)
    tl.store(W + base + e, swapped - mult * urow)
    tl.store(DG + b * LDA + K, tl.where(prow != K, -apk, apk))


@libentry()
@triton.jit
def _det_pivot_scan_kernel(
    W,
    PR,
    AP,
    AK,
    DG,
    N,
    K,
    LDA: tl.constexpr,
    ROWS: tl.constexpr,
    TOT: tl.constexpr,
    BB: tl.constexpr,
):
    """Pivot search for BB matrices at once, one program per group.

    Only the pivot column of each matrix is touched, as a [BB, ROWS] tile, so
    the three reductions stay on a single axis (axis=1).  Reducing a second
    axis of the same tile is not tileable on this backend, which is why the
    trailing update cannot live in this kernel and gets its own launch; a
    fused [BB, TOT] version fails to tile with out of resource: uni_sram even
    at BB == 1.  ROWS (not LDA) bounds the scan so the padded n=8 layout does
    not pay for 64 rows, and the pivot row / signed pivot are handed to the
    update kernel through small side buffers.
    """
    g = tle.program_id(0).to(tl.int64)
    m = tl.arange(0, BB).to(tl.int64)
    ridx = tl.arange(0, ROWS)[None, :]
    idx = g * BB + m
    mbase = idx[:, None] * TOT
    cks = tl.load(W + mbase + ridx * LDA + K)
    cand = tl.where((ridx >= K) & (ridx < N), tl.abs(cks), -1.0)
    best = tl.max(cand, axis=1)[:, None]
    prow = tl.min(tl.where(cand == best, ridx, ROWS), axis=1)
    apk = tl.sum(tl.where(ridx == prow[:, None], cks, 0.0), axis=1)
    akk = tl.sum(tl.where(ridx == K, cks, 0.0), axis=1)
    tl.store(PR + idx, prow)
    tl.store(AP + idx, apk)
    tl.store(AK + idx, akk)
    tl.store(DG + idx * LDA + K, tl.where(prow != K, -apk, apk))


@libentry()
@triton.jit
def _det_update_group_kernel(
    SRC,
    DST,
    PR,
    AP,
    AK,
    K,
    LDA: tl.constexpr,
    TOT: tl.constexpr,
    BB: tl.constexpr,
    NLANE: tl.constexpr,
):
    """Swap + trailing rank-1 update for BB matrices, one flat NLANE tile.

    Two backend constraints shape this kernel:

    * It must not write the buffer it reads.  With one matrix per program the
      in-place form of ``_det_step_kernel`` is safe, but once a program owns
      BB matrices the tile is split into chunks and a later chunk gathers a
      row an earlier chunk has already overwritten (every matrix came out
      wrong at NLANE >= 2048, deterministically, even with grid == 2).  The
      caller therefore ping-pongs two work buffers across K.
    * The L column must be addressed as ``off - col + K``.  The algebraically
      identical ``mbase + row * LDA + K`` gather is miscompiled for
      NLANE >= 2048 (it was the only construct that changed results between
      BB == 4 and BB == 8 in a per-construct differential); the row-broadcast
      gathers ``K * LDA + col`` / ``prow * LDA + col`` are fine.
    """
    g = tle.program_id(0).to(tl.int64)
    e = tl.arange(0, NLANE)
    gb = g * BB
    mm = e // TOT
    inner = e % TOT
    row = inner // LDA
    col = inner % LDA
    off = gb * TOT + e
    mbase = gb * TOT + mm * TOT
    w = tl.load(SRC + off)
    prow = tl.load(PR + gb + mm)
    apk = tl.load(AP + gb + mm)
    akk = tl.load(AK + gb + mm)
    row_k = tl.load(SRC + mbase + K * LDA + col)
    row_p = tl.load(SRC + mbase + prow * LDA + col)
    col_k = tl.load(SRC + off - col + K)
    swapped = tl.where(row == K, row_p, tl.where(row == prow, row_k, w))
    lcol = tl.where(row == K, apk, tl.where(row == prow, akk, col_k))
    safe = tl.where(apk == 0.0, 1.0, apk)
    mult = tl.where(row > K, lcol / safe, 0.0)
    urow = tl.where(col > K, row_p, 0.0)
    tl.store(DST + off, swapped - mult * urow)


@libentry()
@triton.jit
def _det_pivot_swap_kernel(W, DG, N, K, LDA: tl.constexpr, ROWS: tl.constexpr):
    """Pivot search plus physical row swap, one matrix per program.

    ``tl.argmax`` is unreliable on this backend, so the pivot row is the
    smallest row attaining a plain 1-D ``tl.max`` (LAPACK first-strict-maximum
    order).  The signed pivot goes straight into DG[K] so the determinant is a
    plain product over K and no separate parity buffer is needed.
    """
    b = tle.program_id(0).to(tl.int64)
    base = b * ROWS * LDA
    rows = tl.arange(0, ROWS)
    cols = tl.arange(0, LDA)
    live = rows < N
    vals = tl.load(W + base + tl.where(live, rows, 0) * LDA + K)
    cand = tl.where(live & (rows >= K), tl.abs(vals), -1.0)
    best = tl.max(cand, axis=0)
    prow = tl.min(tl.where(cand == best, rows, ROWS), axis=0)
    prow = tl.where(prow >= N, K, prow)
    row_k = tl.load(W + base + K * LDA + cols)
    row_p = tl.load(W + base + prow * LDA + cols)
    tl.store(W + base + K * LDA + cols, row_p)
    tl.store(W + base + prow * LDA + cols, row_k)
    pivot = tl.sum(tl.where(cols == K, row_p, 0.0), axis=0)
    tl.store(DG + b * LDA + K, tl.where(prow != K, -pivot, pivot))


@libentry()
@triton.jit
def _det_update_kernel(
    W, N, K, LDA: tl.constexpr, BLK: tl.constexpr, TOT: tl.constexpr
):
    """Trailing rank-1 update of one flat, contiguous BLK-lane chunk.

    Used for matrices too large for ``_det_step_kernel``; the swap has already
    been applied by ``_det_pivot_swap_kernel`` so chunks never read a row that
    another program is rewriting.
    """
    b = tle.program_id(0).to(tl.int64)
    blk = tle.program_id(1).to(tl.int64)
    base = b * TOT
    e = blk * BLK + tl.arange(0, BLK)
    row = e // LDA
    col = e % LDA
    pivot_row = W + base + K * LDA
    pivot = tl.load(pivot_row + K)
    safe = tl.where(pivot == 0.0, 1.0, pivot)
    urow = tl.where(col > K, tl.load(pivot_row + col), 0.0)
    lcol = tl.load(W + base + row * LDA + K)
    mult = tl.where((row > K) & (row < N), lcol / safe, 0.0)
    tile = tl.load(W + base + e)
    tl.store(W + base + e, tile - mult * urow)


@libentry()
@triton.jit
def _det_reduce_kernel(DG, OUT, N, LDA: tl.constexpr):
    b = tle.program_id(0).to(tl.int64)
    cols = tl.arange(0, LDA)
    v = tl.load(DG + b * LDA + cols)
    det = tl.reduce(tl.where(cols < N, v, 1.0), 0, combine_fn=_reduce_mul)
    tl.store(OUT + b, det)


@libentry()
@triton.jit
def _det_elim_kernel(
    W0, W1, OUT, N, LDA: tl.constexpr, TOT: tl.constexpr, BLK: tl.constexpr
):
    """Single-launch Gaussian elimination with partial pivoting, one program
    per matrix.

    The elimination runs entirely inside the kernel (``for k in range(N)``
    with an inner ``for c`` over BLK-lane chunks for matrices that do not fit
    one 4096-lane block).  The workspace is double buffered: iteration k reads
    ``W0``/``W1`` and writes the other (``is_even`` selects both the source and
    the destination from k).  No iteration reads the buffer it writes, so the
    backend's in-kernel store->load reordering -- which corrupted ~4% of
    matrices in a single-buffer loop even with debug barriers -- cannot
    surface: a reordered load only hits the buffer that is not being written
    this iteration, and the source pointer is a function of k so the compiler
    cannot CSE loads across iterations.

    Measured on this backend the pattern is exact only when the per-iteration
    tile is >= 1024 lanes (TOT >= 1024); smaller tiles are reordered and give
    wrong results, so n < 32 is padded up to _ELIM_MIN_TOT lanes before taking
    this path.  Padding alone is not enough once many programs run at once:
    with the padded small-n layouts the result is bit-exact and run-to-run
    stable up to 8 matrices and deterministically corrupt from 16 (it is the
    natural TOT >= 1024 layouts, n >= 32, that stay exact at batch 512), hence
    the _ELIM_SMALL_MAX_BATCH gate on the caller side.
    """
    b = tle.program_id(0).to(tl.int64)
    base = b * TOT
    ridx = tl.arange(0, LDA)
    acc = 1.0
    for k in range(0, N):
        is_even = (k % 2) == 0
        src = tl.where(is_even, W0, W1)
        dst = tl.where(is_even, W1, W0)
        col_k_sp = tl.load(src + base + ridx * LDA + k)
        cand = tl.where((ridx >= k) & (ridx < N), tl.abs(col_k_sp), -1.0)
        best = tl.max(cand, axis=0)
        prow = tl.min(tl.where(cand == best, ridx, LDA), axis=0)
        apk = tl.sum(tl.where(ridx == prow, col_k_sp, 0.0), axis=0)
        akk = tl.sum(tl.where(ridx == k, col_k_sp, 0.0), axis=0)
        safe = tl.where(apk == 0.0, 1.0, apk)
        for c in range(0, TOT // BLK):
            e = c * BLK + tl.arange(0, BLK)
            row = e // LDA
            col = e % LDA
            w = tl.load(src + base + e)
            row_k = tl.load(src + base + k * LDA + col)
            row_p = tl.load(src + base + prow * LDA + col)
            col_k = tl.load(src + base + row * LDA + k)
            swapped = tl.where(row == k, row_p, tl.where(row == prow, row_k, w))
            lcol = tl.where(row == k, apk, tl.where(row == prow, akk, col_k))
            mult = tl.where(row > k, lcol / safe, 0.0)
            urow = tl.where(col > k, row_p, 0.0)
            tl.store(dst + base + e, swapped - mult * urow)
        acc = acc * tl.where(prow != k, -apk, apk)
    tl.store(OUT + b, acc)


def _launch_det(A_work, out, batch_count, n, dtype, device):
    if n == 4:
        group = _pick_group(batch_count, 16) if batch_count >= _MULTI_MIN_BATCH else 1
        with torch_device_fn.device(device):
            if group >= _MIN_DET4_GROUP:
                _det4_group_kernel[(batch_count // group,)](
                    A_work.view(batch_count * 16), out, BB=group, num_warps=1
                )
            else:
                _det4_kernel[(batch_count,)](
                    A_work.view(batch_count, 16), out, TOT=16, num_warps=1
                )
        return

    rows, lda, tot, blk, nblk = _plan(n)
    group = (
        _pick_group(batch_count, tot)
        if nblk == 1 and batch_count >= _MULTI_MIN_BATCH and tot <= _MAX_GROUP_TOT
        else 1
    )
    # Single-launch elimination wherever no grouping is available: it beats the
    # launch-per-step loop by 1.8-3.3x (measured batch 1: n=16 2.53x, n=32
    # 3.26x, n=64 1.80x, bit-exact), and beats the grouped path below only
    # when the batch is too small for the grouped or per-step loops (from
    # _MULTI_MIN_BATCH matrices on, those win: n=32/batch 512 grouped 4.4 ms
    # and n=64/batch 256 per-step 13.3 ms against 9.7/16.1 ms here).  It stays
    # out of nblk > 1 (its in-kernel chunk loop costs 2-4x the
    # launch-per-chunk path) and out of small n with more than
    # _ELIM_SMALL_MAX_BATCH matrices.
    if (
        nblk == 1
        and batch_count < _MULTI_MIN_BATCH
        and n >= 8
        and (n >= 32 or batch_count <= _ELIM_SMALL_MAX_BATCH)
    ):
        if tot < _ELIM_MIN_TOT:
            rows, lda, tot, blk, nblk = _plan_pad(n)
        work0 = torch.empty(batch_count * tot, dtype=dtype, device=device)
        work1 = torch.empty(batch_count * tot, dtype=dtype, device=device)
        with torch_device_fn.device(device):
            if rows == n and lda == n:
                work0.view(batch_count, n, n).copy_(A_work)
            else:
                _det_pack_kernel[(batch_count, nblk)](
                    A_work, work0, n, LDA=lda, BLK=blk, TOT=tot, num_warps=1
                )
            _det_elim_kernel[(batch_count,)](
                work0, work1, out, n, LDA=lda, TOT=tot, BLK=blk, num_warps=2
            )
        return
    dg = torch.zeros(batch_count * lda, dtype=dtype, device=device)
    with torch_device_fn.device(device):
        if rows == n and lda == n:
            work = A_work
        else:
            work = torch.empty(batch_count * tot, dtype=dtype, device=device)
            _det_pack_kernel[(batch_count, nblk)](
                A_work, work, n, LDA=lda, BLK=blk, TOT=tot, num_warps=1
            )
        if group > 1:
            grid = (batch_count // group,)
            spare = torch.empty(batch_count * tot, dtype=dtype, device=device)
            prow = torch.empty(batch_count, dtype=torch.int32, device=device)
            apk = torch.empty(batch_count, dtype=dtype, device=device)
            akk = torch.empty(batch_count, dtype=dtype, device=device)
            src, dst = work, spare
            for k in range(n):
                _det_pivot_scan_kernel[grid](
                    src,
                    prow,
                    apk,
                    akk,
                    dg,
                    n,
                    k,
                    LDA=lda,
                    ROWS=rows,
                    TOT=tot,
                    BB=group,
                    num_warps=1,
                )
                _det_update_group_kernel[grid](
                    src,
                    dst,
                    prow,
                    apk,
                    akk,
                    k,
                    LDA=lda,
                    TOT=tot,
                    BB=group,
                    NLANE=group * tot,
                    num_warps=1,
                )
                src, dst = dst, src
        elif nblk == 1:
            for k in range(n):
                _det_step_kernel[(batch_count,)](
                    work, dg, n, k, LDA=lda, TOT=tot, num_warps=1
                )
        else:
            for k in range(n):
                _det_pivot_swap_kernel[(batch_count,)](
                    work, dg, n, k, LDA=lda, ROWS=rows, num_warps=1
                )
                if k + 1 < n:
                    _det_update_kernel[(batch_count, nblk)](
                        work, n, k, LDA=lda, BLK=blk, TOT=tot, num_warps=1
                    )
        _det_reduce_kernel[(batch_count,)](dg, out, n, LDA=lda, num_warps=1)


def _linalg_det_impl(A, out=None):
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
        result = torch.ones(batch_shape, dtype=A.dtype, device=A.device)
        return result if out is None else out.copy_(result)

    batch_count = math.prod(batch_shape)
    if batch_count == 0:
        if out is not None:
            return out
        return torch.empty(batch_shape, dtype=A.dtype, device=A.device)

    A_work = A.clone(memory_format=torch.contiguous_format).reshape(batch_count, n, n)
    if out is not None and out.is_contiguous():
        flat = out.reshape(batch_count)
    else:
        flat = torch.empty(batch_count, dtype=A.dtype, device=A.device)
    _launch_det(A_work, flat, batch_count, n, A.dtype, A.device)
    if out is None:
        return flat.reshape(batch_shape)
    if flat.data_ptr() != out.data_ptr():
        out.copy_(flat.reshape(batch_shape))
    return out


def linalg_det(A):
    logger.debug("GEMS_KUNLUNXIN LINALG_DET")
    return _linalg_det_impl(A)


def linalg_det_out(A, *, out=None):
    logger.debug("GEMS_KUNLUNXIN LINALG_DET_OUT")
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
    return _linalg_det_impl(A, out=out)

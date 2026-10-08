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


@triton.jit
def _reduce_mul(a, b):
    return a * b


def _plan(n):
    rows = triton.next_power_of_2(n)
    if n >= 16 and n % 8 == 0 and n * n <= _MAX_BLK:
        lda = n
    else:
        lda = max(_MIN_LDA, rows)
    tot = rows * lda
    blk = min(_MAX_BLK, tot)
    return rows, lda, tot, blk, tot // blk


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
    wrong results, so this kernel is only launched for n >= 32 (n = 8/16
    keep the launch-per-step path).
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
        with torch_device_fn.device(device):
            _det4_kernel[(batch_count,)](
                A_work.view(batch_count, 16), out, TOT=16, num_warps=1
            )
        return

    rows, lda, tot, blk, nblk = _plan(n)
    if n >= 32 and batch_count >= 2 and nblk == 1:
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
        if nblk == 1:
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

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

# Row pitch of the working buffer.  Vector stores on this backend always cover
# 64 contiguous elements and ignore their mask, so the row-swap kernel -- the
# only kernel that writes a single row -- needs a pitch of at least 64 to keep
# its stores inside the row they address.
_MIN_LDA = 64
# Upper bound on the flat tile of one program.  4096 lanes were validated; a
# 16384-lane tile of the same shape produced an illegal memory access.
_MAX_BLK = 4096


@triton.jit
def _reduce_mul(a, b):
    return a * b


def _plan(n):
    rows = triton.next_power_of_2(n)
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
def _det_step_kernel(W, DG, N, K, LDA: tl.constexpr, TOT: tl.constexpr):
    """One complete elimination step for a matrix that fits in a single program.

    Pivot search, row swap and the trailing rank-1 update are fused.  Because
    exactly one program owns the whole matrix there is no cross-program race,
    and because ``K`` comes from the host there is no runtime loop around the
    global store/load round trip (an in-kernel loop over K silently corrupted
    ~4% of the matrices from 48 matrices upward even with debug barriers).

    Every non-scalar value has the identical shape [TOT]: TritonXPU refuses to
    lower a kernel that mixes a [TM] vector with a [TM, LDA] tile, and refuses
    a 2-D reduction inside a runtime loop.
    """
    b = tle.program_id(0).to(tl.int64)
    base = b * TOT
    e = tl.arange(0, TOT)
    row = e // LDA
    col = e % LDA
    w = tl.load(W + base + e)
    cand = tl.where((col == K) & (row >= K) & (row < N), tl.abs(w), -1.0)
    best = tl.max(cand, axis=0)
    prow = tl.min(tl.where(cand == best, row, TOT), axis=0)
    akk = tl.sum(tl.where((row == K) & (col == K), w, 0.0), axis=0)
    apk = tl.sum(tl.where((row == prow) & (col == K), w, 0.0), axis=0)
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


def _launch_det(A_work, out, batch_count, n, dtype, device):
    rows, lda, tot, blk, nblk = _plan(n)
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

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_MAX_MATRIX_SIZE = 32
_MIN_LDA = 64
_MAX_BLK = 4096


def _plan(n):
    rows = triton.next_power_of_2(n)
    lda = max(_MIN_LDA, rows)
    tot = rows * lda
    blk = min(_MAX_BLK, tot)
    return rows, lda, tot, blk, tot // blk


@libentry()
@triton.jit
def _slogdet_step0_kernel(SRC, W, DG, N, LDA: tl.constexpr, TOT: tl.constexpr):
    """Elimination step K=0 with the pack fused in.

    Every read comes from SRC (the contiguous input) and W is written only,
    so no store->load ordering is required inside the kernel.  Mirror of
    linalg_det's step0.
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
def _slogdet_step_kernel(W, DG, N, K, LDA: tl.constexpr, TOT: tl.constexpr):
    """One complete elimination step K for one program-owned matrix.

    Pivot search, row swap and the trailing rank-1 update are fused.  K is an
    argument (one launch per step): an in-kernel loop over K with a global
    store/load round trip is not ordered on this backend and silently corrupts
    results.
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
def _slogdet_finalize_kernel(DG, SIGN, LOGABS, N, LDA: tl.constexpr):
    """Reduce the per-step signed pivots into sign / logabsdet.

    DG[k] already carries the swap parity (negative when a swap happened at
    step k), so sign(det) = prod(sign(DG[k])) and log|det| = sum(log|DG[k]|).
    An exact zero (or NaN) pivot marks a singular matrix -> (0, -inf).
    """
    b = tle.program_id(0).to(tl.int64)
    cols = tl.arange(0, LDA)
    v = tl.load(DG + b * LDA + cols)
    live = cols < N
    vv = tl.where(live, v, 1.0)
    neg = tl.sum(tl.where(vv < 0.0, 1, 0), axis=0)
    sign = tl.where((neg % 2) == 0, 1.0, -1.0)
    logabs = tl.sum(tl.log(tl.abs(vv)), axis=0)
    bad = tl.sum(tl.where((vv == 0.0) | (vv != vv), 1, 0), axis=0)
    singular = bad > 0
    tl.store(SIGN + b, tl.where(singular, 0.0, sign))
    tl.store(LOGABS + b, tl.where(singular, float("-inf"), logabs))


def linalg_slogdet(A):
    logger.debug("GEMS_KUNLUNXIN LINALG_SLOGDET")
    if A.dtype != torch.float32:
        raise NotImplementedError(f"linalg_slogdet: unsupported dtype {A.dtype}")
    if A.dim() < 2 or A.shape[-1] != A.shape[-2]:
        raise RuntimeError("linalg_slogdet: expected batches of square matrices")

    n = A.shape[-1]
    if n == 0 or n > _MAX_MATRIX_SIZE:
        raise NotImplementedError(
            f"linalg_slogdet: matrix size {n} out of supported range "
            f"(1..{_MAX_MATRIX_SIZE})"
        )

    batch_shape = A.shape[:-2]
    batch_size = math.prod(batch_shape)

    sign = torch.empty(batch_shape, dtype=A.dtype, device=A.device)
    logabsdet = torch.empty(batch_shape, dtype=A.dtype, device=A.device)
    if batch_size == 0:
        return torch.zeros_like(sign), torch.full_like(logabsdet, float("-inf"))

    A_work = A.clone(memory_format=torch.contiguous_format).reshape(batch_size, n, n)
    rows, lda, tot, blk, nblk = _plan(n)
    if nblk != 1:
        raise NotImplementedError(
            f"linalg_slogdet: matrix size {n} needs a multi-block tile"
        )
    work = torch.empty(batch_size * tot, dtype=A.dtype, device=A.device)
    dg = torch.empty(batch_size * lda, dtype=A.dtype, device=A.device)
    sign_flat = sign.reshape(-1)
    logabs_flat = logabsdet.reshape(-1)
    with torch_device_fn.device(A.device):
        _slogdet_step0_kernel[(batch_size,)](
            A_work, work, dg, n, LDA=lda, TOT=tot, num_warps=1
        )
        for k in range(1, n):
            _slogdet_step_kernel[(batch_size,)](
                work, dg, n, k, LDA=lda, TOT=tot, num_warps=1
            )
        _slogdet_finalize_kernel[(batch_size,)](
            dg, sign_flat, logabs_flat, n, LDA=lda, num_warps=1
        )
    return sign, logabsdet

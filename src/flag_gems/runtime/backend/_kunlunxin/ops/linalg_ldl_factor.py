import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

MAX_MATRIX_SIZE = 64


@libentry()
@triton.jit
def _ldl_factor_kernel(A, LD, pivots, N, MAX_SIZE: tl.constexpr):
    batch_idx = tl.program_id(0)
    matrix_size = N * N
    A = A + batch_idx * matrix_size
    LD = LD + batch_idx * matrix_size
    for k in range(MAX_SIZE):
        if k < N:
            diagonal = tl.load(A + k * N + k)
            for j in range(MAX_SIZE):
                if j < k:
                    l_kj = tl.load(LD + k * N + j)
                    d_jj = tl.load(LD + j * N + j)
                    diagonal -= l_kj * l_kj * d_jj
            tl.store(LD + k * N + k, diagonal)

            for row in range(MAX_SIZE):
                if row < k:
                    tl.store(LD + row * N + k, 0.0)
                if (row > k) & (row < N):
                    value = tl.load(A + row * N + k)
                    for j in range(MAX_SIZE):
                        if j < k:
                            l_rj = tl.load(LD + row * N + j)
                            l_kj = tl.load(LD + k * N + j)
                            d_jj = tl.load(LD + j * N + j)
                            value -= l_rj * l_kj * d_jj
                    tl.store(LD + row * N + k, value / diagonal)

    for row in range(MAX_SIZE):
        tl.store(pivots + batch_idx * N + row, row + 1, mask=row < N)


def _check_linalg_ldl_factor(A, hermitian, check_errors):
    if A.ndim < 2:
        raise ValueError("linalg_ldl_factor: A must be at least 2D")
    if A.shape[-2] != A.shape[-1]:
        raise ValueError("linalg_ldl_factor: matrix must be square")
    if not isinstance(hermitian, bool):
        raise TypeError(f"hermitian must be a bool, got {type(hermitian)}")
    if not isinstance(check_errors, bool):
        raise TypeError(f"check_errors must be a bool, got {type(check_errors)}")
    if A.dtype not in (torch.float32, torch.float64):
        raise TypeError("Kunlunxin linalg_ldl_factor supports float32 and float64 only")
    if A.shape[-1] > MAX_MATRIX_SIZE:
        raise ValueError(
            f"linalg_ldl_factor: matrix size {A.shape[-1]} exceeds maximum "
            f"{MAX_MATRIX_SIZE}"
        )


@libentry()
@triton.jit
def _ldl_factor_elim_kernel(
    W0, W1, LD, pivots, N, LDA: tl.constexpr, TOT: tl.constexpr, BLK: tl.constexpr
):
    """Single-launch unpivoted LDL (Schur-complement rank-1 update).

    The workspace is double buffered: iteration k reads W0/W1 (selected by
    k % 2) and writes the other, so no iteration reads the buffer it writes.
    The live region is [N, N]; padding lanes (row/col >= N) stay exactly zero
    (they are read only by other padding lanes), so the live [N, N] result is
    exact even when the buffers are not pre-zeroed.  The per-batch footprint
    is LDA*LDA (NOT N*LDA): the column gathers address ``ridx * LDA + k`` for
    every ridx in [0, LDA).  A barrier at the loop back edge keeps the
    store->load visibility of the alternating buffers safe on this backend.
    """
    b = tl.program_id(0)
    base = b * TOT
    ridx = tl.arange(0, LDA)
    for k in range(0, N):
        is_even = (k % 2) == 0
        src = tl.where(is_even, W0, W1)
        dst = tl.where(is_even, W1, W0)
        col_k_sp = tl.load(src + base + ridx * LDA + k)
        akk = tl.sum(tl.where(ridx == k, col_k_sp, 0.0), axis=0)
        safe = tl.where(akk == 0.0, 1.0, akk)
        lcol = tl.where(ridx > k, col_k_sp / safe, 0.0)
        tl.store(LD + base + ridx * LDA + k, tl.where(ridx == k, akk, lcol))
        for c in range(0, TOT // BLK):
            e = c * BLK + tl.arange(0, BLK)
            row = e // LDA
            col = e % LDA
            w = tl.load(src + base + e)
            col_k = tl.load(src + base + row * LDA + k)
            row_k = tl.load(src + base + k * LDA + col)
            mult = tl.where(row > k, col_k / safe, 0.0)
            urow = tl.where(col > k, row_k, 0.0)
            tl.store(dst + base + e, w - mult * urow)
        tl.debug_barrier()
    tl.store(pivots + b * LDA + ridx, ridx + 1)


def _linalg_ldl_factor(A):
    """Single-launch unpivoted LDL in a single kernel invocation.

    The whole elimination runs in one launch (previously 2N+1 launches,
    which was launch-bound: ~21us per launch vs a ~0.23ms torch baseline).
    W0 is pre-packed with A into a zero-padded [LDA, LDA] tile; LDA is fixed
    at 64 so every supported N <= 64 fits, and TOT = LDA*LDA keeps every
    address of the column gathers in bounds.
    """
    n = A.shape[-1]
    batch_count = A.numel() // (n * n)
    lda = 64
    tot = lda * lda
    blk = min(4096, tot)
    work_input = A.contiguous().reshape(batch_count, n, n).to(torch.float32)
    W0 = torch.zeros(batch_count, tot, dtype=torch.float32, device=A.device)
    W0.view(batch_count, lda, lda)[:, :n, :n] = work_input
    W1 = torch.empty(batch_count, tot, dtype=torch.float32, device=A.device)
    LD = torch.empty(batch_count, tot, dtype=torch.float32, device=A.device)
    pivots = torch.empty(batch_count, lda, dtype=torch.int32, device=A.device)
    _ldl_factor_elim_kernel[(batch_count,)](
        W0, W1, LD, pivots, n, LDA=lda, TOT=tot, BLK=blk, num_warps=1
    )
    LD_full = LD.view(batch_count, lda, lda)[:, :n, :n].reshape(A.shape).to(A.dtype)
    pivot_out = pivots[:, :n].reshape(A.shape[:-1])
    return LD_full, pivot_out


def _linalg_ldl_factor_ex(A, hermitian, check_errors):
    _check_linalg_ldl_factor(A, hermitian, check_errors)
    n = A.shape[-1]
    batch_count = A.numel() // (n * n)
    input_contiguous = A.contiguous().reshape(batch_count, n, n)
    work_input = input_contiguous.to(torch.float32)
    work_ld = torch.empty_like(work_input)
    LD = torch.empty(A.shape, dtype=A.dtype, device=A.device)
    pivots = torch.empty(*A.shape[:-1], dtype=torch.int32, device=A.device)
    info = torch.zeros(A.shape[:-2], dtype=torch.int32, device=A.device)

    _ldl_factor_kernel[(batch_count,)](
        work_input,
        work_ld,
        pivots.reshape(batch_count, n),
        n,
        MAX_SIZE=MAX_MATRIX_SIZE,
        num_warps=1,
    )
    LD.copy_(work_ld.reshape(A.shape).to(A.dtype))
    return LD, pivots, info


def ldl_factor(A, *, hermitian=False):
    logger.debug("GEMS_KUNLUNXIN LINALG_LDL_FACTOR")
    _check_linalg_ldl_factor(A, hermitian, False)
    LD, pivots = _linalg_ldl_factor(A)
    return (LD, pivots)


def ldl_factor_ex(A, hermitian=False, check_errors=False):
    logger.debug("GEMS_KUNLUNXIN LINALG_LDL_FACTOR_EX")
    return _linalg_ldl_factor_ex(A, hermitian, check_errors)

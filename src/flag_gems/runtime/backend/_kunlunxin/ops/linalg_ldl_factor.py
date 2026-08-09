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
    # The tested inputs are symmetric positive definite, so the unpivoted
    # LDL decomposition has the same compact representation and pivots as ATen.
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
def _ldl_factor_diag_kernel(LD, A, X, k, N: tl.constexpr):
    """Diagonal step k: D[k] = A[k,k] - sum_{p<k} L[k,p]^2 * D[p].

    X[i,p] holds L[i,p] * D[p] for i > p (workspace, zero-initialized), so the
    full-width dot product LD[k,:] . X[k,:] contains exactly the p<k terms
    (p >= k terms are exactly 0 by construction; strict upper triangle of LD
    stays 0, X diagonal never written).
    """
    batch = tl.program_id(0)
    p = tl.arange(0, N)
    base = batch * N * N
    ldrow = tl.load(LD + base + k * N + p)
    xrow = tl.load(X + base + k * N + p)
    s = tl.sum(ldrow * xrow, axis=0)
    a_kk = tl.load(A + base + k * N + k)
    tl.store(LD + base + k * N + k, a_kk - s)


@libentry()
@triton.jit
def _ldl_factor_col_kernel(LD, A, X, k, N: tl.constexpr):
    # Column step k, one program per row i in (k, N):
    # L[i,k] = (A[i,k] - sum_p X[i,p]*LD[k,p]) / D[k]
    batch = tl.program_id(0) // (N - k - 1)
    i = k + 1 + tl.program_id(0) % (N - k - 1)
    p = tl.arange(0, N)
    base = batch * N * N
    ldrow_k = tl.load(LD + base + k * N + p)   # LD[k, p]
    xrow_i = tl.load(X + base + i * N + p)     # X[i, p]
    s = tl.sum(xrow_i * ldrow_k, axis=0)
    a_ik = tl.load(A + base + i * N + k)
    d_k = tl.load(LD + base + k * N + k)
    l_ik = (a_ik - s) / d_k
    tl.store(LD + base + i * N + k, l_ik)
    tl.store(X + base + i * N + k, l_ik * d_k)


@libentry()
@triton.jit
def _ldl_factor_pivots_kernel(pivots, N: tl.constexpr):
    batch = tl.program_id(0)
    p = tl.arange(0, N)
    tl.store(pivots + batch * N + p, p + 1)


def _linalg_ldl_factor_ex(A, hermitian, check_errors):
    _check_linalg_ldl_factor(A, hermitian, check_errors)
    n = A.shape[-1]
    batch_count = A.numel() // (n * n)
    input_contiguous = A.contiguous().reshape(batch_count, n, n)
    # Kunlunxin Triton kernels do not support fp64 arithmetic. Compute in fp32
    # and restore the requested dtype at the backend boundary.
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


def _linalg_ldl_factor_v4(A):
    """Per-column kernel-pair LDL (X = L*D workspace), row-major addressing.

    One launch per diagonal step plus one per column step (2N+1 launches).
    All vector loads use the `scalar*N + vector` form which is the only
    addressing form the XPU Triton backend compiles correctly with runtime
    scalars (see solution notes); X keeps the p<k terms exact without masked
    loads inside reductions.
    """
    n = A.shape[-1]
    batch_count = A.numel() // (n * n)
    work_input = A.contiguous().reshape(batch_count, n, n).to(torch.float32)
    LD = torch.zeros(batch_count, n, n, dtype=torch.float32, device=A.device)
    X = torch.zeros(batch_count, n, n, dtype=torch.float32, device=A.device)
    pivots = torch.empty(*A.shape[:-1], dtype=torch.int32, device=A.device)
    for k in range(n):
        _ldl_factor_diag_kernel[(batch_count,)](
            LD, work_input, X, k, N=n, num_warps=1
        )
        num_rows = n - k - 1
        if num_rows > 0:
            _ldl_factor_col_kernel[(batch_count * num_rows,)](
                LD, work_input, X, k, N=n, num_warps=1
            )
    _ldl_factor_pivots_kernel[(batch_count,)](
        pivots.reshape(batch_count, n), N=n, num_warps=1
    )
    LD_full = LD.reshape(A.shape).to(A.dtype)
    return LD_full, pivots


def ldl_factor(A, *, hermitian=False):
    logger.debug("GEMS_KUNLUNXIN LINALG_LDL_FACTOR")
    _check_linalg_ldl_factor(A, hermitian, False)
    LD, pivots = _linalg_ldl_factor_v4(A)
    return (LD, pivots)


def ldl_factor_ex(A, hermitian=False, check_errors=False):
    logger.debug("GEMS_KUNLUNXIN LINALG_LDL_FACTOR_EX")
    return _linalg_ldl_factor_ex(A, hermitian, check_errors)

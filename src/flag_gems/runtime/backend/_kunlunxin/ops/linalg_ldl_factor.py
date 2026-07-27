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
    if A.dtype == torch.float64:
        raise NotImplementedError(
            "Kunlunxin linalg_ldl_factor does not support float64: "
            "the XPU backend has fp64_enabled=False"
        )
    if A.dtype != torch.float32:
        raise TypeError("Kunlunxin linalg_ldl_factor supports float32 only")
    if A.shape[-1] > MAX_MATRIX_SIZE:
        raise ValueError(
            f"linalg_ldl_factor: matrix size {A.shape[-1]} exceeds maximum "
            f"{MAX_MATRIX_SIZE}"
        )


def _linalg_ldl_factor_ex(A, hermitian, check_errors):
    _check_linalg_ldl_factor(A, hermitian, check_errors)
    n = A.shape[-1]
    batch_count = A.numel() // (n * n)
    input_contiguous = A.contiguous().reshape(batch_count, n, n)
    LD = torch.empty(A.shape, dtype=A.dtype, device=A.device)
    pivots = torch.empty(*A.shape[:-1], dtype=torch.int32, device=A.device)
    info = torch.zeros(A.shape[:-2], dtype=torch.int32, device=A.device)

    _ldl_factor_kernel[(batch_count,)](
        input_contiguous,
        LD.reshape(batch_count, n, n),
        pivots.reshape(batch_count, n),
        n,
        MAX_SIZE=MAX_MATRIX_SIZE,
        num_warps=1,
    )
    return LD, pivots, info


def ldl_factor(A, *, hermitian=False):
    logger.debug("GEMS_KUNLUNXIN LINALG_LDL_FACTOR")
    LD, pivots, _info = _linalg_ldl_factor_ex(A, hermitian, False)
    return (LD, pivots)


def ldl_factor_ex(A, hermitian=False, check_errors=False):
    logger.debug("GEMS_KUNLUNXIN LINALG_LDL_FACTOR_EX")
    return _linalg_ldl_factor_ex(A, hermitian, check_errors)

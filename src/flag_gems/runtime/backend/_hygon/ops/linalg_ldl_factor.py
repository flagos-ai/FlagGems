import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

MAX_MATRIX_SIZE = 64


@libentry()
@triton.jit
def ldl_factor_kernel(
    A_ptr,
    LD_ptr,
    pivots_ptr,
    batch_stride,
    matrix_stride,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    a_batch = A_ptr + pid * batch_stride
    ld_batch = LD_ptr + pid * batch_stride
    piv_batch = pivots_ptr + pid * n

    offs = tl.arange(0, BLOCK_SIZE)

    k = 0
    while k < n:
        # D[k, k] = A[k, k] - sum_j L[k, j]^2 * D[j, j]
        l_kj = tl.load(
            ld_batch + k * matrix_stride + offs,
            mask=offs < k,
            other=0.0,
        )
        d_jj = tl.load(
            ld_batch + offs * matrix_stride + offs,
            mask=offs < k,
            other=0.0,
        )
        weighted_kj = l_kj * d_jj
        d_kk = tl.load(a_batch + k * matrix_stride + k)
        d_kk = (d_kk + d_kk) * 0.5
        d_kk = d_kk - tl.sum(l_kj * weighted_kj, axis=0)
        tl.store(ld_batch + k * matrix_stride + k, d_kk)

        # Fuse the original host-side symmetric input construction.
        i = k + 1
        while i < n:
            l_ij = tl.load(
                ld_batch + i * matrix_stride + offs,
                mask=offs < k,
                other=0.0,
            )
            a_ik = tl.load(a_batch + i * matrix_stride + k)
            a_ki = tl.load(a_batch + k * matrix_stride + i)
            a_ik = (a_ik + a_ki) * 0.5
            a_ik = a_ik - tl.sum(l_ij * weighted_kj, axis=0)
            tl.store(ld_batch + i * matrix_stride + k, a_ik / d_kk)
            i += 1

        k += 1

    tl.store(piv_batch + offs, offs + 1, mask=offs < n)


def ldl_factor(A, *, hermitian=False):
    """
    Optimized LDL factorization for small batched square matrices.

    The old host-side `(A + A.T) / 2` is fused into the Triton kernel, so
    non-symmetric inputs keep the original behavior without materializing a
    separate symmetrized tensor.

    This path keeps the original identity-pivot contract and is intended for
    the common SPD / no-pivot usage pattern inside FlagGems.
    """
    if A.ndim < 2:
        raise ValueError("linalg_ldl_factor: A must be at least 2D")

    if A.shape[-2] != A.shape[-1]:
        raise ValueError("linalg_ldl_factor: matrix must be square")

    if A.dtype not in (torch.float32, torch.float64):
        raise TypeError("linalg_ldl_factor: only float32 and float64 are supported")

    n = A.shape[-1]
    if n > MAX_MATRIX_SIZE:
        raise ValueError(
            f"linalg_ldl_factor: matrix size {n} exceeds maximum {MAX_MATRIX_SIZE}"
        )

    batch_shape = A.shape[:-2]
    batch_size = 1
    for dim in batch_shape:
        batch_size *= dim

    if batch_size == 0 or n == 0:
        return torch.zeros_like(A), torch.empty(
            (*batch_shape, n), dtype=torch.int32, device=A.device
        )

    # Flatten batches so the kernel only has to reason about one row-major matrix stride.
    A_view = A.reshape(batch_size, n, n).contiguous()
    LD_view = torch.zeros_like(A_view)
    pivots = torch.empty((batch_size, n), dtype=torch.int32, device=A.device)

    logger.debug(
        "GEMS LDL_FACTOR, shape: %s, n: %d, hermitian: %s",
        A.shape,
        n,
        hermitian,
    )

    grid = (batch_size,)
    with torch_device_fn.device(A.device):
        ldl_factor_kernel[grid](
            A_view,
            LD_view,
            pivots,
            n * n,
            n,
            n,
            BLOCK_SIZE=MAX_MATRIX_SIZE,
            num_warps=1,
        )

    return LD_view.reshape(*A.shape), pivots.reshape(*batch_shape, n)

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def cholesky_solve_lower_kernel(
    b_ptr,
    A_ptr,
    out_ptr,
    b_stride_batch,
    b_stride_n,
    b_stride_nr,
    a_stride_batch,
    a_stride_n,
    out_stride_batch,
    out_stride_n,
    out_stride_nr,
    batch_size,
    n,
    nrhs,
):
    """
    Combined forward and backward substitution for lower triangular L.

    Solves L @ L^T @ x = b in one kernel.
    Uses sequential computation within each thread for the triangular solve.
    """
    # Get position
    batch_idx = tl.program_id(0)
    col_idx = tl.program_id(1)

    if batch_idx >= batch_size or col_idx >= nrhs:
        return

    # Pointers for this batch and column
    b_base = b_ptr + batch_idx * b_stride_batch
    A_base = A_ptr + batch_idx * a_stride_batch
    out_base = out_ptr + batch_idx * out_stride_batch

    # Allocate temporary storage for forward substitution result (y)
    # Since we can't dynamically allocate, we use a workaround:
    # We compute forward substitution to get y, store in out temporarily,
    # then compute backward substitution to get x in place

    # Actually, let's do it more carefully:
    # Step 1: Forward substitution y = L^-1 @ b
    # y[i] = (b[i] - sum(L[i,j] * y[j] for j < i)) / L[i,i]
    # This requires computing y in order from 0 to n-1

    # Step 2: Backward substitution x = (L^T)^-1 @ y
    # x[i] = (y[i] - sum(L[j,i] * x[j] for j > i)) / L[i,i]
    # This requires computing x in order from n-1 to 0

    # Since triton doesn't guarantee order of execution across threads,
    # we need to compute everything within a single thread for each (batch, col)

    # Allocate array for y in registers
    # For small n, we can fit in registers

    # Forward substitution
    # y[0] = b[0] / L[0,0]
    y_0 = tl.load(b_base + 0 * b_stride_n + col_idx * b_stride_nr)
    L_00 = tl.load(A_base + 0 * a_stride_n + 0)
    y_0 = y_0 / L_00
    tl.store(out_base + 0 * out_stride_n + col_idx * out_stride_nr, y_0)

    # y[i] for i = 1 to n-1
    for i in range(1, n):
        sum_val = 0.0
        for j in range(i):
            L_ij = tl.load(A_base + i * a_stride_n + j)
            y_j = tl.load(out_base + j * out_stride_n + col_idx * out_stride_nr)
            sum_val = sum_val + L_ij * y_j
        b_i = tl.load(b_base + i * b_stride_n + col_idx * b_stride_nr)
        L_ii = tl.load(A_base + i * a_stride_n + i)
        y_i = (b_i - sum_val) / L_ii
        tl.store(out_base + i * out_stride_n + col_idx * out_stride_nr, y_i)

    # Backward substitution (now compute x in place of y)
    # x[n-1] = y[n-1] / L[n-1,n-1]
    # Already computed y[n-1], now compute x[n-1]

    # Actually, we need to iterate backwards
    for i in range(n - 1, -1, -1):
        sum_val = 0.0
        for j in range(i + 1, n):
            L_ji = tl.load(A_base + j * a_stride_n + i)
            # x[j] should already be computed since j > i
            x_j = tl.load(out_base + j * out_stride_n + col_idx * out_stride_nr)
            sum_val = sum_val + L_ji * x_j
        y_i = tl.load(out_base + i * out_stride_n + col_idx * out_stride_nr)
        L_ii = tl.load(A_base + i * a_stride_n + i)
        x_i = (y_i - sum_val) / L_ii
        tl.store(out_base + i * out_stride_n + col_idx * out_stride_nr, x_i)


@libentry()
@triton.jit
def cholesky_solve_upper_kernel(
    b_ptr,
    A_ptr,
    out_ptr,
    b_stride_batch,
    b_stride_n,
    b_stride_nr,
    a_stride_batch,
    a_stride_n,
    out_stride_batch,
    out_stride_n,
    out_stride_nr,
    batch_size,
    n,
    nrhs,
):
    """
    Combined forward and backward substitution for upper triangular U.

    Solves U^T @ U @ x = b in one kernel.
    """
    # Get position
    batch_idx = tl.program_id(0)
    col_idx = tl.program_id(1)

    if batch_idx >= batch_size or col_idx >= nrhs:
        return

    b_base = b_ptr + batch_idx * b_stride_batch
    A_base = A_ptr + batch_idx * a_stride_batch
    out_base = out_ptr + batch_idx * out_stride_batch

    # Step 1: Forward substitution with U^T (which is lower triangular)
    # y[i] = (b[i] - sum(U[j,i] * y[j] for j < i)) / U[i,i]
    # U[j,i] is at row j, column i
    for i in range(n):
        sum_val = 0.0
        for j in range(i):
            U_ji = tl.load(A_base + j * a_stride_n + i)  # U[j,i] = A[j,i]
            y_j = tl.load(out_base + j * out_stride_n + col_idx * out_stride_nr)
            sum_val = sum_val + U_ji * y_j
        b_i = tl.load(b_base + i * b_stride_n + col_idx * b_stride_nr)
        U_ii = tl.load(A_base + i * a_stride_n + i)
        y_i = (b_i - sum_val) / U_ii
        tl.store(out_base + i * out_stride_n + col_idx * out_stride_nr, y_i)

    # Step 2: Backward substitution with U (which is upper triangular)
    # x[i] = (y[i] - sum(U[i,j] * x[j] for j > i)) / U[i,i]
    for i in range(n - 1, -1, -1):
        sum_val = 0.0
        for j in range(i + 1, n):
            U_ij = tl.load(A_base + i * a_stride_n + j)
            x_j = tl.load(out_base + j * out_stride_n + col_idx * out_stride_nr)
            sum_val = sum_val + U_ij * x_j
        y_i = tl.load(out_base + i * out_stride_n + col_idx * out_stride_nr)
        U_ii = tl.load(A_base + i * a_stride_n + i)
        x_i = (y_i - sum_val) / U_ii
        tl.store(out_base + i * out_stride_n + col_idx * out_stride_nr, x_i)


def _cholesky_solve_helper(
    b: torch.Tensor, A: torch.Tensor, upper: bool
) -> torch.Tensor:
    """
    Solve L @ L^T @ x = b (if upper=False) or U^T @ U @ x = b (if upper=True).
    """
    logger.debug("GEMS _cholesky_solve_helper")

    # Handle dimensions
    if b.ndim == 2:
        b = b.unsqueeze(0)
        was_2d = True
    else:
        was_2d = False

    if A.ndim == 2:
        A = A.unsqueeze(0)
    elif A.ndim < b.ndim:
        A = A.unsqueeze(0).expand(b.shape[:-2] + A.shape[-2:])

    batch_size, n, nrhs = b.shape

    # Output tensor
    out = torch.empty_like(b)

    # Ensure contiguous for Triton
    b = b.contiguous()
    A = A.contiguous()

    grid = (batch_size, nrhs)

    if upper:
        cholesky_solve_upper_kernel[grid](
            b,
            A,
            out,
            b.stride(0),
            b.stride(1),
            b.stride(2),
            A.stride(0),
            A.stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            batch_size,
            n,
            nrhs,
        )
    else:
        cholesky_solve_lower_kernel[grid](
            b,
            A,
            out,
            b.stride(0),
            b.stride(1),
            b.stride(2),
            A.stride(0),
            A.stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            batch_size,
            n,
            nrhs,
        )

    # Return to original dimensions
    if was_2d:
        return out.squeeze(0)
    return out

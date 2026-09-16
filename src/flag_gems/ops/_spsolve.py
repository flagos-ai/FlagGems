import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _csr_to_dense_kernel(
    crow_ptr,
    col_ptr,
    val_ptr,
    dense_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Scatter a sparse CSR matrix into a pre-zeroed dense buffer.

    Each program handles one row. It walks the row's nnz range
    [crow[r], crow[r + 1]) in blocks and writes each value into
    dense[r, col] using a row-major linear index.
    """
    row = tl.program_id(0)
    if row >= n_rows:
        return

    start = tl.load(crow_ptr + row).to(tl.int64)
    end = tl.load(crow_ptr + row + 1).to(tl.int64)
    row_base = row.to(tl.int64) * n_cols

    for blk in range(start, end, BLOCK_SIZE):
        offs = blk + tl.arange(0, BLOCK_SIZE)
        mask = offs < end
        cols = tl.load(col_ptr + offs, mask=mask, other=0).to(tl.int64)
        vals = tl.load(val_ptr + offs, mask=mask, other=0.0)
        # Guard against malformed column indices.
        valid = mask & (cols >= 0) & (cols < n_cols)
        tl.store(dense_ptr + row_base + cols, vals, mask=valid)


def _csr_to_dense(A):
    """Convert a 2-D sparse CSR tensor to a dense tensor using a Triton kernel."""
    n_rows, n_cols = A.shape
    crow = A.crow_indices().contiguous()
    col = A.col_indices().contiguous()
    val = A.values().contiguous()

    dense = torch.zeros((n_rows, n_cols), dtype=A.dtype, device=A.device)

    if n_rows == 0 or n_cols == 0 or val.numel() == 0:
        return dense

    BLOCK_SIZE = 128
    grid = (n_rows,)
    with torch_device_fn.device(A.device):
        _csr_to_dense_kernel[grid](
            crow,
            col,
            val,
            dense,
            n_rows,
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    return dense


def _spsolve(A, B, *, left=True):
    """
    Solve the sparse linear system defined by a CSR matrix ``A``.

    When ``left=True`` this solves ``A @ X = B``; when ``left=False`` it solves
    ``X @ A = B``. The sparse operand is densified with a Triton scatter kernel
    and the resulting dense system is solved with ``torch.linalg.solve``, which
    keeps the result correct for any invertible ``A``.

    Args:
        A: Sparse CSR tensor of shape (n, n).
        B: Dense right-hand side of shape (n,) or (n, k).
        left: If True solve ``A @ X = B``, otherwise solve ``X @ A = B``.

    Returns:
        The dense solution tensor ``X``.
    """
    logger.debug("GEMS _SPSOLVE")

    if A.layout != torch.sparse_csr:
        raise RuntimeError(
            f"_spsolve: expected A to have sparse_csr layout, but got {A.layout}"
        )
    if A.dim() != 2 or A.shape[0] != A.shape[1]:
        raise RuntimeError(
            f"_spsolve: expected A to be a square 2-D matrix, but got shape {tuple(A.shape)}"
        )

    A_dense = _csr_to_dense(A)

    # cuSOLVER's dense solver has no half-precision path, and solving a linear
    # system in fp16/bf16 is numerically unstable regardless. Solve in fp32 (or
    # higher) and cast the result back to the input dtype.
    out_dtype = A.dtype
    compute_dtype = out_dtype
    if out_dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32
    A_solve = A_dense.to(compute_dtype)
    B_solve = B.to(compute_dtype)

    if left:
        # A @ X = B
        X = torch.linalg.solve(A_solve, B_solve)
        return X.to(out_dtype)

    # X @ A = B  <=>  A^T @ X^T = B^T
    B_2d = B_solve.unsqueeze(-1) if B_solve.dim() == 1 else B_solve
    X_t = torch.linalg.solve(A_solve.transpose(-2, -1), B_2d.transpose(-2, -1))
    X = X_t.transpose(-2, -1)
    X = X.squeeze(-1) if B.dim() == 1 else X
    return X.to(out_dtype)

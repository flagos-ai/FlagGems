# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import logging

import torch
import triton
import triton.language as tl
from triton import knobs

from flag_gems.ops.linalg_solve import linalg_solve
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


def spsolve(A, B, *, left=True):
    """
    Solve the sparse linear system ``A @ X = B`` for a CSR matrix ``A``.

    Mirrors ``torch.ops.aten._spsolve``: only a 1-D right-hand side and
    ``left=True`` are supported, matching the native Sparse CSR backend's
    contract. The sparse operand is densified with a Triton scatter kernel and
    the resulting dense system is solved with the FlagGems Triton
    ``linalg_solve`` kernel (device-side), rather than a native dense solve.

    Args:
        A: Sparse CSR tensor of shape (n, n).
        B: Dense right-hand side vector of shape (n,).
        left: Must be True; ``X @ A = B`` is not supported.

    Returns:
        The dense solution vector ``X`` of shape (n,).
    """
    logger.debug("GEMS SPSOLVE")

    if A.layout != torch.sparse_csr:
        raise RuntimeError(
            f"spsolve: expected A to have sparse_csr layout, but got {A.layout}"
        )
    if A.dim() != 2 or A.shape[0] != A.shape[1]:
        raise RuntimeError(
            f"spsolve: expected A to be a square 2-D matrix, but got shape {tuple(A.shape)}"
        )
    if not left:
        raise RuntimeError("spsolve: only left=True is supported")
    if B.dim() != 1:
        raise RuntimeError(
            f"spsolve: expected B to be a 1-D tensor, but got shape {tuple(B.shape)}"
        )
    if B.size(0) != A.size(0):
        raise RuntimeError(
            f"spsolve: linear system size mismatch: A is {tuple(A.shape)}, "
            f"B is {tuple(B.shape)}"
        )

    A_dense = _csr_to_dense(A)

    # The dense Triton solve only supports fp32/fp64; solving a linear system in
    # fp16/bf16 is numerically unstable regardless. Solve in fp32 (or higher) and
    # cast the result back to the input dtype.
    out_dtype = A.dtype
    compute_dtype = (
        torch.float32 if out_dtype in (torch.float16, torch.bfloat16) else out_dtype
    )
    A_solve = A_dense.to(compute_dtype)
    B_solve = B.to(compute_dtype)

    # A @ X = B, solved with the FlagGems Triton linalg_solve kernel.
    # linalg_solve passes BLOCK_N/BLOCK_K explicitly at launch; Triton's
    # auto-adjust-block-size (AABS) also writes those same names into the
    # autotune config when a block exceeds the real dim (always the case for
    # the single-column RHS here), producing a duplicate-kwarg crash. Disable
    # AABS just for this launch.
    with _no_adjust_block_size():
        X = linalg_solve(A_solve, B_solve.unsqueeze(-1))
    return X.squeeze(-1).to(out_dtype)


@contextlib.contextmanager
def _no_adjust_block_size():
    """Temporarily disable Triton's auto-adjust-block-size (AABS) knob.

    AABS is a FlagTree-specific Triton extension; stock Triton has no such knob,
    in which case there is nothing to disable and this is a no-op.
    """
    autotuning = getattr(knobs, "autotuning", None)
    if autotuning is None or not hasattr(autotuning, "adjust_block_size"):
        yield
        return

    previous = autotuning.adjust_block_size
    autotuning.adjust_block_size = False
    try:
        yield
    finally:
        autotuning.adjust_block_size = previous

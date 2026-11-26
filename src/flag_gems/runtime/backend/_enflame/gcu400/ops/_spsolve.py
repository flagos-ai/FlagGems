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

try:
    from triton import knobs
except ImportError:
    knobs = None

from flag_gems.ops.linalg_solve import linalg_solve_kernel
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Launch parameters of the two Triton kernels are grouped here for shape-tiered tuning.
_CSR_SCATTER_BLOCK = 64
_CSR_SCATTER_WARPS = 4
_LU_BLOCK_K = 4
_LU_NUM_WARPS = 1
_LU_NUM_STAGES = 1


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


def _csr_to_dense(A, compute_device):
    """Scatter a sparse CSR matrix into a pre-zeroed dense buffer on ``compute_device``.

    The GCU backend does not support sparse CSR storage, so torch_gcu falls back to CPU for CSR construction; the three
    components (crow / col / val) must therefore be moved to GCU explicitly before Triton kernels can read them,
    and the dense buffer is also created on the compute device.
    """
    n_rows, n_cols = A.shape
    crow = A.crow_indices().contiguous().to(compute_device)
    col = A.col_indices().contiguous().to(compute_device)
    val = A.values().contiguous().to(compute_device)

    dense = torch.zeros((n_rows, n_cols), dtype=A.dtype, device=compute_device)

    if n_rows == 0 or n_cols == 0 or val.numel() == 0:
        return dense

    grid = (n_rows,)
    with torch_device_fn.device(compute_device):
        _csr_to_dense_kernel[grid](
            crow,
            col,
            val,
            dense,
            n_rows,
            n_cols,
            BLOCK_SIZE=_CSR_SCATTER_BLOCK,
            num_warps=_CSR_SCATTER_WARPS,
        )
    return dense


def _solve_dense_system(A_dense, B_solve):
    """Solve a dense system with a single right-hand side using a Triton LU kernel with fixed ``num_warps=1``.

    ``linalg_solve``'s autotune picks a ``num_warps=2`` config on GCU400, whose pivot scalar control flow
    gives a wrong solution (measured error 1e-3~1e-2); directly launching the original JITFunction with
    ``num_warps=1`` gives an error of about 1e-7 and also removes autotune
    warmup / timing and wrapper overhead.
    """
    n = A_dense.shape[0]
    # The LU kernel addresses memory by three strides (batch, row, col) and must keep the 3D batch layout,
    # otherwise 2D stride(1)=1 is mistaken for the row span and the matrix contents are read completely out of place.
    A_work = A_dense.reshape(1, n, n).contiguous()
    B_work = B_solve.reshape(1, n, 1).contiguous()
    LU_workspace = torch.empty_like(A_work)
    X_workspace = torch.empty_like(B_work)
    block_n = max(triton.next_power_of_2(n), 4)

    with torch.no_grad(), _no_adjust_block_size():
        linalg_solve_kernel.jit_function[(1,)](
            A_work,
            LU_workspace,
            B_work,
            X_workspace,
            n,
            1,
            A_work.stride(0),
            A_work.stride(1),
            LU_workspace.stride(0),
            LU_workspace.stride(1),
            B_work.stride(0),
            B_work.stride(1),
            X_workspace.stride(0),
            X_workspace.stride(1),
            BLOCK_N=block_n,
            BLOCK_K=_LU_BLOCK_K,
            num_warps=_LU_NUM_WARPS,
            num_stages=_LU_NUM_STAGES,
        )
    return X_workspace.reshape(n)


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
    logger.debug("GEMS_ENFLAME_GCU400 SPSOLVE")

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
    # torch_gcu falls back to CPU for sparse CSR construction on GCU, so A residing on CPU is a normal input;
    # the compute device is taken from B, and B must indeed be on GCU to avoid a ValueError from a CPU-side B
    # entering the Triton launch (the public contract requires a clean RuntimeError).
    compute_device = B.device
    if compute_device.type != "gcu":
        raise RuntimeError(
            f"spsolve: expected B on a GCU device, but got {B.device}; "
            "a sparse CSR operand may stay on CPU (torch_gcu fallback), "
            "but the right-hand side must be on the compute device"
        )
    if A.device.type not in ("cpu", "gcu"):
        raise RuntimeError(f"spsolve: expected A on CPU or GCU, but got {A.device}")

    A_dense = _csr_to_dense(A, compute_device)

    # The dense Triton solve only supports fp32/fp64; solving a linear system in
    # fp16/bf16 is numerically unstable regardless. Solve in fp32 (or higher) and
    # cast the result back to the input dtype.
    out_dtype = A.dtype
    compute_dtype = (
        torch.float32 if out_dtype in (torch.float16, torch.bfloat16) else out_dtype
    )
    A_solve = A_dense.to(compute_dtype)
    B_solve = B.to(compute_dtype)

    # Solve A @ X = B with the Triton LU kernel fixed at num_warps=1, avoiding the wrong config chosen by
    # autotune on GCU400; disabling AABS is also handled uniformly inside that launch.
    X = _solve_dense_system(A_solve, B_solve)
    return X.to(out_dtype)


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

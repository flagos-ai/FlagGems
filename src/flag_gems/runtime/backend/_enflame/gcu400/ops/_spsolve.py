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

# Launch parameters of the CSR scatter kernel.
_CSR_SCATTER_BLOCK = 64
_CSR_SCATTER_WARPS = 4

# Launch parameters of the legacy dense-solve fallback (fp64 / oversized n).
_LU_BLOCK_K = 4
_LU_NUM_WARPS = 1
_LU_NUM_STAGES = 1

# The panel LU kernel addresses A with 32-bit row*N+col offsets; keep the
# fast path inside the i32-safe domain (n*n < 2^31) and fall back to the
# legacy kernel beyond it.
_LU_FAST_MAX_N = 46340


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
    dense[r, col] using a row-major linear index. Values are cast to the
    dense buffer's dtype on store, so a fp16/bf16 CSR values tensor can be
    densified directly into the fp32 working buffer of the solver.
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
        tl.store(
            dense_ptr + row_base + cols,
            vals.to(dense_ptr.dtype.element_ty),
            mask=valid,
        )


def _csr_to_dense(A, compute_device, dtype):
    """Scatter a sparse CSR matrix into a dense buffer of ``dtype`` on ``compute_device``.

    The GCU backend does not support sparse CSR storage, so torch_gcu falls back to CPU for CSR construction; the three
    components (crow / col / val) must therefore be moved to GCU explicitly before Triton kernels can read them,
    and the dense buffer is also created on the compute device. The dtype cast of the values is folded into the
    scatter kernel's store, removing a separate full-matrix copy kernel from the op's critical path.
    """
    n_rows, n_cols = A.shape
    crow = A.crow_indices().contiguous().to(compute_device)
    col = A.col_indices().contiguous().to(compute_device)
    val = A.values().contiguous().to(compute_device)

    dense = torch.zeros((n_rows, n_cols), dtype=dtype, device=compute_device)

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


# ---------------------------------------------------------------------------
# Fused panel LU + RHS elimination + backward substitution (single program).
#
# The generic linalg_solve_kernel walks the matrix row by row through global
# memory with serial scalar pivot scans; on gcu400 that costs ~2.5*n^2
# dependent load/store round trips per solve (n=1024 lands at ~600ms). This
# kernel instead follows the proven structure of the gcu400 linalg_det panel
# kernel (same barrier discipline, same pivot search, same band layout), so
# the elimination becomes n serial steps of fixed-cost tile ops:
#   - ONE_PANEL band (n <= 128): the whole trailing submatrix is one
#     [BLOCK, PANEL] tile op per step; num_warps=1 (the det band measurement:
#     intra-warp reduction trees beat every cross-warp shape on this stack).
#   - SINGLE band (n <= 256): BLOCK covers all rows, PANEL=128 narrows the
#     per-step rank-1 tile; num_warps=4 ([256,128] tiles exceed the DSM
#     budget at lower warp counts).
#   - multi-block band (n > 256): per-panel factorization + strip phase +
#     rank-PANEL tl.dot trailing updates, all inside the one program.
# The right-hand side rides along the elimination instead of a separate
# forward substitution: X rows are swapped with the pivot rows and every
# rank-1 step applies X[rows] -= l_col * X[k] immediately over all rows
# below the pivot (swap-consistent by construction). Only the backward
# substitution remains after the factorization.
# The factorization runs in place on the densified working buffer (the
# "LU is the workspace" trick from _linalg_det), so the A->LU copy phase of
# the generic kernel disappears as well.
# ---------------------------------------------------------------------------


@libentry()
@triton.jit
def _spsolve_lu_kernel(
    A,
    B,
    X,
    N,
    PANEL: tl.constexpr,
    BLOCK: tl.constexpr,
    SINGLE: tl.constexpr,
    ONE_PANEL: tl.constexpr,
):
    # X = B, cast to the working (fp32) dtype.
    for i0 in range(0, N, BLOCK):
        idx = i0 + tl.arange(0, BLOCK)
        m = idx < N
        bval = tl.load(B + idx, mask=m, other=0.0)
        tl.store(X + idx, bval.to(tl.float32), mask=m)
    tl.debug_barrier()

    if ONE_PANEL:
        for k in range(0, N):
            # Pivot: argmax |A[k:n, k]| (first max wins, like LAPACK idamax).
            rows = k + tl.arange(0, BLOCK)
            col = tl.load(A + rows * N + k, mask=rows < N, other=0.0)
            abs_col = tl.abs(col)
            best_val = tl.max(abs_col, axis=0)
            best_row = tl.min(tl.where(abs_col == best_val, rows, N), axis=0)
            best_row = tl.where(best_val <= 0.0, k, best_row)

            # Swap rows k and best_row over the full width, plus the RHS.
            # The guard skips the no-op swap when the pivot already sits at
            # row k (the common case for diagonally dominant inputs).
            cols = tl.arange(0, BLOCK)
            cmask = cols < N
            if best_row != k:
                row_k = tl.load(A + k * N + cols, mask=cmask, other=0.0)
                row_p = tl.load(A + best_row * N + cols, mask=cmask, other=0.0)
                tl.store(A + k * N + cols, row_p, mask=cmask)
                tl.store(A + best_row * N + cols, row_k, mask=cmask)
                xk = tl.load(X + k)
                xp = tl.load(X + best_row)
                tl.store(X + k, xp)
                tl.store(X + best_row, xk)

            tl.debug_barrier()

            pivot = tl.load(A + k * N + k)
            safe_pivot = tl.where(pivot == 0.0, 1.0, pivot)
            rows = k + 1 + tl.arange(0, BLOCK)
            rmask = rows < N
            l_col = tl.load(A + rows * N + k, mask=rmask, other=0.0)
            l_col = l_col / safe_pivot

            # RHS: X[rows] -= l_col * X[k] over the whole trailing range.
            xk2 = tl.load(X + k)
            xvec = tl.load(X + rows, mask=rmask, other=0.0)
            xvec = xvec - l_col * xk2
            tl.store(X + rows, xvec, mask=rmask)

            # Rank-1 update of the trailing submatrix (single tile op).
            pcols = k + 1 + tl.arange(0, PANEL)
            pcmask = pcols < N
            u_row = tl.load(A + k * N + pcols, mask=pcmask, other=0.0)
            tmask = rmask[:, None] & pcmask[None, :]
            tile = tl.load(
                A + rows[:, None] * N + pcols[None, :], mask=tmask, other=0.0
            )
            tile = tile - l_col[:, None] * u_row[None, :]
            tl.store(A + rows[:, None] * N + pcols[None, :], tile, mask=tmask)

            tl.debug_barrier()
    else:
        for k0 in range(0, N, PANEL):
            kend = tl.minimum(k0 + PANEL, N)

            if SINGLE:
                for k in range(k0, kend):
                    rows = k + tl.arange(0, BLOCK)
                    col = tl.load(A + rows * N + k, mask=rows < N, other=0.0)
                    abs_col = tl.abs(col)
                    best_val = tl.max(abs_col, axis=0)
                    best_row = tl.min(tl.where(abs_col == best_val, rows, N), axis=0)
                    best_row = tl.where(best_val <= 0.0, k, best_row)

                    cols = k0 + tl.arange(0, BLOCK)
                    cmask = cols < N
                    if best_row != k:
                        row_k = tl.load(A + k * N + cols, mask=cmask, other=0.0)
                        row_p = tl.load(A + best_row * N + cols, mask=cmask, other=0.0)
                        tl.store(A + k * N + cols, row_p, mask=cmask)
                        tl.store(A + best_row * N + cols, row_k, mask=cmask)
                        xk = tl.load(X + k)
                        xp = tl.load(X + best_row)
                        tl.store(X + k, xp)
                        tl.store(X + best_row, xk)

                    tl.debug_barrier()

                    pivot = tl.load(A + k * N + k)
                    safe_pivot = tl.where(pivot == 0.0, 1.0, pivot)
                    rows = k + 1 + tl.arange(0, BLOCK)
                    rmask = rows < N
                    l_col = tl.load(A + rows * N + k, mask=rmask, other=0.0)
                    l_col = l_col / safe_pivot
                    # The stored L factors feed the strip / dot phases of this
                    # panel; on the last panel there is no trailing reader.
                    if kend < N:
                        tl.store(A + rows * N + k, l_col, mask=rmask)

                    # RHS: X[rows] -= l_col * X[k] over the whole trailing
                    # range (BLOCK >= N), applied immediately per step.
                    xk2 = tl.load(X + k)
                    xvec = tl.load(X + rows, mask=rmask, other=0.0)
                    xvec = xvec - l_col * xk2
                    tl.store(X + rows, xvec, mask=rmask)

                    # Rank-1 update inside the panel columns.
                    pcols = k + 1 + tl.arange(0, PANEL)
                    pcmask = pcols < kend
                    u_row = tl.load(A + k * N + pcols, mask=pcmask, other=0.0)
                    tmask = rmask[:, None] & pcmask[None, :]
                    tile = tl.load(
                        A + rows[:, None] * N + pcols[None, :],
                        mask=tmask,
                        other=0.0,
                    )
                    tile = tile - l_col[:, None] * u_row[None, :]
                    tl.store(A + rows[:, None] * N + pcols[None, :], tile, mask=tmask)

                    tl.debug_barrier()
            else:
                for k in range(k0, kend):
                    best_val = tl.full((), -1.0, dtype=A.dtype.element_ty)
                    best_row = tl.full((), k, dtype=tl.int32)
                    for i0 in range(k, N, BLOCK):
                        rows = i0 + tl.arange(0, BLOCK)
                        col = tl.load(A + rows * N + k, mask=rows < N, other=0.0)
                        abs_col = tl.where((rows >= k) & (rows < N), tl.abs(col), -1.0)
                        tile_max = tl.max(abs_col, axis=0)
                        tile_row = tl.min(
                            tl.where(abs_col == tile_max, rows, N), axis=0
                        )
                        is_better = tile_max > best_val
                        best_row = tl.where(is_better, tile_row, best_row)
                        best_val = tl.where(is_better, tile_max, best_val)
                    best_row = tl.where(best_val <= 0.0, k, best_row)

                    if best_row != k:
                        for j0 in range(k0, N, BLOCK):
                            cols = j0 + tl.arange(0, BLOCK)
                            cmask = cols < N
                            row_k = tl.load(A + k * N + cols, mask=cmask, other=0.0)
                            row_p = tl.load(
                                A + best_row * N + cols, mask=cmask, other=0.0
                            )
                            tl.store(A + k * N + cols, row_p, mask=cmask)
                            tl.store(A + best_row * N + cols, row_k, mask=cmask)
                        xk = tl.load(X + k)
                        xp = tl.load(X + best_row)
                        tl.store(X + k, xp)
                        tl.store(X + best_row, xk)

                    tl.debug_barrier()

                    pivot = tl.load(A + k * N + k)
                    safe_pivot = tl.where(pivot == 0.0, 1.0, pivot)

                    # Rank-1 update inside the panel columns, by row block.
                    # The scaled L factors are computed once here (fusing the
                    # separate scale pass of the classic layout) and stored
                    # for this panel's strip / dot phases; the RHS rides
                    # along on every row block immediately (X[rows] -=
                    # l_col * X[k]), which keeps the X entries
                    # swap-consistent, because a row swapped in from the
                    # trailing region later in the panel has already
                    # received every earlier step's update.
                    pcols = k + 1 + tl.arange(0, PANEL)
                    pcmask = pcols < kend
                    u_row = tl.load(A + k * N + pcols, mask=pcmask, other=0.0)
                    xk2 = tl.load(X + k)
                    for i0 in range(k + 1, N, BLOCK):
                        rows = i0 + tl.arange(0, BLOCK)
                        rmask = rows < N
                        l_col = tl.load(A + rows * N + k, mask=rmask, other=0.0)
                        l_col = l_col / safe_pivot
                        tl.store(A + rows * N + k, l_col, mask=rmask)
                        xv = tl.load(X + rows, mask=rmask, other=0.0)
                        tl.store(X + rows, xv - l_col * xk2, mask=rmask)
                        tmask = rmask[:, None] & pcmask[None, :]
                        tile = tl.load(
                            A + rows[:, None] * N + pcols[None, :],
                            mask=tmask,
                            other=0.0,
                        )
                        tile = tile - l_col[:, None] * u_row[None, :]
                        tl.store(
                            A + rows[:, None] * N + pcols[None, :],
                            tile,
                            mask=tmask,
                        )

                    tl.debug_barrier()

            if kend < N:
                # Strip phase: apply the panel's L factors to the panel rows
                # of the trailing columns.
                for c in range(k0, kend):
                    prows = c + 1 + tl.arange(0, PANEL)
                    prmask = prows < kend
                    l_strip = tl.load(A + prows * N + c, mask=prmask, other=0.0)
                    for j0 in range(kend, N, BLOCK):
                        cols = j0 + tl.arange(0, BLOCK)
                        cmask = cols < N
                        u_strip = tl.load(A + c * N + cols, mask=cmask, other=0.0)
                        tmask = prmask[:, None] & cmask[None, :]
                        tile = tl.load(
                            A + prows[:, None] * N + cols[None, :],
                            mask=tmask,
                            other=0.0,
                        )
                        tile = tile - l_strip[:, None] * u_strip[None, :]
                        tl.store(
                            A + prows[:, None] * N + cols[None, :],
                            tile,
                            mask=tmask,
                        )
                    tl.debug_barrier()

                # Trailing rank-PANEL update via tl.dot.
                pcols = k0 + tl.arange(0, PANEL)
                pmask = pcols < kend
                for i0 in range(kend, N, BLOCK):
                    rows = i0 + tl.arange(0, BLOCK)
                    rmask = rows < N
                    l_tile = tl.load(
                        A + rows[:, None] * N + pcols[None, :],
                        mask=rmask[:, None] & pmask[None, :],
                        other=0.0,
                    )
                    for j0 in range(kend, N, BLOCK):
                        cols = j0 + tl.arange(0, BLOCK)
                        cmask = cols < N
                        u_tile = tl.load(
                            A + pcols[:, None] * N + cols[None, :],
                            mask=pmask[:, None] & cmask[None, :],
                            other=0.0,
                        )
                        tmask = rmask[:, None] & cmask[None, :]
                        tile = tl.load(
                            A + rows[:, None] * N + cols[None, :],
                            mask=tmask,
                            other=0.0,
                        )
                        tile = tile - tl.dot(l_tile, u_tile, input_precision="ieee")
                        tl.store(
                            A + rows[:, None] * N + cols[None, :],
                            tile,
                            mask=tmask,
                        )

                tl.debug_barrier()

    # Backward substitution: x_i = (y_i - sum_{j>i} U[i, j] * x_j) / U[i, i].
    # The columns beyond the first BLOCK must be folded in block by block
    # whenever N > BLOCK (the multi-block band).
    i = N - 1
    while i >= 0:
        s = tl.zeros((), dtype=tl.float32)
        for j0 in range(i + 1, N, BLOCK):
            cols = j0 + tl.arange(0, BLOCK)
            cmask = cols < N
            u_row = tl.load(A + i * N + cols, mask=cmask, other=0.0)
            xj = tl.load(X + cols, mask=cmask, other=0.0)
            s = s + tl.sum(u_row * xj, axis=0)
        ui = tl.load(A + i * N + i)
        xi = tl.load(X + i)
        xi = (xi - s) / ui
        tl.store(X + i, xi)
        tl.debug_barrier()
        i -= 1


def _solve_dense_system_fast(A_dense, B, n):
    """Solve the dense system with the fused panel LU kernel (fp32 only).

    Band layout mirrors the gcu400 linalg_det panel kernel:
    - n <= 128: ONE_PANEL, BLOCK = PANEL = next_pow2(n), num_warps=1;
    - n <= 256: SINGLE, BLOCK=256 / PANEL=128, num_warps=4;
    - else:     multi-block panels, BLOCK=256 / PANEL=64, num_warps=4.
    """
    X = torch.empty((n,), dtype=A_dense.dtype, device=A_dense.device)
    if n <= 128:
        block = max(32, triton.next_power_of_2(n))
        panel = max(16, triton.next_power_of_2(n))
        single = 0
        one_panel = 1
        num_warps = 1
    elif n <= 256:
        block = 256
        panel = 128
        single = 1
        one_panel = 0
        num_warps = 4
    else:
        # PANEL=64 beats 128 by 6-9% at n>=512 (sweep on GCU400: narrower
        # per-k rank-1 tiles outweigh the extra strip/dot passes); BLOCK=256
        # with num_warps=4 is the DSM-feasible optimum (num_warps=2 needs
        # 1187840B > the 917504B budget).
        block = 256
        panel = 64
        single = 0
        one_panel = 0
        num_warps = 4

    with torch.no_grad(), _no_adjust_block_size():
        _spsolve_lu_kernel[(1,)](
            A_dense,
            B,
            X,
            n,
            PANEL=panel,
            BLOCK=block,
            SINGLE=single,
            ONE_PANEL=one_panel,
            num_warps=num_warps,
            num_stages=1,
        )
    return X


def _solve_dense_system_legacy(A_dense, B_solve):
    """Legacy dense solve through the generic linalg_solve kernel.

    Kept for fp64 inputs (no fp64 arithmetic on gcu400, so this fails the
    same way the previous implementation did) and for matrices beyond the
    i32 offset domain of the fast kernel. The direct JITFunction launch
    pins num_warps=1: the autotuned num_warps=2 config mis-computes the
    pivot control flow on gcu400.
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
    contract. The sparse operand is densified with a Triton scatter kernel
    and the resulting dense system is solved with the fused gcu400 panel LU
    Triton kernel (device-side), rather than a native dense solve.

    Args:
        A: Sparse CSR tensor of shape (n, n).
        B: Dense right-hand side vector of shape (n,).
        left: Must be True; ``X @ A = B`` is not supported.

    Returns:
        The dense solution vector ``X`` of shape (n,).
    """
    logger.debug("GEMS_ENFLAME SPSOLVE")

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

    n = A.size(0)
    if n == 0:
        return B.contiguous().to(A.dtype)

    # The dense Triton solve only supports fp32/fp64; solving a linear system in
    # fp16/bf16 is numerically unstable regardless. Solve in fp32 (or higher) and
    # cast the result back to the input dtype.
    out_dtype = A.dtype
    compute_dtype = (
        torch.float32 if out_dtype in (torch.float16, torch.bfloat16) else out_dtype
    )

    # Densify with the dtype cast folded into the scatter kernel's store, so
    # the fp32 working buffer is ready for the solver without an extra copy.
    A_solve = _csr_to_dense(A, compute_device, compute_dtype)
    B_c = B.contiguous()

    if compute_dtype == torch.float32 and n <= _LU_FAST_MAX_N:
        X = _solve_dense_system_fast(A_solve, B_c, n)
    else:
        B_solve = B_c.to(compute_dtype)
        X = _solve_dense_system_legacy(A_solve, B_solve)
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

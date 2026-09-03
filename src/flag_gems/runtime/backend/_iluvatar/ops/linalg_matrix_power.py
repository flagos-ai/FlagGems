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

"""linalg_matrix_power override for the iluvatar (CoreX) backend.

CoreX has no fp64 compute path, so:
  * the triangular-solve update gemm accumulates in fp32 (the private TRSM
    below replaces the generic fp64-accumulating update);
  * fp32 negative powers run the df64 (double-single) route of the generic
    module (_inverse(use_df64=True) + the df64 power kernels).

Everything else - validation, positive powers, out handling - is the generic
NV dispatch, re-entered after the df64 route with the (possibly inverted)
input.
"""

import importlib

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.ops.linalg_matrix_power import (
    _eye_like,
    _inverse,
    _matrix_power_df64,
    _trsm_solve_register,
)
from flag_gems.utils import libentry

# NB: bind the *module* explicitly - ``import flag_gems.ops.linalg_matrix_power
# as _generic`` would resolve through the package attribute, which the
# re-exported ``linalg_matrix_power`` function shadows.
_generic = importlib.import_module("flag_gems.ops.linalg_matrix_power")


@triton.jit
def _trsm_update_register(
    A_ptr,
    B_ptr,
    stride_a_n,
    stride_b_k,
    blk_start,
    blk_end,
    blk_sz,
    M_REM,
    rem_s,
    bound,
    x_all,
    a_cols,
    rr,
    col_offs,
    col_mask,
    BM: tl.constexpr,
    K_SLICE: tl.constexpr,
):
    """Update phase of _trsm_kernel (fp32 accumulation): reuses the register
    tile x_all as the X panel and accumulates the update gemm elementwise in
    fp32 (this backend has no fp64 compute path)."""
    # Reuse the register tile (x_all) as the X panel.
    for m_start in range(0, M_REM, BM):
        rm = rem_s + m_start + rr
        mask_m = rm < bound
        a_sub = tl.load(
            A_ptr + rm[:, None] * stride_a_n + (blk_start + a_cols)[None, :],
            mask=mask_m[:, None] & (a_cols[None, :] < blk_sz),
            other=0.0,
        )
        # No fp64 compute path on this backend: accumulate the update gemm
        # elementwise in fp32 (the fp64 dot cannot be lowered here).
        acc = tl.sum(a_sub[:, :, None] * x_all[None, :, :], axis=1)
        b_base = B_ptr + rm[:, None] * stride_b_k + col_offs[None, :]
        b_curr = tl.load(b_base, mask=mask_m[:, None] & col_mask[None, :], other=0.0)
        b_curr = b_curr.to(acc.dtype) - acc
        tl.store(b_base, b_curr, mask=mask_m[:, None] & col_mask[None, :])


@libentry()
@triton.jit
def _trsm_kernel(
    A_ptr,
    B_ptr,
    INV_ptr,
    N,
    K,
    stride_a_n,
    stride_b_k,
    BLOCK_SIZE: tl.constexpr,
    K_SLICE: tl.constexpr,
    BM: tl.constexpr,
    UPPER: tl.constexpr,
    UNIT: tl.constexpr,
):
    """Blocked triangular solve A X = B in place (B <- X), one program per
    K_SLICE-column group of the RHS.

    Rows are processed in BLOCK_SIZE blocks.  The diagonal block is solved by
    serial forward/backward substitution (row-by-row, parallel across the
    K_SLICE columns), then the remaining rows are updated with a tl.dot gemm —
    every data dependency stays within a single program, so the kernel is
    barrier-free.  This replaces the (batch, n)-grid scalar substitution
    kernels, whose one-program-per-column serial O(n^2) loop was the dominant
    cost of the negative-power path (~77 ms per solve at n=1024 vs ~1 ms
    here).
    """
    pid = tl.program_id(0)
    col_start = pid * K_SLICE
    if col_start >= K:
        return

    num_blocks = tl.cdiv(N, BLOCK_SIZE)

    a_cols = tl.arange(0, BLOCK_SIZE)
    x_rows = tl.arange(0, BLOCK_SIZE)
    x_kcols = tl.arange(0, K_SLICE)
    xr = tl.broadcast_to(x_rows[:, None], (BLOCK_SIZE, K_SLICE))
    col_offs = col_start + x_kcols
    col_mask = col_offs < K
    rr = tl.arange(0, BM)

    for block_idx in range(num_blocks):
        bk = block_idx if not UPPER else num_blocks - 1 - block_idx
        blk_start = bk * BLOCK_SIZE
        blk_end = tl.minimum(blk_start + BLOCK_SIZE, N)
        blk_sz = blk_end - blk_start

        # ═══ Diagonal block: serial substitution over rows ═══
        # Pre-compute diagonal reciprocals (division out of the serial chain).
        if not UNIT:
            diag_vals = tl.load(
                A_ptr + (blk_start + a_cols) * stride_a_n + (blk_start + a_cols),
                mask=a_cols < blk_sz,
                other=1.0,
            )
            tl.store(
                INV_ptr + pid * BLOCK_SIZE + a_cols,
                1.0 / diag_vals,
                mask=a_cols < blk_sz,
            )

        # Serial solve of the diagonal block: the block's X stays in a
        # register tile (x_all) across the serial row chain.
        x_all = _trsm_solve_register(
            A_ptr,
            B_ptr,
            INV_ptr,
            pid,
            stride_a_n,
            stride_b_k,
            blk_start,
            blk_end,
            blk_sz,
            a_cols,
            xr,
            col_offs,
            col_mask,
            BLOCK_SIZE,
            UPPER,
            UNIT,
        )

        # ═══ Update: B[rest, kslice] -= A[rest, blk] @ X[blk, kslice] ═══
        need_update = tl.where(UPPER, bk > 0, blk_end < N)
        if need_update:
            M_REM = tl.where(UPPER, blk_start, N - blk_end)
            rem_s = tl.where(UPPER, 0, blk_end)
            bound = tl.where(UPPER, blk_start, N)
            _trsm_update_register(
                A_ptr,
                B_ptr,
                stride_a_n,
                stride_b_k,
                blk_start,
                blk_end,
                blk_sz,
                M_REM,
                rem_s,
                bound,
                x_all,
                a_cols,
                rr,
                col_offs,
                col_mask,
                BM,
                K_SLICE,
            )


def _trsm_solve_2d(A_tri, B, upper: bool, unitriangular: bool):
    """Same blocked triangular solve as the generic one, but the update gemm
    accumulates in fp32 (no fp64 compute path on this backend)."""
    n = A_tri.shape[0]
    k = B.shape[1]
    K_SLICE = 8
    BM = 128
    num_kslices = (k + K_SLICE - 1) // K_SLICE
    if unitriangular:
        inv = B
        unit_flag = True
    else:
        inv = torch.zeros(num_kslices * 32, dtype=A_tri.dtype, device=A_tri.device)
        unit_flag = False
    _trsm_kernel[(num_kslices,)](
        A_tri,
        B,
        inv,
        n,
        k,
        A_tri.stride(0),
        B.stride(0),
        32,
        K_SLICE,
        BM,
        upper,
        unit_flag,
        num_warps=4,
        num_stages=3,
    )
    return B


# Hook the shared hosts so every triangular solve on this backend accumulates
# in fp32 (one backend per process - safe).
_generic._trsm_solve_2d = _trsm_solve_2d


def linalg_matrix_power(A, n, *, out=None):
    """fp32 negatives take the df64 route (module docstring); everything else
    follows the generic NV dispatch."""
    _generic.logger.debug("GEMS LINALG_MATRIX_POWER (iluvatar)")

    # ---- validation (identical to the generic entry) ----
    shape = A.shape
    if len(shape) < 2:
        raise RuntimeError(
            f"linalg_matrix_power: A must be at least 2-D, got shape {shape}"
        )
    m, k = shape[-2], shape[-1]
    if m != k:
        raise RuntimeError(f"linalg_matrix_power: A must be square, got ({m}, {k})")
    if not isinstance(n, int):
        raise TypeError(f"linalg_matrix_power: n must be int, got {type(n).__name__}")
    if A.dtype not in (torch.float32, torch.float64):
        raise RuntimeError(
            f"linalg_matrix_power: flag_gems supports only float32 and float64, "
            f"got {A.dtype}"
        )

    # ---- n == 0 / n == 1 ----
    if n == 0:
        eye = _eye_like(A)
        if out is not None:
            out.copy_(eye)
            return out
        return eye
    if n == 1:
        if out is not None:
            out.copy_(A)
            return out
        return A.clone()

    if A.device.type != flag_gems.device:
        raise RuntimeError(
            f"linalg_matrix_power: flag_gems supports only {flag_gems.device}, "
            f"got {A.device}"
        )

    # ---- fp32 negative powers: df64 route (no fp64 compute path) ----
    if n < 0 and A.dtype == torch.float32:
        inv = _inverse(A, use_df64=True)
        if isinstance(inv, tuple):
            # Small M: df64 inverse (hi/lo) pair + df64 power - the only
            # supported df64 path (in-register kernels).
            Xh, Xl = inv
            return _matrix_power_df64(Xh, Xl, -n, m, shape, out=out)
        # Large M: the external fp32 LU has no df64 low part, so the inverse
        # is a plain fp32 tensor; compute the power in fp32 via the generic
        # positive-power dispatch.
        A = inv
        n = -n
    return _generic.linalg_matrix_power(A, n, out=out)

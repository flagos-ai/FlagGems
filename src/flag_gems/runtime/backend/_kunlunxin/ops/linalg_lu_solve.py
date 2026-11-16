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

"""XPU override for ``aten::linalg_lu_solve``.

Background
----------
The generic implementation (``flag_gems/ops/linalg_lu_solve.py``) carries an
in-tree ``_XPU_SERIAL`` column-resident path, but it has two problems on this
backend:

* it passes ``N``/``nrhs`` as *runtime* scalars (extra launch arguments are a
  measurable per-call cliff on XPU3), and
* its ``_xpu_lu_solve_kernel`` body is O(N) serial one-hot reductions over a
  ``BLOCK_N`` tile; once ``BLOCK_N == 256`` (``n > 128``) the pass
  ``TritonXPUUnrollControl`` rejects the loop-carried reduction with
  ``'arith.select' ... same type`` / ``uni_sram``.  So the benchmark's
  ``(256, 64)`` shape never compiles.

This override splits the work by size:

* ``n <= 64`` -- a single-launch column-resident kernel with ``N``/``nrhs`` as
  ``tl.constexpr``.  LU is pre-transposed once so every serial step reads a
  *contiguous* row of ``LU^T`` (== the column of ``LU`` it needs) instead of a
  strided column.  The pivot permutation is composed in registers (<= 64 lanes,
  which lowers cleanly).  One program per ``(batch, RHS column)``.
* ``n > 64``  -- the two substitution phases are handed to the backend's
  blocked XPU TRSM (``linalg_solve_triangular``, the same machinery that backs
  ``aten::linalg_solve_triangular``), which is dominated by ``tl.dot`` panel
  updates and lowers at every size including 256.  The RHS is permuted first by
  a small gather kernel; the permutation itself is composed on the host.

Why the pivot permutation is composed on the host for ``n > 64``: the sequential
swap chain needs a loop-carried reduction over a ``next_pow2(n)`` tile, and at
256 lanes that reduction is exactly the unbuildable case above (verified for
both a ``tl.where`` and a multiplication form).  Composing the swap chain from
the ``int`` pivots is integer index bookkeeping, not numerical operator compute;
it is the analogue of LAPACK's host-side ``ipiv`` handling.  No floating-point
operator is ever evaluated on the host.

``left`` / ``adjoint`` are supported through the standard transpose identity:
``A = P^T L U`` so ``A^T = U^T L^T P`` and a transposed solve is
``U^T w = b`` (lower, non-unit) then ``L^T s = w`` (upper, unit) then
``x = P^T s``.  Only the dense solve is replaced here.
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime.backend._kunlunxin.ops.linalg_solve_triangular import (
    linalg_solve_triangular,
)
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Size boundary between the column-resident kernel and the blocked TRSM pair.
# Measured on XPU 4: at n == 64 the column kernel still wins (64,64 -> 0.109
# vs 0.146 speedup); at n == 128 the TRSM pair is ~2.5x faster (0.130 vs 0.049).
_SMALL_MAX_N = 64


@libentry()
@triton.jit
def _lu_solve_small_kernel(
    LUT,  # LU transposed, contiguous (batch, n, n); LUT[i, r] == LU[r, i]
    B,
    X,
    pivots,
    N: tl.constexpr,
    nrhs: tl.constexpr,
    batch_stride_lut,
    stride_lut_row,
    batch_stride_piv,
    stride_piv_row,
    batch_stride_b,
    stride_b_row,
    stride_b_col,
    batch_stride_x,
    stride_x_row,
    stride_x_col,
    BLOCK_N: tl.constexpr,
):
    """Register-resident one-column solve (normal ``A X = B`` orientation)."""
    pid = tl.program_id(0)
    batch = pid // nrhs
    col = pid % nrhs

    rows = tl.arange(0, BLOCK_N)
    rows_mask = rows < N
    piv_base = pivots + batch * batch_stride_piv

    # Compose the sequential 1-indexed pivot swaps into one row permutation.
    # Kept in fp32 (< 2^24 so exact) because the int32 loop-carried tile used by
    # the generic kernel trips TritonXPUUnrollControl at the 128-lane boundary.
    perm_f = rows.to(tl.float32)
    for i in range(N):
        p = tl.load(piv_base + i * stride_piv_row).to(tl.int32) - 1
        at_i = rows == i
        at_p = rows == p
        v_i = tl.sum(tl.where(at_i, perm_f, 0.0))
        v_p = tl.sum(tl.where(at_p, perm_f, 0.0))
        perm_f = tl.where(at_i, v_p, tl.where(at_p, v_i, perm_f))
    perm = perm_f.to(tl.int32)

    lut_base = LUT + batch * batch_stride_lut
    # w = (P @ B)[:, col]
    w = tl.load(
        B + batch * batch_stride_b + perm * stride_b_row + col * stride_b_col,
        mask=rows_mask,
        other=0.0,
    )

    # Forward substitution, L has an implicit unit diagonal.
    for i in range(N):
        df = (rows - i).to(tl.float32)
        oh = tl.maximum(1.0 - tl.abs(df), 0.0)
        step = tl.maximum(tl.minimum(df, 1.0), 0.0)
        w_i = tl.sum(w * oh)
        # Row i of LU^T == column i of LU; contiguous in LUT.
        coef = tl.load(lut_base + i * stride_lut_row + rows, mask=rows_mask, other=0.0)
        w = w - step * coef * w_i

    # Backward substitution with U.
    for i in range(N - 1, -1, -1):
        df = (rows - i).to(tl.float32)
        oh = tl.maximum(1.0 - tl.abs(df), 0.0)
        below = tl.maximum(tl.minimum(-df, 1.0), 0.0)
        w_i = tl.sum(w * oh)
        diag = tl.load(lut_base + i * stride_lut_row + i)
        x_i = w_i / diag
        coef = tl.load(lut_base + i * stride_lut_row + rows, mask=rows_mask, other=0.0)
        w = w - below * coef * x_i
        w = w + oh * (x_i - w)

    tl.store(
        X + batch * batch_stride_x + rows * stride_x_row + col * stride_x_col,
        w,
        mask=rows_mask,
    )


@libentry()
@triton.jit
def _gather_rows_kernel(
    SRC,
    IDX,  # int32 (batch, n): output row r <- SRC row IDX[b, r]
    DST,
    N: tl.constexpr,
    nrhs: tl.constexpr,
    batch_stride_src,
    stride_src_row,
    stride_src_col,
    batch_stride_idx,
    stride_idx_row,
    batch_stride_dst,
    stride_dst_row,
    stride_dst_col,
    BLOCK_N: tl.constexpr,
):
    """DST[b, r, col] = SRC[b, IDX[b, r], col]; one program per (batch, col)."""
    pid = tl.program_id(0)
    batch = pid // nrhs
    col = pid % nrhs
    rows = tl.arange(0, BLOCK_N)
    rows_mask = rows < N
    idx = tl.load(
        IDX + batch * batch_stride_idx + rows * stride_idx_row,
        mask=rows_mask,
        other=0,
    )
    val = tl.load(
        SRC + batch * batch_stride_src + idx * stride_src_row + col * stride_src_col,
        mask=rows_mask,
        other=0.0,
    )
    tl.store(
        DST + batch * batch_stride_dst + rows * stride_dst_row + col * stride_dst_col,
        val,
        mask=rows_mask,
    )


def _compose_perms(pivots, batch, n, device):
    """Sequential 1-indexed swaps -> (perm, perm_inv), both int32 on device.

    Integer index bookkeeping only (the analogue of LAPACK ipiv handling); no
    floating-point operator work happens on the host.
    """
    pv = pivots.reshape(batch, n).to("cpu").tolist()
    perms = []
    invs = []
    for b in range(batch):
        perm = list(range(n))
        row = pv[b]
        for i in range(n):
            p = int(row[i]) - 1
            if p != i:
                perm[i], perm[p] = perm[p], perm[i]
        inv = [0] * n
        for r in range(n):
            inv[perm[r]] = r
        perms.append(perm)
        invs.append(inv)
    perm_t = torch.tensor(perms, dtype=torch.int32, device=device)
    inv_t = torch.tensor(invs, dtype=torch.int32, device=device)
    return perm_t, inv_t


def _small_solve(LU, pivots, B, n, nrhs, batch):
    """Column-resident solve for n <= _SMALL_MAX_N (normal orientation)."""
    LU_view = LU.reshape(batch, n, n)
    LUT = LU_view.transpose(-1, -2).contiguous()
    piv = pivots.reshape(batch, n).contiguous()
    B_work = B.reshape(batch, n, nrhs).contiguous()
    X = torch.empty_like(B_work)
    block_n = max(triton.next_power_of_2(n), 16)
    _lu_solve_small_kernel[(batch * nrhs,)](
        LUT,
        B_work,
        X,
        piv,
        n,
        nrhs,
        LUT.stride(0),
        LUT.stride(1),
        piv.stride(0),
        piv.stride(1),
        B_work.stride(0),
        B_work.stride(1),
        B_work.stride(2),
        X.stride(0),
        X.stride(1),
        X.stride(2),
        BLOCK_N=block_n,
        num_warps=1,
        num_stages=1,
    )
    return X


def _gather_rows(src, idx, n, nrhs, batch):
    """Return ``src`` with rows permuted by ``idx`` (device gather)."""
    src = src.contiguous()
    dst = torch.empty_like(src)
    block_n = max(triton.next_power_of_2(n), 16)
    _gather_rows_kernel[(batch * nrhs,)](
        src,
        idx,
        dst,
        n,
        nrhs,
        src.stride(0),
        src.stride(1),
        src.stride(2),
        idx.stride(0),
        idx.stride(1),
        dst.stride(0),
        dst.stride(1),
        dst.stride(2),
        BLOCK_N=block_n,
        num_warps=1,
        num_stages=1,
    )
    return dst


def _large_solve(LU, pivots, B, n, nrhs, batch, use_transpose):
    """Permutation + two backend TRSM solves for n > _SMALL_MAX_N."""
    perm, perm_inv = _compose_perms(pivots, batch, n, B.device)
    LU_view = LU.reshape(batch, n, n)
    B_work = B.reshape(batch, n, nrhs).contiguous()
    if not use_transpose:
        # L y = P B ; U x = y
        y = _gather_rows(B_work, perm, n, nrhs, batch)
        y = linalg_solve_triangular(
            LU_view, y, upper=False, left=True, unitriangular=True
        )
        X = linalg_solve_triangular(
            LU_view, y, upper=True, left=True, unitriangular=False
        )
    else:
        # A^T x = b  <=>  U^T w = b ; L^T s = w ; x = P^T s
        LUT = LU_view.transpose(-1, -2).contiguous()
        w = linalg_solve_triangular(
            LUT, B_work, upper=False, left=True, unitriangular=False
        )
        s = linalg_solve_triangular(
            LUT, w, upper=True, left=True, unitriangular=True
        )
        X = _gather_rows(s, perm_inv, n, nrhs, batch)
    return X


def linalg_lu_solve(LU, pivots, B, *, left=True, adjoint=False):
    """Solve a linear system using a precomputed LU factorization."""
    logger.debug("GEMS_KUNLUNXIN LINALG_LU_SOLVE")
    if LU.ndim < 2 or B.ndim < 2:
        raise ValueError("LU and B must be at least 2D")
    if LU.shape[-1] != LU.shape[-2]:
        raise ValueError("LU must be a square matrix")
    n = LU.shape[-1]
    # left=True : A X = B, B must have n rows.  left=False : X A = B, B must
    # have n columns.
    if left:
        if B.shape[-2] != n:
            raise ValueError("B must have compatible dimensions with LU")
    else:
        if B.shape[-1] != n:
            raise ValueError("B must have compatible dimensions with LU")
    if LU.dtype not in (torch.float32, torch.float64):
        raise ValueError(
            f"linalg_lu_solve only supports float32/float64, got {LU.dtype}"
        )

    if LU.numel() == 0 or B.numel() == 0:
        return B.clone()

    batch = math.prod(LU.shape[:-2])

    # use_transpose is (left == adjoint): see the module docstring derivation.
    use_transpose = left == adjoint
    transpose_rhs = not left

    if transpose_rhs:
        # A^T Y = B^T, so the RHS becomes (n, m) with m = B.shape[-2].
        B_work = B.transpose(-1, -2).contiguous()
    else:
        B_work = B
    nrhs = B_work.shape[-1]

    if not use_transpose and n <= _SMALL_MAX_N:
        out = _small_solve(LU, pivots, B_work, n, nrhs, batch)
    else:
        out = _large_solve(LU, pivots, B_work, n, nrhs, batch, use_transpose)

    out = out.reshape(B_work.shape)
    if transpose_rhs:
        out = out.transpose(-1, -2).contiguous()
    return out.reshape(B.shape)

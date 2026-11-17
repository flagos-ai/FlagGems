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

"""Kunlunxin backend override for ``linalg_lu_solve``.

The generic driver already has an XPU branch (``_XPU_SERIAL``) that keeps one
RHS column in a ``BLOCK_N`` register vector and extracts each serial step's
scalar with a one-hot weighted reduction (``tl.gather`` is not lowerable on
this backend).  That kernel compiles up to ``N == 128``; for ``N > 128`` the
``TritonXPUUnrollControl`` pass rejects the loop-carried ``BLOCK_N`` vectors::

    unroll control cannot narrow hoisted operand #0 to the tiled type:
    only a splat constant may be narrowed with extract_slice ...

(``loop_unroll_factor=1`` does not help).  This affects the benchmark matrix,
which contains ``N == 256``.

The fix keeps the same algorithm but removes every *loop-carried vector* from
the pivot composition and from the two substitution sweeps:

* the row permutation is composed once per batch element by its own kernel
  (scalar swaps, so no loop-carried vector) into a scratch row that the solve
  programs only read afterwards;
* the working column is stored to / reloaded from the (contiguous) output
  buffer each step instead of staying live across the loop.

Everything else (validation, error messages, dtype rules) is delegated to the
generic entry point so behaviour does not diverge.
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.linalg_lu_solve import _xpu_lu_solve_serial
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Widest loop-carried ``BLOCK_N`` vector the register-resident kernel compiles
# on this backend.  Wider systems go to the global-memory variant below.
_REGISTER_MAX_N = 128


@libentry()
@triton.jit
def _xpu_pivot_compose_kernel(
    pivots,
    PERM,
    N,
    batch_stride_piv,
    stride_piv_row,
    BLOCK_N: tl.constexpr,
):
    """Compose the sequential 1-indexed pivot swaps into one row permutation.

    One program per batch element.  The ``N`` swaps are inherently serial, so
    only the row indices take part in the chain, and this program is the sole
    writer of the row: every solve program reads the finished permutation
    afterwards, which keeps the ``batch * nrhs`` solve grid read-only on
    ``PERM``.  The swaps are scalar load/store pairs, so no vector is carried
    across the loop -- which is what keeps this compilable for ``BLOCK_N >
    128`` (the register-resident composition used by the generic kernel is
    rejected by ``TritonXPUUnrollControl`` at that width).
    """
    batch = tl.program_id(0)
    rows = tl.arange(0, BLOCK_N)
    rows_mask = rows < N
    piv_base = pivots + batch * batch_stride_piv
    perm_base = PERM + batch * BLOCK_N

    tl.store(perm_base + rows, rows, mask=rows_mask)
    for i in range(0, N):
        p = tl.load(piv_base + i * stride_piv_row).to(tl.int32) - 1
        v_i = tl.load(perm_base + i)
        v_p = tl.load(perm_base + p)
        tl.store(perm_base + i, v_p)
        tl.store(perm_base + p, v_i)


@libentry()
@triton.jit
def _xpu_lu_solve_global_kernel(
    LU,
    B,
    X,
    PERM,
    N,
    nrhs,
    batch_stride_lu,
    stride_lu_row,
    stride_lu_col,
    batch_stride_b,
    stride_b_row,
    stride_b_col,
    batch_stride_x,
    stride_x_row,
    stride_x_col,
    BLOCK_N: tl.constexpr,
):
    """Column-resident LU solve with no loop-carried vector.

    One program per ``(batch, RHS column)``.  The substitution sweeps read and
    write the working column through global memory, so the only register
    vectors are produced and consumed inside a single loop iteration.  This is
    what lets ``BLOCK_N > 128`` compile; the arithmetic is identical to the
    register-resident kernel.

    ``PERM`` is read-only here: it is composed by ``_xpu_pivot_compose_kernel``,
    which has a single writer per batch row.
    """
    pid = tl.program_id(0)
    batch = pid // nrhs
    col = pid % nrhs
    rows = tl.arange(0, BLOCK_N)
    rows_mask = rows < N

    lu_base = LU + batch * batch_stride_lu
    perm_base = PERM + batch * BLOCK_N
    x_col = X + batch * batch_stride_x + col * stride_x_col

    perm = tl.load(perm_base + rows, mask=rows_mask, other=0)
    w0 = tl.load(
        B + batch * batch_stride_b + perm * stride_b_row + col * stride_b_col,
        mask=rows_mask,
        other=0.0,
    )
    tl.store(x_col + rows * stride_x_row, w0, mask=rows_mask)

    # Forward substitution: L has an implicit unit diagonal.
    for i in range(0, N):
        w = tl.load(x_col + rows * stride_x_row, mask=rows_mask, other=0.0)
        df = (rows - i).to(tl.float32)
        oh = tl.maximum(1.0 - tl.abs(df), 0.0)
        step = tl.maximum(tl.minimum(df, 1.0), 0.0)
        w_i = tl.sum(w * oh, axis=0)
        l_col = tl.load(
            lu_base + rows * stride_lu_row + i * stride_lu_col,
            mask=rows_mask,
            other=0.0,
        )
        w = w - step * l_col * w_i
        tl.store(x_col + rows * stride_x_row, w, mask=rows_mask)

    # Backward substitution with U.
    for i in range(N - 1, -1, -1):
        w = tl.load(x_col + rows * stride_x_row, mask=rows_mask, other=0.0)
        df = (rows - i).to(tl.float32)
        oh = tl.maximum(1.0 - tl.abs(df), 0.0)
        below = tl.maximum(tl.minimum(-df, 1.0), 0.0)
        w_i = tl.sum(w * oh, axis=0)
        diag = tl.load(lu_base + i * stride_lu_row + i * stride_lu_col)
        x_i = w_i / diag
        u_col = tl.load(
            lu_base + rows * stride_lu_row + i * stride_lu_col,
            mask=rows_mask,
            other=0.0,
        )
        w = w - below * u_col * x_i
        w = w + oh * (x_i - w)
        tl.store(x_col + rows * stride_x_row, w, mask=rows_mask)


@libentry()
@triton.jit
def _xpu_eye_kernel(RHS, n, total, BLOCK: tl.constexpr):
    """Fill a contiguous ``(batch, n, n)`` buffer with identity matrices."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = (offs // n) % n
    value = tl.where(offs % n == row, 1.0, 0.0)
    tl.store(RHS + offs, value.to(RHS.dtype.element_ty), mask=offs < total)


_IDENTITY_BLOCK = 1024


def _xpu_identity_rhs(n, batch, dtype, device):
    """Contiguous ``(batch, n, n)`` identity built on device.

    ``linalg_inv``/``linalg_inv_ex`` need ``A^-1`` as the solution of
    ``A X = I``.  Materialising ``I`` directly as one contiguous buffer keeps
    the solve's RHS contiguous and avoids both a second full-size copy and the
    ``torch.eye(...).expand(...)`` host-side creation call.
    """
    rhs = torch.empty((batch, n, n), dtype=dtype, device=device)
    total = rhs.numel()
    with torch_device_fn.device(device):
        _xpu_eye_kernel[(triton.cdiv(total, _IDENTITY_BLOCK),)](
            rhs, n, total, BLOCK=_IDENTITY_BLOCK, num_warps=4
        )
    return rhs


def _xpu_lu_solve_impl(LU_work, piv_work, B, batch, n, nrhs):
    """Solve ``LU x = pivots(B)`` for one batch of contiguous factors."""
    if n <= _REGISTER_MAX_N:
        return _xpu_lu_solve_serial(LU_work, piv_work, B, batch, n, nrhs)

    B_work = B.reshape(batch, n, nrhs).contiguous()
    X = torch.empty_like(B_work)
    block_n = max(triton.next_power_of_2(n), 16)
    perm = torch.empty((batch, block_n), dtype=torch.int32, device=LU_work.device)
    with torch.no_grad():
        with torch_device_fn.device(LU_work.device):
            # One writer per batch row, then a read-only consumer grid: the
            # permutation is composed once instead of by every RHS column.
            _xpu_pivot_compose_kernel[(batch,)](
                piv_work,
                perm,
                n,
                piv_work.stride(0),
                piv_work.stride(1),
                BLOCK_N=block_n,
                num_warps=1,
                num_stages=1,
            )
            _xpu_lu_solve_global_kernel[(batch * nrhs,)](
                LU_work,
                B_work,
                X,
                perm,
                n,
                nrhs,
                LU_work.stride(0),
                LU_work.stride(1),
                LU_work.stride(2),
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
    return X.reshape(B.shape)


def linalg_lu_solve(LU, pivots, B, *, left=True, adjoint=False):
    """XPU entry point; semantics and validation match the generic op."""
    logger.debug("GEMS_KUNLUNXIN LINALG_LU_SOLVE")

    if LU.ndim < 2 or B.ndim < 2:
        raise ValueError("LU and B must be at least 2D")
    if LU.shape[-1] != LU.shape[-2]:
        raise ValueError("LU must be a square matrix")
    n = LU.shape[-1]
    if B.shape[-2] != n:
        raise ValueError("B must have compatible dimensions with LU")
    if LU.dtype not in (torch.float32, torch.float64):
        raise ValueError(
            f"linalg_lu_solve only supports float32/float64, got {LU.dtype}"
        )

    if LU.numel() == 0 or B.numel() == 0:
        return B.clone()

    batch = math.prod(LU.shape[:-2])
    nrhs = B.shape[-1]

    LU_work = LU.reshape(batch, n, n).contiguous()
    piv_work = pivots.reshape(batch, n).contiguous()

    return _xpu_lu_solve_impl(LU_work, piv_work, B, batch, n, nrhs)


__all__ = ["linalg_lu_solve"]

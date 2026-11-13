# Copyright 2026, The FlagOS Contributors.
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

"""Kunlunxin backend override for ``inverse``.

Implements matrix inversion with a partial-pivoted Gauss-Jordan elimination on
an augmented ``[A | I]`` system, one program per matrix, driven by a host-side
per-column loop (pivot scan + elimination step).  The generic kernel
(``flag_gems/ops/inverse.py``) is rejected by the TritonXPU
legalizer because it relies on ``tl.gather`` (see the previous fallback's
docstring for the failed alternatives); this kernel follows the proven
``linalg_det._det_elim_kernel`` pattern instead:

* the working tile lives in a global-memory workspace (padded to
  ``ROWS x LDA`` lanes), double-buffered across elimination steps so no
  iteration reads the buffer it writes (avoiding the backend's unsafe in-kernel
  store->load reordering);
* rows/columns are plain 1-D strided vector loads (``row * LDA + col``), never
  ``tl.gather``;
* the pivot search is a masked 1-D ``tl.max``/``tl.min`` over ``ROWS`` lanes
  (``tl.argmax`` is unreliable on this backend, so the smallest row attaining
  the maximum is used, matching LAPACK first-strict-maximum order);
* the inverse accumulator ``X`` is updated by the exact same row operations
  (swap / normalise / eliminate) with the factors taken from ``A``.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_MIN_LDA = 64
_MAX_BLK = 4096
# The single-launch elimination is exact only when the per-iteration tile is at
# least 1024 lanes (smaller tiles get store->load reordered); small n is padded
# up to this many lanes.
_ELIM_MIN_TOT = 1024


def _plan_inv(n):
    rows = triton.next_power_of_2(n)
    if n >= 16 and n % 8 == 0 and n * n <= _MAX_BLK:
        lda = n
    else:
        lda = max(_MIN_LDA, rows)
    tot = rows * lda
    while tot < _ELIM_MIN_TOT:
        lda *= 2
        tot = rows * lda
    blk = min(_MAX_BLK, tot)
    return rows, lda, tot, blk, tot // blk


@libentry()
@triton.jit
def _inv_pack_kernel(
    A,
    W,
    X,
    N,
    LDA: tl.constexpr,
    TOT: tl.constexpr,
    BLK: tl.constexpr,
):
    """Scatter contiguous (batch, N, N) into padded (batch, ROWS, LDA) and
    initialise the inverse accumulator to the padded identity."""
    b = tl.program_id(0).to(tl.int64)
    blk = tl.program_id(1).to(tl.int64)
    e = blk * BLK + tl.arange(0, BLK)
    row = e // LDA
    col = e % LDA
    live = (row < N) & (col < N)
    idx = tl.where(live, row * N + col, 0)
    val = tl.load(A + b * N * N + idx)
    tl.store(W + b * TOT + e, tl.where(live, val, 0.0))
    tl.store(X + b * TOT + e, tl.where(live & (row == col), 1.0, 0.0))


@libentry()
@triton.jit
def _inv_unpack_kernel(
    X,
    OUT,
    N,
    LDA: tl.constexpr,
    TOT: tl.constexpr,
    BLK: tl.constexpr,
):
    """Copy the top-left N x N block of the padded inverse back to the
    contiguous output."""
    b = tl.program_id(0).to(tl.int64)
    blk = tl.program_id(1).to(tl.int64)
    e = blk * BLK + tl.arange(0, BLK)
    mask = e < N * N
    row = e // N
    col = e % N
    # Masked-out lanes address past the N x N block (N*N can be far below
    # TOT when small n is padded up to LDA); clamp their source/destination
    # indices before addressing.  The older flagtree compiler computes the
    # address before applying the mask, so an unmasked out-of-bounds index
    # faults with err_code -714 (illegal memory access).
    src_idx = tl.where(mask, row * LDA + col, 0)
    dst_idx = tl.where(mask, e, 0)
    xval = tl.load(X + b * TOT + src_idx, mask=mask, other=0.0)
    tl.store(OUT + b * N * N + dst_idx, xval, mask=mask)


@libentry()
@triton.jit
def _inv_pivot_scan_kernel(
    W,
    PR,
    AP,
    AK,
    N,
    K,
    LDA: tl.constexpr,
    ROWS: tl.constexpr,
):
    """Pivot search for one matrix (one program per matrix).

    The pivot row and signed pivot values are written to side buffers so the
    chunked update kernel (launched across ``(batch, nblk)`` programs) can
    share them without recomputing a reduction per chunk.  The reduction runs
    over ``LDA`` lanes (the padded row stride) rather than ``ROWS``: a 1-D
    ``tl.max``/``tl.min`` over a small ``ROWS`` tile is miscompiled by the
    flagtree 0.7.0 compiler (illegal memory access for ROWS in 4..16), while
    the ``LDA``-lane form matches the proven ``linalg_det._det_step_kernel``.
    ``tl.argmax`` is unreliable on this backend, so the smallest row attaining
    ``tl.max`` is used (LAPACK first-strict-maximum order).
    """
    b = tl.program_id(0).to(tl.int64)
    base = b * (ROWS * LDA)
    ridx = tl.arange(0, LDA)
    col_k_sp = tl.load(W + base + ridx * LDA + K)
    cand = tl.where((ridx >= K) & (ridx < N), tl.abs(col_k_sp), -1.0)
    best = tl.max(cand, axis=0)
    prow = tl.min(tl.where(cand == best, ridx, LDA), axis=0)
    apk = tl.sum(tl.where(ridx == prow, col_k_sp, 0.0), axis=0)
    akk = tl.sum(tl.where(ridx == K, col_k_sp, 0.0), axis=0)
    tl.store(PR + b, prow)
    tl.store(AP + b, apk)
    tl.store(AK + b, akk)


@libentry()
@triton.jit
def _inv_elim_step_kernel(
    SW,
    DW,
    SX,
    DX,
    PR,
    AP,
    AK,
    N,
    K,
    LDA: tl.constexpr,
    TOT: tl.constexpr,
    BLK: tl.constexpr,
):
    """One Gauss-Jordan elimination step for a chunk of one matrix.

    The caller launches one program per chunk per step (grid ``(batch, nblk)``)
    and ping-pongs two buffers across steps so a program never reads the buffer
    it writes.  The pivot values arrive through side buffers filled by
    ``_inv_pivot_scan_kernel`` instead of an in-kernel reduction; the swap /
    normalise / eliminate arithmetic is the partial-pivoted Gauss-Jordan
    elimination described in the module docstring.
    """
    b = tl.program_id(0).to(tl.int64)
    blk = tl.program_id(1).to(tl.int64)
    base = b * TOT
    e = blk * BLK + tl.arange(0, BLK)
    row = e // LDA
    col = e % LDA

    prow = tl.load(PR + b)
    apk = tl.load(AP + b)
    akk = tl.load(AK + b)
    safe = tl.where(apk == 0.0, 1.0, apk)

    w = tl.load(SW + base + e)
    row_k = tl.load(SW + base + K * LDA + col)
    row_p = tl.load(SW + base + prow * LDA + col)
    col_k = tl.load(SW + base + row * LDA + K)

    swapped = tl.where(row == K, row_p, tl.where(row == prow, row_k, w))
    lcol = tl.where(row == K, apk, tl.where(row == prow, akk, col_k))
    urow = row_p / safe
    mult = tl.where(row != K, lcol, 0.0)
    a_new = tl.where(row == K, urow, swapped - mult * urow)

    x = tl.load(SX + base + e)
    xrow_k = tl.load(SX + base + K * LDA + col)
    xrow_p = tl.load(SX + base + prow * LDA + col)
    xswapped = tl.where(row == K, xrow_p, tl.where(row == prow, xrow_k, x))
    xurow = xrow_p / safe
    x_new = tl.where(row == K, xurow, xswapped - mult * xurow)

    tl.store(DW + base + e, a_new)
    tl.store(DX + base + e, x_new)


def _inverse_kernel_impl(A):
    n = A.shape[-1]
    batch_shape = A.shape[:-2]
    batch = 1
    for dim in batch_shape:
        batch *= dim

    if batch == 0:
        return torch.empty_like(A)

    A_work = A.reshape(batch, n, n).contiguous()
    out_work = torch.empty((batch, n, n), dtype=A.dtype, device=A.device)

    rows, lda, tot, blk, nblk = _plan_inv(n)
    device = A.device

    w0 = torch.empty(batch * tot, dtype=A.dtype, device=device)
    w1 = torch.empty(batch * tot, dtype=A.dtype, device=device)
    x0 = torch.empty(batch * tot, dtype=A.dtype, device=device)
    x1 = torch.empty(batch * tot, dtype=A.dtype, device=device)

    with torch_device_fn.device(device):
        _inv_pack_kernel[(batch, nblk)](
            A_work, w0, x0, n, LDA=lda, TOT=tot, BLK=blk, num_warps=1
        )
        # One pivot scan + one elimination step per column, ping-ponging the
        # double buffers on the host.  The single-launch in-kernel loop faults
        # with an illegal memory access on the flagtree 0.7.0 CI compiler for
        # small n, so this per-step form is used for every size.
        prow = torch.empty(batch, dtype=torch.int32, device=device)
        apk = torch.empty(batch, dtype=A.dtype, device=device)
        akk = torch.empty(batch, dtype=A.dtype, device=device)
        for k in range(n):
            if k % 2 == 0:
                sw, dw, sx, dx = w0, w1, x0, x1
            else:
                sw, dw, sx, dx = w1, w0, x1, x0
            _inv_pivot_scan_kernel[(batch,)](
                sw, prow, apk, akk, n, k, LDA=lda, ROWS=rows, num_warps=1
            )
            _inv_elim_step_kernel[(batch, nblk)](
                sw,
                dw,
                sx,
                dx,
                prow,
                apk,
                akk,
                n,
                k,
                LDA=lda,
                TOT=tot,
                BLK=blk,
                num_warps=1,
            )
        # After the loop the result lives in the buffer written by the last
        # iteration (k = n - 1).  Even k writes to X1 (xdst = X1 for even k),
        # odd k writes to X0, so the final buffer is X0 when n is even
        # (last k = n - 1 is odd) and X1 when n is odd (last k is even).
        if n % 2 == 0:
            xfinal = x0
        else:
            xfinal = x1
        _inv_unpack_kernel[(batch, nblk)](
            xfinal, out_work, n, LDA=lda, TOT=tot, BLK=blk, num_warps=1
        )

    return out_work.reshape(A.shape)


def inverse(A):
    """Invert a square matrix (batched), equivalent to ``torch.inverse``."""
    logger.debug("GEMS_KUNLUNXIN INVERSE")

    assert A.dim() >= 2, "inverse: input must be at least 2D"
    assert A.shape[-1] == A.shape[-2], "inverse: input must be a square matrix"
    assert A.dtype in (
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    ), f"inverse: unsupported dtype {A.dtype}"

    if A.numel() == 0:
        return torch.empty_like(A)

    if A.dtype in (torch.complex64, torch.complex128):
        raise NotImplementedError(
            "inverse: complex dtype is not supported by the Kunlunxin kernel"
        )

    return _inverse_kernel_impl(A)

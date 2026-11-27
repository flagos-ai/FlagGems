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

"""Public-profile ``linalg_solve_ex`` via an augmented ``[A|B]`` Gauss-Jordan.

``aten::linalg_solve_ex`` only returns ``(result, info)``; it does not expose
the LU factors.  The private ``_linalg_solve_ex`` override still has to produce
a real LU/pivots pair for autograd, and pays one launch chain per LU step.  This
module solves the public variant instead by putting ``B`` in the accumulator
slot of a Gauss-Jordan elimination:

    [ A | B ]  --partial-pivoted GJ-->  [ I | A^-1 B ]

so the leading ``n x nrhs`` block of the transformed right-hand side is the
solution, and ``info`` (first exactly-zero pivot, LAPACK order) falls out of the
same elimination for free.  The elimination kernels follow the proven backend
patterns (``inverse.py`` step kernels and ``linalg_det._det_elim_kernel``):

* every read is a plain strided 1-D/2-D vector load, never ``tl.gather``;
* the pivot search is a masked 1-D ``tl.max``/``tl.min`` over ``LDA`` lanes
  (``tl.argmax`` is unreliable on this backend);
* ``batch <= 4`` runs the whole ``for k`` chain inside one kernel with the
  coefficient/accumulator buffers double-buffered (a single-buffer in-kernel
  loop gets silently reordered by this backend); larger batches launch one
  fused pivot/swap/eliminate kernel per column, which is stream ordered and
  therefore safe with a single buffer.

Measured trap (2026-11-26): a *scalar* int32 store inside one of these vector
kernels serialises the whole kernel on this backend — putting the ``INFO`` clear
into the coefficient-pack kernel made it 196-394 us per call, against 7-8 us
with it removed.  ``INFO`` is therefore never cleared by a vector kernel: the
single-launch path stores its register-resident trace once at the end, and the
per-step chain starts its trace from zero on the ``K == 0`` launch.
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_MIN_LDA = 64
_MAX_BLK = 4096
# The single-launch elimination is exact only when the per-iteration tile is at
# least 1024 lanes, and padded small-n layouts are only reproducible for a
# handful of concurrent programs (see linalg_det._ELIM_SMALL_MAX_BATCH).
_GJ_MIN_TOT = 1024
_GJ_SMALL_MAX_BATCH = 4
# Largest n the augmented path covers; anything bigger keeps the multi-kernel
# LU implementation (the test/benchmark matrix tops out at n = 32).
_GJ_MAX_N = 32
_UNPACK_BLK = 1024


def _use_one_pass(batch):
    """Whether the whole elimination fits in one kernel for this batch."""
    return batch <= _GJ_SMALL_MAX_BATCH


def _plan_step(n):
    """Tight layout for the launch-per-column path (one program per matrix)."""
    rows = triton.next_power_of_2(n)
    if n >= 16 and n % 8 == 0 and n * n <= _MAX_BLK:
        lda = n
    else:
        lda = max(_MIN_LDA, rows)
    tot = rows * lda
    blk = min(_MAX_BLK, tot)
    return rows, lda, tot, blk


def _plan_small(n):
    """Layout for the single-launch path: pad the tile to >= 1024 lanes."""
    rows = triton.next_power_of_2(n)
    lda = max(_MIN_LDA, rows)
    while rows * lda < _GJ_MIN_TOT:
        lda *= 2
    tot = rows * lda
    return rows, lda, tot, tot


@libentry()
@triton.jit
def _gj_pack_a_kernel(
    A,
    W,
    N,
    LDA: tl.constexpr,
    TOT: tl.constexpr,
    BLK: tl.constexpr,
):
    """Scatter contiguous (batch, N, N) into the padded (batch, ROWS, LDA)
    coefficient tile.

    Padding lanes are written as zeros (the packed tile always has ROWS >= N and
    LDA >= N).
    """
    b = tl.program_id(0).to(tl.int64)
    blk = tl.program_id(1).to(tl.int64)
    e = blk * BLK + tl.arange(0, BLK)
    row = e // LDA
    col = e % LDA
    live = (row < N) & (col < N)
    idx = tl.where(live, row * N + col, 0)
    val = tl.load(A + b * N * N + idx)
    tl.store(W + b * TOT + e, tl.where(live, val, 0.0))


@libentry()
@triton.jit
def _gj_seed_kernel(
    B,
    X,
    N,
    NRHS,
    LDA: tl.constexpr,
    TOT: tl.constexpr,
    BLK: tl.constexpr,
):
    """Seed the accumulator's leading N x NRHS block with the right-hand side.

    Only the live block is written: padding lanes of ``X`` never feed a live
    lane (row operations only ever gather ``X[k]``/``X[prow]`` with ``k, prow <
    N`` and update a column from itself), so they can stay uninitialised.
    """
    b = tl.program_id(0).to(tl.int64)
    blk = tl.program_id(1).to(tl.int64)
    e = blk * BLK + tl.arange(0, BLK)
    mask = e < N * NRHS
    row = e // NRHS
    col = e % NRHS
    # Masked-out lanes address past the N x NRHS block; clamp their source and
    # destination indices (the compiler computes the address before the mask).
    src = tl.where(mask, e, 0)
    dst = tl.where(mask, row * LDA + col, 0)
    val = tl.load(B + b * N * NRHS + src, mask=mask, other=0.0)
    tl.store(X + b * TOT + dst, val, mask=mask)


@libentry()
@triton.jit
def _gj_elim_kernel(
    W0,
    W1,
    X0,
    X1,
    INFO,
    N,
    ROWS: tl.constexpr,
    LDA: tl.constexpr,
    TOT: tl.constexpr,
    BLK: tl.constexpr,
):
    """Whole partial-pivoted Gauss-Jordan chain for one matrix, one launch.

    ``W`` (coefficient tile, seeded with ``A``) and ``X`` (accumulator, seeded
    with ``B``) are both double buffered: iteration ``k`` reads the buffer pair
    selected by ``k``'s parity and writes the other, so no in-kernel load can be
    reordered onto a buffer being written.  ``info`` is kept register-resident
    as a scalar (a global write/read inside the loop is exactly the pattern this
    backend reorders).
    """
    b = tl.program_id(0).to(tl.int64)
    base = b * TOT
    ridx = tl.arange(0, LDA)
    # ``LDA`` can exceed ``ROWS`` once small n is padded up; clamp the scan's row
    # index so the padding lanes re-read the last valid row instead of past the
    # buffer (the compiler computes the address before any mask applies).
    rclamp = tl.minimum(ridx, ROWS - 1)
    singular = 0
    for k in range(0, N):
        is_even = (k % 2) == 0
        wsrc = tl.where(is_even, W0, W1)
        wdst = tl.where(is_even, W1, W0)
        xsrc = tl.where(is_even, X0, X1)
        xdst = tl.where(is_even, X1, X0)
        col_k_sp = tl.load(wsrc + base + rclamp * LDA + k)
        cand = tl.where((ridx >= k) & (ridx < N), tl.abs(col_k_sp), -1.0)
        best = tl.max(cand, axis=0)
        prow = tl.min(tl.where(cand == best, ridx, LDA), axis=0)
        akk = tl.sum(tl.where(ridx == k, col_k_sp, 0.0), axis=0)
        apk = tl.sum(tl.where(ridx == prow, col_k_sp, 0.0), axis=0)
        safe = tl.where(apk == 0.0, 1.0, apk)
        singular = tl.where((apk == 0.0) & (singular == 0), k + 1, singular)
        for c in range(0, TOT // BLK):
            e = c * BLK + tl.arange(0, BLK)
            row = e // LDA
            col = e % LDA
            w = tl.load(wsrc + base + e)
            row_k = tl.load(wsrc + base + k * LDA + col)
            row_p = tl.load(wsrc + base + prow * LDA + col)
            col_k = tl.load(wsrc + base + row * LDA + k)
            x = tl.load(xsrc + base + e)
            xrow_k = tl.load(xsrc + base + k * LDA + col)
            xrow_p = tl.load(xsrc + base + prow * LDA + col)
            swapped = tl.where(row == k, row_p, tl.where(row == prow, row_k, w))
            lcol = tl.where(row == k, apk, tl.where(row == prow, akk, col_k))
            urow = row_p / safe
            mult = tl.where(row != k, lcol, 0.0)
            xswapped = tl.where(row == k, xrow_p, tl.where(row == prow, xrow_k, x))
            xurow = xrow_p / safe
            tl.store(
                wdst + base + e,
                tl.where(row == k, urow, swapped - mult * urow),
            )
            tl.store(
                xdst + base + e,
                tl.where(row == k, xurow, xswapped - mult * xurow),
            )
    tl.store(INFO + b, singular.to(tl.int32))


@libentry()
@triton.jit(do_not_specialize=["K"])
def _gj_step_kernel(
    W,
    X,
    INFO,
    N,
    K,
    ROWS: tl.constexpr,
    LDA: tl.constexpr,
    TOT: tl.constexpr,
):
    """One fused pivot-scan + swap + Gauss-Jordan step, in place.

    One program owns the whole matrix, and ``K`` is a host-side loop variable,
    so the kernel never sits inside a runtime loop around a global
    store->load round trip — the safe per-step form on this backend.
    """
    b = tl.program_id(0).to(tl.int64)
    base = b * TOT
    e = tl.arange(0, TOT)
    row = e // LDA
    col = e % LDA
    ridx = tl.arange(0, LDA)
    rclamp = tl.minimum(ridx, ROWS - 1)
    col_k_sp = tl.load(W + base + rclamp * LDA + K)
    cand = tl.where((ridx >= K) & (ridx < N), tl.abs(col_k_sp), -1.0)
    best = tl.max(cand, axis=0)
    prow = tl.min(tl.where(cand == best, ridx, LDA), axis=0)
    akk = tl.sum(tl.where(ridx == K, col_k_sp, 0.0), axis=0)
    apk = tl.sum(tl.where(ridx == prow, col_k_sp, 0.0), axis=0)
    safe = tl.where(apk == 0.0, 1.0, apk)
    # The first step (K == 0) starts the trace from 0, so INFO needs no
    # pre-fill; the discarded load of an uninitialised value only feeds the
    # non-first branches of the select.
    prev = tl.where(K == 0, 0, tl.load(INFO + b))
    w = tl.load(W + base + e)
    row_k = tl.load(W + base + K * LDA + col)
    row_p = tl.load(W + base + prow * LDA + col)
    col_k = tl.load(W + base + row * LDA + K)
    x = tl.load(X + base + e)
    xrow_k = tl.load(X + base + K * LDA + col)
    xrow_p = tl.load(X + base + prow * LDA + col)
    swapped = tl.where(row == K, row_p, tl.where(row == prow, row_k, w))
    lcol = tl.where(row == K, apk, tl.where(row == prow, akk, col_k))
    urow = row_p / safe
    mult = tl.where(row != K, lcol, 0.0)
    xswapped = tl.where(row == K, xrow_p, tl.where(row == prow, xrow_k, x))
    xurow = xrow_p / safe
    tl.store(W + base + e, tl.where(row == K, urow, swapped - mult * urow))
    tl.store(X + base + e, tl.where(row == K, xurow, xswapped - mult * xurow))
    tl.store(INFO + b, tl.where((apk == 0.0) & (prev == 0), K + 1, prev))


@libentry()
@triton.jit
def _gj_unpack_kernel(
    X,
    OUT,
    N,
    NRHS,
    LDA: tl.constexpr,
    TOT: tl.constexpr,
    BLK: tl.constexpr,
):
    """Copy the leading N x NRHS block of the accumulator back out."""
    b = tl.program_id(0).to(tl.int64)
    blk = tl.program_id(1).to(tl.int64)
    e = blk * BLK + tl.arange(0, BLK)
    mask = e < N * NRHS
    row = e // NRHS
    col = e % NRHS
    # Masked-out lanes address past the N x NRHS block; clamp their source and
    # destination indices (the compiler computes the address before the mask).
    src = tl.where(mask, row * LDA + col, 0)
    dst = tl.where(mask, e, 0)
    val = tl.load(X + b * TOT + src, mask=mask, other=0.0)
    tl.store(OUT + b * N * NRHS + dst, val, mask=mask)


def _launch_gj(A_work, B_work, out_work, info, batch, n, nrhs, dtype, device):
    """Run the augmented Gauss-Jordan for contiguous (batch, n, n) inputs."""
    one_pass = _use_one_pass(batch)
    rows, lda, tot, blk = _plan_small(n) if one_pass else _plan_step(n)
    nblk = tot // blk
    seed_blk = min(_UNPACK_BLK, triton.next_power_of_2(n * nrhs))
    seed_nblk = triton.cdiv(n * nrhs, seed_blk)

    with torch_device_fn.device(device):
        if one_pass:
            w0 = torch.empty(batch * tot, dtype=dtype, device=device)
            w1 = torch.empty(batch * tot, dtype=dtype, device=device)
            x0 = torch.empty(batch * tot, dtype=dtype, device=device)
            x1 = torch.empty(batch * tot, dtype=dtype, device=device)
            _gj_pack_a_kernel[(batch, nblk)](
                A_work,
                w0,
                n,
                LDA=lda,
                TOT=tot,
                BLK=blk,
                num_warps=1,
            )
            _gj_seed_kernel[(batch, seed_nblk)](
                B_work,
                x0,
                n,
                nrhs,
                LDA=lda,
                TOT=tot,
                BLK=seed_blk,
                num_warps=1,
            )
            _gj_elim_kernel[(batch,)](
                w0,
                w1,
                x0,
                x1,
                info,
                n,
                ROWS=rows,
                LDA=lda,
                TOT=tot,
                BLK=blk,
                num_warps=2,
            )
            # Last iteration k = n - 1 writes the opposite buffer when n is
            # even (the other one when n is odd).
            xfinal = x0 if n % 2 == 0 else x1
        else:
            w = torch.empty(batch * tot, dtype=dtype, device=device)
            x = torch.empty(batch * tot, dtype=dtype, device=device)
            _gj_pack_a_kernel[(batch, nblk)](
                A_work,
                w,
                n,
                LDA=lda,
                TOT=tot,
                BLK=blk,
                num_warps=1,
            )
            _gj_seed_kernel[(batch, seed_nblk)](
                B_work,
                x,
                n,
                nrhs,
                LDA=lda,
                TOT=tot,
                BLK=seed_blk,
                num_warps=1,
            )
            for k in range(n):
                _gj_step_kernel[(batch,)](
                    w,
                    x,
                    info,
                    n,
                    k,
                    ROWS=rows,
                    LDA=lda,
                    TOT=tot,
                    num_warps=1,
                )
            xfinal = x
        _gj_unpack_kernel[(batch, seed_nblk)](
            xfinal,
            out_work,
            n,
            nrhs,
            LDA=lda,
            TOT=tot,
            BLK=seed_blk,
            num_warps=1,
        )


def _solve_ex_public(A, B, *, left=True, check_errors=False):
    """Solve ``AX = B`` returning ``(result, info)`` for the public profile."""
    from ._linalg_solve_ex import (
        _TRITON_DTYPES,
        _check_solve_ex_args,
        _linalg_solve_ex,
        _raise_singular,
    )

    vector_rhs, B_mat = _check_solve_ex_args(A, B, left)
    n = A.shape[-1]
    a_batch_shape = A.shape[:-2]
    batch_shape = torch.broadcast_shapes(  # allow-torch-compute: shape metadata only
        a_batch_shape, B_mat.shape[:-2]
    )
    rows, cols = B_mat.shape[-2:]

    if (
        A.dtype not in _TRITON_DTYPES
        or not left
        or A.numel() == 0
        or n < 2
        or n > _GJ_MAX_N
    ):
        # Complex inputs keep the vendor capture, ``XA = B`` and the empty case
        # keep the LU-based implementation; all of them are Triton or the
        # pre-existing captured vendor kernels, never a CPU fallback.
        result, _lu, _pivots, info = _linalg_solve_ex(
            A, B, left=left, check_errors=check_errors
        )
        return result, info

    batch = math.prod(batch_shape) if batch_shape else 1
    A_work = A.expand(batch_shape + (n, n)).reshape(batch, n, n).contiguous()
    out_work = torch.empty((batch, n, cols), dtype=A.dtype, device=A.device)
    # Both elimination paths write INFO themselves: the single-launch one stores
    # the register-resident trace once at the end, the per-step chain rewrites
    # it from its first K == 0 launch on, so no host-side zero fill is needed.
    info = torch.empty(batch_shape, dtype=torch.int32, device=A.device)
    B_work = B_mat.expand(batch_shape + (rows, cols)).reshape(
        batch, rows, cols
    ).contiguous()

    _launch_gj(
        A_work,
        B_work,
        out_work,
        info.reshape(batch),
        batch,
        n,
        cols,
        A.dtype,
        A.device,
    )

    if check_errors and bool((info != 0).any()):
        _raise_singular(info)

    result = out_work.reshape(batch_shape + (rows, cols))
    if vector_rhs:
        result = result.squeeze(-1)

    # ``info`` describes A alone, so it carries A's own batch shape (ATen
    # returns e.g. shape () for a 2-D A solved against a batched B).  When B
    # widened the batch the extra copies are identical; drop them.
    info = info.reshape(batch_shape)
    if batch_shape != a_batch_shape:
        offset = len(batch_shape) - len(a_batch_shape)
        index = tuple(
            slice(None) if dim == a_dim else slice(0, a_dim)
            for dim, a_dim in zip(
                batch_shape[len(batch_shape) - len(a_batch_shape) :],
                a_batch_shape,
            )
        )
        info = info[(0,) * offset + index]
    return result, info

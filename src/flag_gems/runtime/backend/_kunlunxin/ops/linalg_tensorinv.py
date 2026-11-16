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

"""Kunlunxin backend override for ``linalg_tensorinv`` (XPU3).

Semantics
---------
``aten::linalg_tensorinv(A, ind)`` reshapes ``A`` to
``(m, m)`` with ``m = prod(A.shape[:ind]) == prod(A.shape[ind:])`` and returns
``B^{-1}`` viewed as ``A.shape[ind:] + A.shape[:ind]`` (the flat element order
of the result equals ``B^{-1}``'s, so the output is one dense inversion).

Why a dedicated implementation exists
-------------------------------------
The generic ``flag_gems/ops/linalg_tensorinv.py`` keeps the working matrix and
the transform in register tiles and extracts columns with
``tl.sum(tl.where(cols == k, a, 0), axis=1)`` over an *arithmetic* 2-D tile (and
uses ``tl.argmax``).  TritonXPU cannot legalize that -- every call dies at
``linalg_tensorinv.py:101: 'arith.cmpi' op requires all operands to have the
same type`` with ``TritonXPULegalize`` / ``uni_sram``, so HEAD launches **no**
kernel at all.

The backend already has a gather-free pure-Triton Gauss-Jordan
(``_kunlunxin/ops/inverse.py``, ``kernel-inverse-kunlunxin``), but it cannot be
reused directly here:

* ``inverse`` rejects ``float16`` (``assert A.dtype in (fp32, fp64, complex)``)
  while the tensorinv suite parametrizes half inputs;
* ``inverse`` deliberately clamps an exactly-zero pivot with
  ``safe = where(apk == 0, 1, apk)``, so a singular input yields a *finite*
  wrong result, whereas ATen ``tensorinv`` must propagate ``inf``/``nan``
  (``tests/test_linalg_tensorinv.py::test_linalg_tensorinv_singular``).

This module therefore re-implements the *same proven Gauss-Jordan pattern*
locally (one source of truth for ``inverse`` remains untouched):

* ``pack`` scatters ``A`` into padded ``W`` and sets ``X = I``; the padding is
  the identity, so the padded system is block diagonal and the leading
  ``n x n`` block of ``X`` is ``A^{-1}``.  The same kernel upcasts ``fp16`` /
  ``bf16`` to the ``float32`` working buffers.
* ``n <= 32`` -- the whole ``for k`` chain runs in one kernel, ``W``/``X``
  pairs ping-ponged across iterations (single-buffer in-kernel loops are
  silently reordered on this backend).  Partial pivoting uses
  ``tl.max`` + ``tl.min(where(...))`` (never ``tl.argmax``); every reduction
  runs on a load-rooted 1-D column.  The pivot divisor is used **raw** so an
  exact-zero pivot produces ``inf``/``nan`` like ATen.
* ``n > 32`` -- the per-step chunked pair from ``inverse`` (pivot/swap/normalise
  launch + elimination launch), again with the raw pivot.  This layout
  (``AW >= 64``) is the regime in which the per-step kernels were measured
  exact on this stack.

``pack`` and ``unpack`` stay separate launches: a same-kernel global
write-then-read is reordered on XPU3 (``pattern-xpu3-same-kernel-global-raw-hazard``).
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# In-kernel Gauss-Jordan iteration floor (``inverse._ELIM_MIN_TOT``): the loop
# is only lowered exactly with a >= 1024-lane iteration tile.
_ELIM_MIN_TOT = 1024
# Row width above which the in-kernel loop is replaced by the per-step chunked
# pair (``inverse`` gates its own single-launch loop at rows <= 32).
_MAX_INKERNEL_ROWS = 32
# Chunk width for the pack / eliminate / unpack helper kernels.
_CHUNK_BLK = 1024


def _check_inv_input(A, ind):
    """Validate input exactly like the generic implementation / ATen."""
    if ind <= 0:
        raise RuntimeError(
            "linalg.tensorinv: Expected a strictly positive integer for "
            f"'ind', but got {ind}"
        )
    if A.dim() < 2:
        raise RuntimeError(
            "linalg.tensorinv: Expected input to be at least 2D, " f"got {A.dim()}D"
        )
    if ind > A.dim():
        raise RuntimeError(
            "linalg.tensorinv: Expected 0 <= ind <= input.dim(), "
            f"got ind={ind} for input.dim()={A.dim()}"
        )
    m = 1
    for i in range(ind):
        m *= A.shape[i]
    n = 1
    for i in range(ind, A.dim()):
        n *= A.shape[i]
    if m != n:
        raise RuntimeError(
            "linalg.tensorinv: Expected self to satisfy the requirement "
            "prod(self.shape[ind:]) == prod(self.shape[:ind]), "
            f"but got {n} != {m}"
        )
    return int(m)


def _plan_pad(n):
    """Square-ish padded tile for the in-kernel loop.

    ``rows`` is the live row count rounded up to a power of two; the row stride
    ``aw`` is kept square where that already reaches the 1024-lane floor (for
    ``n = 32`` this is a 32 x 32 tile, 19% faster than padding to a 64-wide
    stride), and doubled below it.  A stride >= 8 avoids the backend's
    narrow-stride miscompile.
    """
    rows = triton.next_power_of_2(n)
    aw = rows if rows >= 8 else 64
    while rows * aw < _ELIM_MIN_TOT:
        aw *= 2
    return rows, aw, rows * aw


def _plan_chunk(n):
    """Padded tile for the per-step chunked path (wide row stride)."""
    rows = triton.next_power_of_2(n)
    aw = max(rows, 64)
    return rows, aw, rows * aw


@triton.jit
def _tinv_pack_kernel(A, W, X, N, AW: tl.constexpr, TOT: tl.constexpr, BLK: tl.constexpr):
    """Scatter ``A`` into padded ``W`` (cast to ``W.dtype``) and set ``X = I``.

    ``BLK`` divides ``TOT`` exactly, so the tile arithmetic stays int32 and no
    masked store is needed.  Padding lanes get the identity (block-diagonal
    padded system).  Out-of-range source indices are clamped and the loaded
    value discarded (``other=`` pollutes live lanes on this backend).
    """
    pid = tl.program_id(0)
    e = pid * BLK + tl.arange(0, BLK)
    row = e // AW
    col = e % AW
    live = (row < N) & (col < N)
    idx = tl.where(live, row * N + col, 0)
    a = tl.load(A + idx).to(W.dtype.element_ty)
    ident = (row == col).to(W.dtype.element_ty)
    tl.store(W + e, tl.where(live, a, ident))
    tl.store(X + e, ident)


@triton.jit
def _tinv_gj_kernel(
    W0,
    W1,
    X0,
    X1,
    N,
    AW: tl.constexpr,
    ROWS: tl.constexpr,
    TOT: tl.constexpr,
):
    """Whole Gauss-Jordan elimination in one launch (one matrix).

    ``W``/``X`` pairs are ping-ponged: each iteration writes the pair it did
    not read, so the last written pair has the ``(n - 1)`` parity and the
    caller picks it.  The pivot is used raw (``inf`` on an exact-zero pivot,
    matching ATen's singular behaviour).
    """
    ridx = tl.arange(0, ROWS)
    live_rows = ridx < N
    for k in range(0, N):
        is_even = (k % 2) == 0
        src_w = tl.where(is_even, W0, W1)
        dst_w = tl.where(is_even, W1, W0)
        src_x = tl.where(is_even, X0, X1)
        dst_x = tl.where(is_even, X1, X0)

        # Pivot search on the load-rooted 1-D column (never tl.argmax).
        col_k_sp = tl.load(src_w + ridx * AW + k)
        cand = tl.where(live_rows & (ridx >= k), tl.abs(col_k_sp), -1.0)
        best = tl.max(cand, axis=0)
        prow = tl.min(tl.where(cand == best, ridx, ROWS - 1), axis=0)
        akk = tl.sum(tl.where(ridx == k, col_k_sp, 0.0), axis=0)
        apk = tl.sum(tl.where(ridx == prow, col_k_sp, 0.0), axis=0)

        e = tl.arange(0, TOT)
        row = e // AW
        col = e % AW
        w = tl.load(src_w + e)
        x = tl.load(src_x + e)
        row_k_w = tl.load(src_w + k * AW + col)
        row_p_w = tl.load(src_w + prow * AW + col)
        row_k_x = tl.load(src_x + k * AW + col)
        row_p_x = tl.load(src_x + prow * AW + col)
        col_k = tl.load(src_w + e - col + k)
        swapped_w = tl.where(row == prow, row_k_w, w)
        swapped_x = tl.where(row == prow, row_k_x, x)
        lcol = tl.where(row == prow, akk, col_k)
        # Raw pivot: an exact-zero divisor yields inf/nan, as ATen does.
        mult = tl.where(row == k, 0.0, lcol / apk)
        new_w = swapped_w - mult * row_p_w
        new_x = swapped_x - mult * row_p_x
        new_w = tl.where(row == k, row_p_w / apk, new_w)
        new_x = tl.where(row == k, row_p_x / apk, new_x)
        new_w = tl.where(col == k, tl.where(row == k, 1.0, 0.0), new_w)
        tl.store(dst_w + e, new_w)
        tl.store(dst_x + e, new_x)


@triton.jit
def _tinv_pivot_swap_norm_kernel(
    W,
    X,
    FACT,
    N,
    K,
    AW: tl.constexpr,
    ROWS: tl.constexpr,
):
    """Pivot search + physical swap + row-K normalisation for one step.

    Separate launch so the following eliminate launch never writes the pivot
    row it reads.  The post-swap elimination factors are saved to ``FACT``.
    The pivot is divided raw (``inf`` on an exact-zero pivot).
    """
    ridx = tl.arange(0, ROWS)
    live_rows = ridx < N
    col_k = tl.load(W + ridx * AW + K)
    cand = tl.where(live_rows & (ridx >= K), tl.abs(col_k), -1.0)
    best = tl.max(cand, axis=0)
    prow = tl.min(tl.where(cand == best, ridx, ROWS - 1), axis=0)
    akk = tl.sum(tl.where(ridx == K, col_k, 0.0), axis=0)
    apk = tl.sum(tl.where(ridx == prow, col_k, 0.0), axis=0)

    cols = tl.arange(0, AW)
    row_k_w = tl.load(W + K * AW + cols)
    row_p_w = tl.load(W + prow * AW + cols)
    row_k_x = tl.load(X + K * AW + cols)
    row_p_x = tl.load(X + prow * AW + cols)

    same = prow == K
    norm_w = row_p_w / apk
    norm_x = row_p_x / apk
    tl.store(W + K * AW + cols, norm_w)
    tl.store(X + K * AW + cols, norm_x)
    tl.store(W + prow * AW + cols, tl.where(same, norm_w, row_k_w))
    tl.store(X + prow * AW + cols, tl.where(same, norm_x, row_k_x))

    fact = tl.where(ridx == K, 0.0, tl.where(ridx == prow, akk, col_k))
    tl.store(FACT + ridx, fact)


@triton.jit
def _tinv_elim_kernel(
    W,
    X,
    FACT,
    K,
    AW: tl.constexpr,
    TOT: tl.constexpr,
    BLK: tl.constexpr,
):
    """Rank-1 elimination of the remaining rows for one step.

    ``BLK`` divides ``TOT`` exactly (no masked store).  Reads the stable pivot
    row K and the per-row factors; writes every row ``i != K`` of both tiles.
    """
    blk = tl.program_id(0)
    e = blk * BLK + tl.arange(0, BLK)
    row = e // AW
    col = e % AW
    is_k = row == K
    factor = tl.load(FACT + row)
    row_k_w = tl.load(W + K * AW + col)
    row_k_x = tl.load(X + K * AW + col)
    w = tl.load(W + e)
    x = tl.load(X + e)
    new_w = tl.where(col == K, 0.0, w - factor * row_k_w)
    new_x = x - factor * row_k_x
    tl.store(W + e, tl.where(is_k, w, new_w))
    tl.store(X + e, tl.where(is_k, x, new_x))


@triton.jit
def _tinv_unpack_flat_kernel(
    X,
    OUT,
    AW: tl.constexpr,
    NC: tl.constexpr,
    CU: tl.constexpr,
    PAD: tl.constexpr,
    BLK: tl.constexpr,
):
    """Copy the leading ``NC x NC`` block of ``X`` into a flat output.

    The tile walks a ``CU x CU`` square (``CU = next_pow2(NC)``), so the only
    integer division is by the compile-time power-of-two ``CU`` (a shift);
    dividing the flat index by ``NC`` (e.g. 24) was measured ~3x more expensive
    on XPU3.  Padded lanes are *wrapped* back onto the leading block
    (``row - NC`` / ``col - NC``), which only requires ``CU < 2 * NC``: every
    lane then stores the value of a live element, so all redundant writes are
    identical and idempotent.  (Masked stores and "clamp all padding to the
    last element" both misbehave here: the former is unreliable, the latter
    serialises hundreds of same-address writes.)
    """
    pid = tl.program_id(0)
    e = pid * BLK + tl.arange(0, BLK)
    row = e // CU
    col = e - row * CU
    if PAD:
        r2 = tl.where(row < NC, row, row - NC)
        c2 = tl.where(col < NC, col, col - NC)
    else:
        r2 = row
        c2 = col
    val = tl.load(X + r2 * AW + c2)
    tl.store(OUT + r2 * NC + c2, val.to(OUT.dtype.element_ty))


@triton.jit
def _tinv_unpack_strided_kernel(
    X,
    OUT,
    OS0,
    OS1,
    AW: tl.constexpr,
    NC: tl.constexpr,
    CU: tl.constexpr,
    PAD: tl.constexpr,
    BLK: tl.constexpr,
):
    """Copy the leading ``NC x NC`` block of ``X`` into a strided 2-D output.

    The destination offsets are computed as a 1-D vector but stored as a
    ``(BLK, 1)`` tile: a plain 1-D store with a non-unit per-lane stride
    silently drops lanes on XPU3
    (``pattern-xpu3-1d-strided-vector-store-drops-lanes``), and a full 2-D
    ``(CU, CU)`` tile trips the legalizer for padded ``NC``.  Padded lanes wrap
    onto the leading block so the redundant writes are identical.
    """
    pid = tl.program_id(0)
    e = pid * BLK + tl.arange(0, BLK)
    row = e // CU
    col = e - row * CU
    if PAD:
        r2 = tl.where(row < NC, row, row - NC)
        c2 = tl.where(col < NC, col, col - NC)
    else:
        r2 = row
        c2 = col
    val = tl.load(X + r2 * AW + c2)
    tl.store(
        OUT + (r2 * OS0 + c2 * OS1)[:, None], val.to(OUT.dtype.element_ty)[:, None]
    )


def linalg_tensorinv(A, ind=2, *, out=None):
    """Compute the multiplicative inverse of ``tensordot``.

    Args:
        A: Input tensor with ``prod(A.shape[:ind]) == prod(A.shape[ind:])``.
        ind: Index at which to compute the inverse of ``tensordot``.
        out: Optional output tensor.

    Returns:
        ``X`` such that ``tensordot(A, X, ind)`` is the identity, with shape
        ``A.shape[ind:] + A.shape[:ind]``.
    """
    logger.debug("GEMS_KUNLUNXIN LINALG_TENSORINV")
    m = _check_inv_input(A, ind)

    shape = A.shape
    output_shape = shape[ind:] + shape[:ind]
    dtype = A.dtype
    device = A.device

    if dtype in (torch.complex64, torch.complex128):
        raise NotImplementedError(
            "Kunlunxin linalg_tensorinv does not support complex dtypes: the "
            "XPU3 stack provides no complex linear-algebra primitive"
        )

    if m == 0:
        if out is None:
            return torch.empty(output_shape, dtype=dtype, device=device)
        return out

    # Contiguous view of the flattened (m, m) system; a strided input is
    # copied once (data movement only).
    src = A.contiguous().reshape(m, m)

    rows, aw, tot = _plan_pad(m)
    use_inkernel = rows <= _MAX_INKERNEL_ROWS
    if not use_inkernel:
        rows, aw, tot = _plan_chunk(m)

    # float32 working buffers: fp16/bf16 inputs are upcast by the pack kernel.
    w0 = torch.empty(tot, dtype=torch.float32, device=device)
    x0 = torch.empty(tot, dtype=torch.float32, device=device)
    blk = min(_CHUNK_BLK, tot)

    with torch_device_fn.device(device):
        _tinv_pack_kernel[(tot // blk,)](
            src, w0, x0, m, AW=aw, TOT=tot, BLK=blk, num_warps=1
        )
        if use_inkernel:
            w1 = torch.empty(tot, dtype=torch.float32, device=device)
            x1 = torch.empty(tot, dtype=torch.float32, device=device)
            _tinv_gj_kernel[(1,)](
                w0, w1, x0, x1, m, AW=aw, ROWS=rows, TOT=tot, num_warps=2
            )
            # Each iteration writes the pair it did not read, so the last
            # written pair has the (n - 1) parity.
            final_x = x1 if (m % 2 == 1) else x0
        else:
            fact = torch.empty(rows, dtype=torch.float32, device=device)
            for k in range(m):
                _tinv_pivot_swap_norm_kernel[(1,)](
                    w0, x0, fact, m, k, AW=aw, ROWS=rows, num_warps=1
                )
                _tinv_elim_kernel[(tot // blk,)](
                    w0, x0, fact, k, AW=aw, TOT=tot, BLK=blk, num_warps=4
                )
            final_x = x0

        # Unpack walks a CU x CU square (CU = next_pow2(m)), so its tile is a
        # power of two that covers the whole leading m x m block; padded lanes
        # wrap onto live elements (identical redundant writes).
        cu = triton.next_power_of_2(m)
        pad = cu != m
        ublk = min(_CHUNK_BLK, cu * cu)
        ugrid = ((cu * cu) // ublk,)

        if out is None:
            res = torch.empty(output_shape, dtype=dtype, device=device)
            _tinv_unpack_flat_kernel[ugrid](
                final_x,
                res.reshape(-1),
                AW=aw,
                NC=m,
                CU=cu,
                PAD=pad,
                BLK=ublk,
                num_warps=1,
            )
            return res

        # out= path: write into the caller's buffer (contiguous flat fast path,
        # otherwise honour its 2-D strides).  Mirrors the generic handling.
        if out.is_contiguous():
            dst = out.view(m, m)
        elif out.dim() == 2:
            dst = out
        else:
            dst = out.reshape(m, m)
        if dst.is_contiguous():
            _tinv_unpack_flat_kernel[ugrid](
                final_x,
                dst.reshape(-1),
                AW=aw,
                NC=m,
                CU=cu,
                PAD=pad,
                BLK=ublk,
                num_warps=1,
            )
        else:
            os0, os1 = dst.stride()
            _tinv_unpack_strided_kernel[ugrid](
                final_x,
                dst,
                os0,
                os1,
                AW=aw,
                NC=m,
                CU=cu,
                PAD=pad,
                BLK=ublk,
                num_warps=1,
            )
        return out


def linalg_tensorinv_out(A, ind=2, *, out=None):
    """Out-of-place variant writing the tensor inverse into ``out``."""
    logger.debug("GEMS_KUNLUNXIN LINALG_TENSORINV_OUT")
    return linalg_tensorinv(A, ind=ind, out=out)

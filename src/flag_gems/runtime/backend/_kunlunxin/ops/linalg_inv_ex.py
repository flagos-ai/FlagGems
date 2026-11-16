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

"""Kunlunxin backend override for ``linalg_inv_ex`` (XPU3).

``aten::linalg_inv_ex`` computes the same ``A^{-1}`` as ``aten::inverse`` but
also returns a per-matrix ``info`` vector (``0`` = success, ``k + 1`` = the
``k``-th pivot of the pivoted elimination is exactly zero).  It is reached
through a separate ATen entry point, and the generic
``flag_gems/ops/linalg_inv_ex.py`` kernel does not compile on TritonXPU (it
keeps the augmented tile in registers and reduces over arithmetic 2-D tiles):

    linalg_inv_ex.py:98: 'arith.cmpi' op requires all operands to have the same type
    TritonXPULegalize ... Failures have been detected while processing an MLIR pass pipeline

so HEAD fails all 27 test cells at kernel-compile time.

The algorithm is the backend-local, gather-free Triton Gauss-Jordan with
partial pivoting already proven on this stack for ``inverse``
(``_kunlunxin/ops/inverse.py``, see ``harness/solution/inverse/README.md``);
no CPU/ATen/native/composite fallback is involved.  Two dispatches are used:

* **one pass** (``batch <= 4`` and ``n <= 32``): the whole ``for k`` chain runs
  in one kernel on padded ``[W | X]`` buffers ping-ponged across iterations
  (the ``inverse._inv_gj_kernel`` regime).  The pivot sequence comes out of the
  same elimination for free, so ``info`` costs one extra store per matrix --
  no second elimination.  This covers every benchmark shape except
  ``(16, 32, 32)``.
* **two passes** otherwise: the inverse is delegated to the public
  ``inverse`` (the same backend-local Triton path -- never a vendor call) and
  ``info`` gets a narrower elimination on ``A`` alone that records the first
  exactly-zero pivot.

The one-pass path is duplicated here rather than added to ``inverse.py``
because ``linalg_inv_ex`` needs the singularity trace emitted from inside the
elimination, while ``inverse`` must keep its exact public contract; the two
regimes are gated identically.

Reductions run only on load-rooted 1-D column vectors (never a second axis of
an arithmetic 2-D tile); ``tl.argmax``, ``tl.trans`` and single-buffer
in-kernel loops are avoided -- all broken on XPU3 (see the wiki entries).
"""

import logging
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

from .inverse import inverse as _inverse

logger = logging.getLogger(__name__)

LinalgInvExResult = namedtuple("LinalgInvExResult", ["inverse", "info"])

# Small-iteration-tile in-kernel loop floor (``inverse._ELIM_MIN_TOT``).
_ELIM_MIN_TOT = 1024
# Batch bound for the single-launch in-kernel loop.  ``inverse`` raised its own
# gate to 8 after a forced-path sweep; this file keeps the conservative value
# the pattern was originally measured exact with (``linalg_det.
# _ELIM_SMALL_MAX_BATCH``), because the info trace is an extra loop-carried
# value.  Probing the in-kernel info loop at batch 16 reproduced the documented
# silent corruption (wrong info for a zero-column matrix).
_ELIM_MAX_BATCH = 4


def _plan(n):
    """Padded working layout for the per-step info pass (``inverse._plan``)."""
    rows = triton.next_power_of_2(n)
    aw = rows if rows >= 8 else 64
    tot = rows * aw
    blk = min(1024, tot)
    return rows, aw, tot, blk


def _plan_pad(n):
    """Layout for the in-kernel Gauss-Jordan loop.

    Same >= 1024-lane iteration tile as ``inverse._plan_pad`` (the in-kernel
    loop is only exact at or above that width) but the row stride is kept
    square where that already reaches the floor: for ``n = 32`` this yields a
    ``32 x 32`` tile instead of padding to a 64-wide stride, measured 19%
    faster for the one-pass kernel.  A 32-wide stride is far above the
    ``< 8`` narrow-stride miscompile threshold.
    """
    rows = triton.next_power_of_2(n)
    aw = rows if rows >= 8 else 64
    while rows * aw < _ELIM_MIN_TOT:
        aw *= 2
    tot = rows * aw
    return rows, aw, tot


def _use_one_pass(batch, n):
    return batch <= _ELIM_MAX_BATCH and triton.next_power_of_2(n) <= 32


@libentry()
@triton.jit
def _inv_ex_pack_gj_kernel(
    A,
    W,
    X,
    INFO,
    N,
    AW: tl.constexpr,
    TOT: tl.constexpr,
    BLK: tl.constexpr,
):
    """Scatter ``A`` into padded ``W`` and initialise ``X = I`` (one pass).

    ``BLK`` divides ``TOT`` exactly, so the tile arithmetic stays int32 and no
    masked store is needed.  Padding lanes are filled with the identity rather
    than zero: the padded system is then block diagonal and the leading
    ``n x n`` block of the accumulated inverse is exactly ``A^-1``.  Out-of-range
    source indices are clamped and the loaded value discarded (``other=``
    pollutes live lanes on this backend).

    ``INFO`` is reset here so no extra zero-fill launch is needed; every
    program of the ``b``-th matrix writes the same ``0`` before the elimination
    launch starts.
    """
    b = tle.program_id(0).to(tl.int64)
    blk = tl.program_id(1)
    e = blk * BLK + tl.arange(0, BLK)
    row = e // AW
    col = e % AW
    live = (row < N) & (col < N)
    idx = tl.where(live, row * N + col, 0)
    a = tl.load(A + b * N * N + idx)
    ident = (row == col).to(W.dtype.element_ty)
    tl.store(W + b * TOT + e, tl.where(live, a, ident))
    tl.store(X + b * TOT + e, ident)
    tl.store(INFO + b, 0)


@libentry()
@triton.jit
def _inv_ex_gj_kernel(
    W0,
    W1,
    X0,
    X1,
    INFO,
    N,
    AW: tl.constexpr,
    ROWS: tl.constexpr,
    TOT: tl.constexpr,
    BLK: tl.constexpr,
):
    """One-pass Gauss-Jordan on ``[W | X]`` with the singularity trace.

    Same elimination as ``inverse._inv_gj_kernel`` (single launch, one matrix
    per program, ``W``/``X`` pairs ping-ponged because a single-buffer in-kernel
    loop is silently reordered by this compiler), plus a register-resident
    ``singular`` scalar: the first step whose pivot is exactly zero is recorded
    and stored to ``INFO`` after the loop.  The last written buffer pair is the
    ``(n - 1)`` parity one; the caller picks it.
    """
    b = tl.program_id(0).to(tl.int64)
    base = b * TOT
    ridx = tl.arange(0, ROWS)
    live_rows = ridx < N
    nchunk = TOT // BLK
    singular = 0
    for k in range(0, N):
        is_even = (k % 2) == 0
        src_w = tl.where(is_even, W0, W1)
        dst_w = tl.where(is_even, W1, W0)
        src_x = tl.where(is_even, X0, X1)
        dst_x = tl.where(is_even, X1, X0)

        # Pivot search on the load-rooted 1-D column (never tl.argmax).
        col_k_sp = tl.load(src_w + base + ridx * AW + k)
        cand = tl.where(live_rows & (ridx >= k), tl.abs(col_k_sp), -1.0)
        best = tl.max(cand, axis=0)
        prow = tl.min(tl.where(cand == best, ridx, ROWS - 1), axis=0)
        akk = tl.sum(tl.where(ridx == k, col_k_sp, 0.0), axis=0)
        apk = tl.sum(tl.where(ridx == prow, col_k_sp, 0.0), axis=0)
        safe = tl.where(apk == 0.0, 1.0, apk)
        # LAPACK-order info: keep the first exactly-zero pivot.
        singular = tl.where((apk == 0.0) & (singular == 0), k + 1, singular)

        for c in range(0, nchunk):
            e = c * BLK + tl.arange(0, BLK)
            row = e // AW
            col = e % AW
            w = tl.load(src_w + base + e)
            x = tl.load(src_x + base + e)
            row_k_w = tl.load(src_w + base + k * AW + col)
            row_p_w = tl.load(src_w + base + prow * AW + col)
            row_k_x = tl.load(src_x + base + k * AW + col)
            row_p_x = tl.load(src_x + base + prow * AW + col)
            col_k = tl.load(src_w + base + e - col + k)
            swapped_w = tl.where(row == prow, row_k_w, w)
            swapped_x = tl.where(row == prow, row_k_x, x)
            lcol = tl.where(row == prow, akk, col_k)
            mult = tl.where(row == k, 0.0, lcol / safe)
            new_w = swapped_w - mult * row_p_w
            new_x = swapped_x - mult * row_p_x
            new_w = tl.where(row == k, row_p_w / safe, new_w)
            new_x = tl.where(row == k, row_p_x / safe, new_x)
            new_w = tl.where(col == k, tl.where(row == k, 1.0, 0.0), new_w)
            tl.store(dst_w + base + e, new_w)
            tl.store(dst_x + base + e, new_x)

    tl.store(INFO + b, singular.to(tl.int32))


@libentry()
@triton.jit
def _inv_ex_unpack_kernel(
    X, OUT, N, AW: tl.constexpr, TOT: tl.constexpr, BLK: tl.constexpr
):
    """Copy the leading ``n x n`` block of ``X`` back to the output.

    ``BLK`` need not divide ``n * n``; both the source and the destination index
    of the tail lanes are clamped to the last element, so the redundant writes
    are idempotent and no masked store is required.
    """
    b = tle.program_id(0).to(tl.int64)
    blk = tl.program_id(1)
    total = N * N
    e = blk * BLK + tl.arange(0, BLK)
    src = tl.where(e < total, e, total - 1)
    row = src // N
    col = src % N
    val = tl.load(X + b * TOT + row * AW + col)
    tl.store(OUT + b * total + src, val)


@libentry()
@triton.jit
def _inv_ex_pack_kernel(
    A, W, INFO, N, AW: tl.constexpr, TOT: tl.constexpr, BLK: tl.constexpr
):
    """Scatter ``A`` into a padded working tile for the info-only pass.

    Padding lanes are zeroed; they are never selected as pivots (``ridx < N``)
    and never feed a live row, so they cannot change the pivot sequence.  The
    per-matrix ``INFO`` slot is reset here (all programs of the ``b``-th matrix
    write the same ``0``) and this launch always precedes the elimination.
    """
    b = tle.program_id(0).to(tl.int64)
    blk = tl.program_id(1)
    e = blk * BLK + tl.arange(0, BLK)
    row = e // AW
    col = e % AW
    live = (row < N) & (col < N)
    idx = tl.where(live, row * N + col, 0)
    a = tl.load(A + b * N * N + idx)
    zero = tl.zeros([BLK], dtype=W.dtype.element_ty)
    tl.store(W + b * TOT + e, tl.where(live, a, zero))
    tl.store(INFO + b, 0)


@libentry()
@triton.jit
def _inv_ex_info_gj_kernel(
    W0,
    W1,
    INFO,
    N,
    AW: tl.constexpr,
    ROWS: tl.constexpr,
    TOT: tl.constexpr,
    BLK: tl.constexpr,
):
    """Whole pivoted elimination on ``A`` alone, one matrix per program.

    Mirrors ``inverse._inv_gj_kernel`` but updates only the working matrix and
    records the first exactly-zero pivot into ``INFO``.  Used by the info-only
    pass when the inverse itself is delegated to ``inverse`` (large ``n``), in
    the same batch/tile regime the pattern is exact for.
    """
    b = tl.program_id(0).to(tl.int64)
    base = b * TOT
    ridx = tl.arange(0, ROWS)
    live_rows = ridx < N
    nchunk = TOT // BLK
    singular = 0
    for k in range(0, N):
        is_even = (k % 2) == 0
        src = tl.where(is_even, W0, W1)
        dst = tl.where(is_even, W1, W0)

        col_k_sp = tl.load(src + base + ridx * AW + k)
        cand = tl.where(live_rows & (ridx >= k), tl.abs(col_k_sp), -1.0)
        best = tl.max(cand, axis=0)
        prow = tl.min(tl.where(cand == best, ridx, ROWS - 1), axis=0)
        akk = tl.sum(tl.where(ridx == k, col_k_sp, 0.0), axis=0)
        apk = tl.sum(tl.where(ridx == prow, col_k_sp, 0.0), axis=0)
        safe = tl.where(apk == 0.0, 1.0, apk)
        singular = tl.where((apk == 0.0) & (singular == 0), k + 1, singular)

        for c in range(0, nchunk):
            e = c * BLK + tl.arange(0, BLK)
            row = e // AW
            col = e % AW
            w = tl.load(src + base + e)
            row_k = tl.load(src + base + k * AW + col)
            row_p = tl.load(src + base + prow * AW + col)
            col_k = tl.load(src + base + e - col + k)
            swapped = tl.where(row == prow, row_k, w)
            lcol = tl.where(row == prow, akk, col_k)
            mult = tl.where(row == k, 0.0, lcol / safe)
            new = swapped - mult * row_p
            new = tl.where(row == k, row_p / safe, new)
            new = tl.where(col == k, tl.where(row == k, 1.0, 0.0), new)
            tl.store(dst + base + e, new)

    tl.store(INFO + b, singular.to(tl.int32))


@libentry()
@triton.jit
def _inv_ex_info_step_kernel(
    W,
    INFO,
    N,
    K,
    AW: tl.constexpr,
    ROWS: tl.constexpr,
    TOT: tl.constexpr,
):
    """One complete pivoted elimination step on ``A`` alone, in place.

    Mirrors ``inverse._inv_step_kernel`` (fused pivot / swap / normalise /
    rank-1 update) with the update-only-if-still-zero ``INFO`` record instead
    of the inverse accumulator.  Used outside the small-batch in-kernel regime.
    """
    b = tl.program_id(0).to(tl.int64)
    base = b * TOT
    e = tl.arange(0, TOT)
    row = e // AW
    col = e % AW
    w = tl.load(W + base + e)

    ridx = tl.arange(0, ROWS)
    live_rows = ridx < N
    col_k = tl.load(W + base + ridx * AW + K)
    cand = tl.where(live_rows & (ridx >= K), tl.abs(col_k), -1.0)
    best = tl.max(cand, axis=0)
    prow = tl.min(tl.where(cand == best, ridx, ROWS - 1), axis=0)
    akk = tl.sum(tl.where(ridx == K, col_k, 0.0), axis=0)
    apk = tl.sum(tl.where(ridx == prow, col_k, 0.0), axis=0)
    safe = tl.where(apk == 0.0, 1.0, apk)

    row_k = tl.load(W + base + K * AW + col)
    row_p = tl.load(W + base + prow * AW + col)
    col_k_elems = tl.load(W + base + row * AW + K)
    swapped = tl.where(row == prow, row_k, w)
    lcol = tl.where(row == prow, akk, col_k_elems)
    mult = tl.where(row == K, 0.0, lcol / safe)

    new = swapped - mult * row_p
    new = tl.where(row == K, row_p / safe, new)
    new = tl.where(col == K, tl.where(row == K, 1.0, 0.0), new)
    tl.store(W + base + e, new)

    prev = tl.load(INFO + b)
    tl.store(INFO + b, tl.where((apk == 0.0) & (prev == 0), K + 1, prev))


@triton.jit
def _inv_ex_flag_kernel(INFO, FLAG, B, BLOCK_B: tl.constexpr):
    """Reduce the per-matrix info values to a single 0/1 flag on device."""
    offs = tl.arange(0, BLOCK_B)
    acc = 0
    for base in range(0, B, BLOCK_B):
        vals = tl.load(INFO + base + offs, mask=(base + offs) < B, other=0)
        acc = tl.maximum(acc, tl.max(tl.abs(vals), axis=0))
    tl.store(FLAG, (acc != 0).to(tl.int32))


def _run_one_pass(A_flat, out_flat, batch, n, info):
    """Inverse + info in a single elimination (small batch, ``n <= 32``)."""
    rows, aw, tot = _plan_pad(n)
    device = A_flat.device
    dtype = A_flat.dtype
    w0 = torch.empty(batch * tot, dtype=dtype, device=device)
    w1 = torch.empty(batch * tot, dtype=dtype, device=device)
    x0 = torch.empty(batch * tot, dtype=dtype, device=device)
    x1 = torch.empty(batch * tot, dtype=dtype, device=device)
    unpack_blk = min(tot, triton.next_power_of_2(max(n * n, 1)))
    with torch_device_fn.device(device):
        _inv_ex_pack_gj_kernel[(batch, 1)](
            A_flat, w0, x0, info, n, AW=aw, TOT=tot, BLK=tot, num_warps=1
        )
        _inv_ex_gj_kernel[(batch,)](
            w0, w1, x0, x1, info, n, AW=aw, ROWS=rows, TOT=tot, BLK=tot,
            num_warps=2,
        )
        # Each iteration writes the pair it did not read, so the last written
        # pair has the (n - 1) parity (inverse._run_gj).
        final_x = x1 if (n % 2 == 1) else x0
        _inv_ex_unpack_kernel[(batch, triton.cdiv(n * n, unpack_blk))](
            final_x, out_flat, n, AW=aw, TOT=tot, BLK=unpack_blk, num_warps=1
        )


def _compute_info(A_flat, batch, n, info):
    """Record the first exactly-zero pivot of each matrix into ``info``.

    Info-only pass used when the inverse comes from ``inverse`` (two passes).
    """
    device = A_flat.device
    dtype = A_flat.dtype
    if _use_one_pass(batch, n):
        rows, aw, tot = _plan_pad(n)
        w0 = torch.empty(batch * tot, dtype=dtype, device=device)
        w1 = torch.empty(batch * tot, dtype=dtype, device=device)
        with torch_device_fn.device(device):
            _inv_ex_pack_kernel[(batch, 1)](
                A_flat, w0, info, n, AW=aw, TOT=tot, BLK=tot, num_warps=1
            )
            _inv_ex_info_gj_kernel[(batch,)](
                w0, w1, info, n, AW=aw, ROWS=rows, TOT=tot, BLK=tot,
                num_warps=2,
            )
    else:
        rows, aw, tot, blk = _plan(n)
        w = torch.empty(batch * tot, dtype=dtype, device=device)
        with torch_device_fn.device(device):
            _inv_ex_pack_kernel[(batch, tot // blk)](
                A_flat, w, info, n, AW=aw, TOT=tot, BLK=blk, num_warps=1
            )
            for k in range(n):
                _inv_ex_info_step_kernel[(batch,)](
                    w, info, n, k, AW=aw, ROWS=rows, TOT=tot, num_warps=1
                )


def linalg_inv_ex(A, *, check_errors=False):
    """Compute ``(inverse, info)`` of a square matrix, like ATen."""
    logger.debug("GEMS_KUNLUNXIN LINALG_INV_EX")

    assert A.dim() >= 2, "linalg_inv_ex: input must be at least 2D"
    n = A.shape[-1]
    assert n == A.shape[-2], "linalg_inv_ex: input must be a square matrix"
    if A.dtype in (torch.complex64, torch.complex128):
        # The XPU3 stack has no complex linear-algebra path at all; fail
        # explicitly instead of silently producing a real-valued result.
        raise NotImplementedError(
            "Kunlunxin linalg_inv_ex does not support complex dtypes: the XPU3 "
            "stack provides no complex linear-algebra primitive"
        )

    batch_shape = A.shape[:-2]
    if A.numel() == 0 or n == 0:
        return LinalgInvExResult(
            torch.empty(A.shape, dtype=A.dtype, device=A.device),
            torch.empty(batch_shape, dtype=torch.int32, device=A.device),
        )

    batch = A.numel() // (n * n)
    info = torch.empty(batch, dtype=torch.int32, device=A.device)

    if _use_one_pass(batch, n):
        out = torch.empty(A.shape, dtype=A.dtype, device=A.device)
        _run_one_pass(
            A.contiguous().reshape(batch, n, n),
            out.reshape(batch, n, n),
            batch,
            n,
            info,
        )
    else:
        # Same backend-local Triton Gauss-Jordan, via the shared public entry.
        out = _inverse(A)
        _compute_info(
            A.contiguous().reshape(batch, n, n), batch, n, info
        )

    if check_errors:
        flag = torch.empty(1, dtype=torch.int32, device=A.device)
        bb = max(1, triton.next_power_of_2(min(batch, 1024)))
        _inv_ex_flag_kernel[(1,)](info, flag, batch, BLOCK_B=bb, num_warps=1)
        if flag.item() != 0:
            raise torch.linalg.LinAlgError(
                "torch.linalg.inv_ex: The diagonal element of the LU "
                "decomposition is zero."
            )

    return LinalgInvExResult(out, info.view(batch_shape))

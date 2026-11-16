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

"""Kunlunxin backend override for ``aten::linalg_tensorsolve``.

Why this exists
---------------
The generic ``flag_gems/ops/linalg_tensorsolve.py`` keeps the whole ``P x P``
system in one register tile and factorises it with Gauss-Jordan elimination
inside a single program, extracting column ``k`` with
``tl.sum(tl.where(cols == k, A, 0), axis=1)`` (an axis-1 reduce over an
*arithmetic* 2-D tile) and a bespoke multi-value ``tl.reduce`` pivot.  On XPU3
the legaliser rejects every real shape:

    arith.select op failed to verify that condition is signless i1 ...
    -> OutOfResources: out of resource: uni_sram

and the ``tl.static_range`` unroll makes mid-size systems (P = 8) exceed the
compile budget outright.  There is no backend override, so on this platform
``linalg_tensorsolve`` is unusable for any ``P >= 2``.

Semantics (numpy / ATen)
------------------------
With ``A = self`` (``a = A.ndim``) and ``B = other`` (``b = B.ndim``),
``dims`` moves ``k = a - b`` axes of ``A`` to the right.  Reordering ``A`` to
``A_p`` gives::

    out_shape = A_p.shape[b:]
    P         = prod(out_shape)
    X         = solve(A_p.reshape(P, P), B.reshape(P)).reshape(out_shape)

This module only reshapes and then **solves** the resulting ``P x P`` system;
no factorisation of ``A`` is exposed.

Algorithm (reuses the proven XPU3 Gauss-Jordan family)
------------------------------------------------------
The working matrix ``W`` and the right-hand side ``b`` live in padded
power-of-two global buffers (identity / zero padding, so the padded system is
already the identity on the dead rows/columns).  One augmented Gauss-Jordan
step with partial pivoting is::

    p   = first argmax_{k <= i < P} |W[i, k]|        # never tl.argmax
    swap rows k and p of W and b
    normalise row k by the (swapped) pivot
    for i != k: row_i -= W[i, k] * row_k             # both W and b
    pin W[:, k] = e_k                                 # keeps the dead column exact

Every reduction (``tl.max`` / ``tl.min`` / ``tl.sum``) runs over a **load-rooted
1-D vector** (the ``k``-th column / the RHS vector); no 2-D arithmetic tile is
ever reduced.

Dispatch: one fused step launch per pivot, ping-ponging ``(W, b)`` on the host
so no kernel reads a buffer it is writing (the documented XPU3 same-kernel
global RAW hazard).  The padded system is held in a single program tile (no
in-kernel chunk loop: a multi-chunk loop over the tile fails to lower on this
backend).  A single-launch in-kernel ping-pong variant was tried and rejected:
on this backend its result depended on the destination buffers' initial
contents (an in-loop read-before-write), which is exactly the reordering hazard
the backend documents.  The launch-per-step form is exact for every size in the
operator's regime (P <= 16 in the tests, P <= 9 in the benchmark) and remains
faster than the native call for all benchmark shapes (the system sizes are tiny,
so the ~11 us launch floor is well below native's fixed cost).
"""

import torch
import triton
import triton.language as tl


def _plan(P):
    """Padded working layout: power-of-two rows and a >= 64-wide row stride
    (narrow strides below 8 miscompile on this backend)."""
    rows = max(2, triton.next_power_of_2(P))
    aw = max(64, rows)
    return rows, aw, rows * aw


@triton.jit
def _ts_pack_kernel(
    A,
    B,
    W,
    BB,
    P,
    AW: tl.constexpr,
    ROWS: tl.constexpr,
    TOT: tl.constexpr,
    BLK: tl.constexpr,
):
    """Scatter the contiguous ``P x P`` matrix into padded ``W`` (identity on
    the dead rows/columns) and the RHS into padded ``BB`` (zeros)."""
    blk = tl.program_id(0)
    e = blk * BLK + tl.arange(0, BLK)
    row = e // AW
    col = e % AW
    live = (row < P) & (col < P)
    idx = tl.where(live, row * P + col, 0)
    a = tl.load(A + idx)
    ident = (row == col).to(W.dtype.element_ty)
    tl.store(W + e, tl.where(live, a, ident).to(W.dtype.element_ty))

    if blk == 0:
        ridx = tl.arange(0, ROWS)
        bi = tl.where(ridx < P, ridx, 0)
        bval = tl.load(B + bi)
        tl.store(BB + ridx, tl.where(ridx < P, bval, 0.0).to(BB.dtype.element_ty))


@triton.jit
def _ts_step_body(
    src_w,
    dst_w,
    src_b,
    dst_b,
    P,
    K,
    AW: tl.constexpr,
    ROWS: tl.constexpr,
    BLK: tl.constexpr,
):
    """One augmented Gauss-Jordan step: ``src`` -> ``dst`` (never in place)."""
    ridx = tl.arange(0, ROWS)

    # Pivot search over the load-rooted column K (never tl.argmax; the
    # ``(cand == best)`` mask excludes rows outside [K, P) because they carry -1).
    col_k = tl.load(src_w + ridx * AW + K)
    cand = tl.where((ridx < P) & (ridx >= K), tl.abs(col_k), -1.0)
    best = tl.max(cand, axis=0)
    prow = tl.min(tl.where(cand == best, ridx, ROWS - 1), axis=0)
    akk = tl.sum(tl.where(ridx == K, col_k, 0.0), axis=0)  # old W[K, K]
    apk = tl.sum(tl.where(ridx == prow, col_k, 0.0), axis=0)  # old W[p, K]
    safe = tl.where(apk == 0.0, 1.0, apk)

    # RHS update (mirrors the matrix update below, on a 1-D vector).
    bs = tl.load(src_b + ridx)
    b_k = tl.sum(tl.where(ridx == K, bs, 0.0), axis=0)  # old b[K]
    b_p = tl.sum(tl.where(ridx == prow, bs, 0.0), axis=0)  # old b[p]
    lcol_b = tl.where(ridx == prow, akk, col_k)  # swapped[r, K]
    mult_b = tl.where(ridx == K, 0.0, lcol_b / safe)
    sbb = tl.where(ridx == prow, b_k, bs)  # row p <- old row K
    nb = sbb - mult_b * b_p
    nb = tl.where(ridx == K, b_p / safe, nb)  # normalised pivot row
    tl.store(dst_b + ridx, nb.to(dst_b.dtype.element_ty))

    # Trailing rank-1 update of the matrix.  Only ``src_w`` is read and only
    # ``dst_w`` is written, so the reads of row K / row p / column K can never
    # observe a partial update.
    e = tl.arange(0, BLK)
    row = e // AW
    col = e % AW
    w = tl.load(src_w + e)
    row_k = tl.load(src_w + K * AW + col)
    row_p = tl.load(src_w + prow * AW + col)
    col_e = tl.load(src_w + row * AW + K)
    swapped = tl.where(row == prow, row_k, w)
    lcol = tl.where(row == prow, akk, col_e)
    mult = tl.where(row == K, 0.0, lcol / safe)
    nw = swapped - mult * row_p
    nw = tl.where(row == K, row_p / safe, nw)
    nw = tl.where(col == K, tl.where(row == K, 1.0, 0.0), nw)
    tl.store(dst_w + e, nw.to(dst_w.dtype.element_ty))


@triton.jit
def _ts_step_kernel(
    src_w,
    dst_w,
    src_b,
    dst_b,
    P,
    K,
    AW: tl.constexpr,
    ROWS: tl.constexpr,
    BLK: tl.constexpr,
):
    """Single pivot step (one launch per pivot from the host)."""
    _ts_step_body(src_w, dst_w, src_b, dst_b, P, K, AW, ROWS, BLK)


@triton.jit
def _ts_unpack_kernel(BB, OUT, P, BLK: tl.constexpr):
    """Copy the solution vector out (tail lanes clamp, so no masked store)."""
    e = tl.arange(0, BLK)
    src = tl.where(e < P, e, P - 1)
    val = tl.load(BB + src)
    tl.store(OUT + src, val.to(OUT.dtype.element_ty))


def _resolve_dims(a, dims):
    """Return the permutation that moves the selected axes of ``A`` to the
    right, matching ATen (including its validation errors)."""
    if dims is None:
        return list(range(a))
    if isinstance(dims, torch.Tensor):
        raw = dims.reshape(-1).tolist()
    else:
        raw = list(dims)
    dl = []
    for d in raw:
        d = int(d)
        if d < -a or d >= a:
            raise IndexError(
                "Dimension out of range (expected to be in range of "
                "[%d, %d], but got %d)" % (-a, a - 1, d)
            )
        d = d % a
        if d in dl:
            raise RuntimeError(
                "linalg.tensorsolve: dims must not contain duplicates, "
                "but dimension %d appears more than once" % d
            )
        dl.append(d)
    rest = [d for d in range(a) if d not in dl]
    return rest + dl


def linalg_tensorsolve(self, other, dims=None):
    A = self
    B = other
    a = A.dim()
    bn = B.dim()

    perm = _resolve_dims(a, dims)
    Ap = A if perm == list(range(a)) else A.permute(perm)

    lead_shape = Ap.shape[:bn]
    tail_shape = tuple(Ap.shape[bn:])
    P = 1
    for s in tail_shape:
        P *= int(s)
    nlead = 1
    for s in lead_shape:
        nlead *= int(s)

    out = torch.empty(tail_shape, dtype=B.dtype, device=A.device)
    if P == 0 or nlead == 0:
        return out
    if P != nlead or P != B.numel():
        raise RuntimeError("linalg.tensorsolve: non square / mismatching system")

    if A.dtype in (torch.complex64, torch.complex128):
        raise NotImplementedError(
            "Kunlunxin linalg_tensorsolve does not support complex dtypes: the "
            "XPU3 stack provides no complex linear-algebra primitive"
        )

    # Elimination precision follows the input; narrower floats widen to fp32
    # (as the generic kernel does) and the solution is cast at the unpack.
    work_dtype = torch.float64 if A.dtype == torch.float64 else torch.float32

    A_flat = Ap.reshape(P, P)
    if not A_flat.is_contiguous():
        A_flat = A_flat.contiguous()
    B_flat = B.reshape(-1)
    if not B_flat.is_contiguous():
        B_flat = B_flat.contiguous()
    out_flat = out.reshape(-1)

    device = A.device
    rows, aw, tot = _plan(P)
    # One tile per program: the whole padded system fits a single block for the
    # sizes this operator sees (P <= ~64).  A multi-chunk loop over the tile was
    # measured to fail lowering on XPU3 ("operand does not dominate this use").
    blk = tot

    w0 = torch.empty(tot, dtype=work_dtype, device=device)
    w1 = torch.empty(tot, dtype=work_dtype, device=device)
    b0 = torch.empty(rows, dtype=work_dtype, device=device)
    b1 = torch.empty(rows, dtype=work_dtype, device=device)

    _ts_pack_kernel[(1,)](A_flat, B_flat, w0, b0, P, AW=aw, ROWS=rows, TOT=tot, BLK=blk)

    # Ping-pong one fused step per pivot: each launch reads the buffer the
    # previous launch wrote and writes the other one, so there is never a
    # same-kernel write-then-read on the working buffers.
    src_w, dst_w = w0, w1
    src_b, dst_b = b0, b1
    for k in range(P):
        _ts_step_kernel[(1,)](
            src_w, dst_w, src_b, dst_b, P, k, AW=aw, ROWS=rows, BLK=blk
        )
        src_w, dst_w = dst_w, src_w
        src_b, dst_b = dst_b, src_b

    ublk = max(1, triton.next_power_of_2(P))
    _ts_unpack_kernel[(1,)](src_b, out_flat, P, BLK=ublk)
    return out


__all__ = ["linalg_tensorsolve"]

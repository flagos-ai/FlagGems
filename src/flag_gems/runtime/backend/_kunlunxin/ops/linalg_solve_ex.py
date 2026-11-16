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

"""Kunlunxin backend override for ``linalg_solve_ex`` (XPU3).

``aten::linalg_solve_ex`` solves ``A X = B`` for the public operator and
returns ``(result, info)`` (the private ``_linalg_solve_ex`` adds ``LU`` and
``pivots`` and is a separate entry point).  The generic
``flag_gems/ops/linalg_solve_ex.py`` is one fused Triton kernel per batch
element that carries a factor row across the inner row loop and uses a scalar
``if`` for the pivot search; on this backend that miscompiles and every real
case is numerically wrong (see the ``_linalg_solve_ex`` dossier and the wiki
pattern ``xpu3-linalg-solve-ex-generic-broken``).

Rather than re-derive the LU factorization, this override augments the proven
backend-local Gauss-Jordan from ``inverse`` / ``linalg_inv_ex``.  Eliminating
the left block of ``[A | B]`` turns it into ``[I | A^-1 B]``, so the solution
falls out of the same elimination that would compute ``A^-1`` -- no triangular
substitution and no ``LU``/``pivots`` bookkeeping is needed for the public
profile.  Every host-visible kernel here is pure Triton; there is no
CPU/ATen/native/composite fallback.

Two dispatch regimes, both emitting ``info`` from the same elimination:

* **one pass** (``batch <= 4`` and ``n <= 32``): the whole ``for k`` chain runs
  in one kernel on padded ``[W | X]`` buffers ping-ponged across iterations.
  Reuses ``linalg_inv_ex._inv_ex_gj_kernel`` verbatim (already swept exact at
  ``batch <= 4``): it is generic in the initial ``X`` tile, so seeding ``X``
  with ``B`` yields ``A^-1 B`` instead of ``A^-1``.  Covers every benchmark
  shape with ``batch <= 4``.
* **per-step** otherwise: ``_solve_step_kernel`` advances the same ``[W | X]``
  state one pivot per launch (the ``inverse._inv_step_kernel`` regime) and
  records the singularity trace in place.  Works for any batch.

Reductions run only on load-rooted 1-D column vectors; ``tl.argmax``,
``tl.trans`` and single-buffer in-kernel loops are avoided (all broken on
XPU3).
"""

import logging
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

from .inverse import _plan as _inv_plan
from .linalg_inv_ex import _inv_ex_flag_kernel, _inv_ex_gj_kernel, _plan_pad, _use_one_pass

logger = logging.getLogger(__name__)

LinalgSolveExResult = namedtuple("LinalgSolveExResult", ["result", "info"])

# float32/float64 both lower through the same float pointer kernels.
_TRITON_DTYPES = (torch.float32, torch.float64)


@libentry()
@triton.jit
def _solve_pack_kernel(
    A,
    B,
    W,
    X,
    INFO,
    N,
    K,
    AW: tl.constexpr,
    TOT: tl.constexpr,
    BLK: tl.constexpr,
):
    """Scatter ``A`` into padded ``W`` and seed ``X`` with ``B``.

    ``W`` gets ``A`` in the leading ``n x n`` block and the identity on the
    padding lanes, so the padded system is block diagonal and never feeds the
    live block.  ``X`` is the identity except for the leading ``n x K`` block,
    which holds ``B``: the same row operations that reduce ``W`` to ``I`` turn
    that block into ``A^-1 B``.  ``INFO`` is reset here so no extra zero-fill
    launch is needed.
    """
    b = tle.program_id(0).to(tl.int64)
    blk = tl.program_id(1)
    e = blk * BLK + tl.arange(0, BLK)
    row = e // AW
    col = e % AW
    ident = (row == col).to(W.dtype.element_ty)

    live = (row < N) & (col < N)
    aidx = tl.where(live, row * N + col, 0)
    b_live = (row < N) & (col < K)
    bidx = tl.where(b_live, row * K + col, 0)
    # Load both inputs before any store: this backend reorders a same-kernel
    # store-then-load and fails the dominance check ("operand does not dominate
    # this use"), so keep every read ahead of every write.
    a = tl.load(A + b * N * N + aidx)
    bv = tl.load(B + b * N * K + bidx)

    tl.store(W + b * TOT + e, tl.where(live, a, ident))
    tl.store(X + b * TOT + e, tl.where(b_live, bv, ident))
    tl.store(INFO + b, 0)


@libentry()
@triton.jit
def _solve_step_kernel(
    W, X, INFO, N, K, AW: tl.constexpr, ROWS: tl.constexpr, TOT: tl.constexpr
):
    """One complete Gauss-Jordan step of ``[W | X]`` in place, plus ``info``.

    Mirrors ``inverse._inv_step_kernel`` (pivot search on the load-rooted
    column, fused swap/normalise/rank-1 update of both tiles) and folds the
    singularity trace in: the first step whose pivoted ``|pivot|`` is exactly
    zero is recorded as ``K + 1`` (LAPACK order).  Exactly one program owns the
    matrix, so the in-place store cannot race.
    """
    b = tl.program_id(0).to(tl.int64)
    base = b * TOT
    e = tl.arange(0, TOT)
    row = e // AW
    col = e % AW
    w = tl.load(W + base + e)
    x = tl.load(X + base + e)

    ridx = tl.arange(0, ROWS)
    live_rows = ridx < N
    col_k = tl.load(W + base + ridx * AW + K)
    cand = tl.where(live_rows & (ridx >= K), tl.abs(col_k), -1.0)
    best = tl.max(cand, axis=0)
    prow = tl.min(tl.where(cand == best, ridx, ROWS - 1), axis=0)
    akk = tl.sum(tl.where(ridx == K, col_k, 0.0), axis=0)
    apk = tl.sum(tl.where(ridx == prow, col_k, 0.0), axis=0)
    safe = tl.where(apk == 0.0, 1.0, apk)

    row_k_w = tl.load(W + base + K * AW + col)
    row_p_w = tl.load(W + base + prow * AW + col)
    row_k_x = tl.load(X + base + K * AW + col)
    row_p_x = tl.load(X + base + prow * AW + col)
    col_k_elems = tl.load(W + base + row * AW + K)
    swapped_w = tl.where(row == prow, row_k_w, w)
    swapped_x = tl.where(row == prow, row_k_x, x)
    lcol = tl.where(row == prow, akk, col_k_elems)
    mult = tl.where(row == K, 0.0, lcol / safe)

    new_w = swapped_w - mult * row_p_w
    new_x = swapped_x - mult * row_p_x
    new_w = tl.where(row == K, row_p_w / safe, new_w)
    new_x = tl.where(row == K, row_p_x / safe, new_x)
    new_w = tl.where(col == K, tl.where(row == K, 1.0, 0.0), new_w)
    tl.store(W + base + e, new_w)
    tl.store(X + base + e, new_x)

    prev = tl.load(INFO + b)
    tl.store(INFO + b, tl.where((apk == 0.0) & (prev == 0), K + 1, prev))


@libentry()
@triton.jit
def _solve_unpack_kernel(
    X, OUT, N, K, AW: tl.constexpr, TOT: tl.constexpr, BLK: tl.constexpr
):
    """Copy the leading ``n x K`` block of ``X`` back to the output.

    ``BLK`` need not divide ``n * K``; both source and destination indices of
    the tail lanes are clamped to the last element, so the redundant writes are
    idempotent and no masked store is required.
    """
    b = tle.program_id(0).to(tl.int64)
    blk = tl.program_id(1)
    total = N * K
    e = blk * BLK + tl.arange(0, BLK)
    src = tl.where(e < total, e, total - 1)
    row = src // K
    col = src % K
    val = tl.load(X + b * TOT + row * AW + col)
    tl.store(OUT + b * total + src, val)


def _run_one_pass(A_flat, B_flat, out_flat, batch, n, k, info):
    """Solve ``A X = B`` and record ``info`` in a single elimination."""
    rows, aw, tot = _plan_pad(n)
    device = A_flat.device
    dtype = A_flat.dtype
    w0 = torch.empty(batch * tot, dtype=dtype, device=device)
    w1 = torch.empty(batch * tot, dtype=dtype, device=device)
    x0 = torch.empty(batch * tot, dtype=dtype, device=device)
    x1 = torch.empty(batch * tot, dtype=dtype, device=device)
    unpack_blk = min(tot, triton.next_power_of_2(max(n * k, 1)))
    with torch_device_fn.device(device):
        _solve_pack_kernel[(batch, 1)](
            A_flat, B_flat, w0, x0, info, n, k, AW=aw, TOT=tot, BLK=tot, num_warps=1
        )
        _inv_ex_gj_kernel[(batch,)](
            w0, w1, x0, x1, info, n, AW=aw, ROWS=rows, TOT=tot, BLK=tot,
            num_warps=2,
        )
        # Each iteration writes the pair it did not read, so the last written
        # pair has the (n - 1) parity (inverse._run_gj).
        final_x = x1 if (n % 2 == 1) else x0
        _solve_unpack_kernel[(batch, triton.cdiv(n * k, unpack_blk))](
            final_x, out_flat, n, k, AW=aw, TOT=tot, BLK=unpack_blk, num_warps=1
        )


def _run_per_step(A_flat, B_flat, out_flat, batch, n, k, info):
    """Solve ``A X = B`` one pivot per launch; any batch size."""
    rows, aw, tot, blk, nchunk = _inv_plan(n)
    device = A_flat.device
    dtype = A_flat.dtype
    w = torch.empty(batch * tot, dtype=dtype, device=device)
    x = torch.empty(batch * tot, dtype=dtype, device=device)
    unpack_blk = min(blk, triton.next_power_of_2(max(n * k, 1)))
    with torch_device_fn.device(device):
        _solve_pack_kernel[(batch, nchunk)](
            A_flat, B_flat, w, x, info, n, k, AW=aw, TOT=tot, BLK=blk, num_warps=1
        )
        for kk in range(n):
            _solve_step_kernel[(batch,)](
                w, x, info, n, kk, AW=aw, ROWS=rows, TOT=tot, num_warps=1
            )
        _solve_unpack_kernel[(batch, triton.cdiv(n * k, unpack_blk))](
            x, out_flat, n, k, AW=aw, TOT=tot, BLK=unpack_blk, num_warps=1
        )


def linalg_solve_ex(A, B, *, left=True, check_errors=False):
    """Solve the linear system ``A X = B``, returning ``(result, info)``.

    Mirrors the generic entry point's contract: ``left=True`` only (the public
    ``torch.linalg.solve_ex`` with ``left=False`` is unsupported here as well),
    ``A`` is ``(*, n, n)`` and ``B`` is ``(*, n, k)`` with matching batch.
    ``info`` is ``0`` on success or the ``k + 1`` of the first zero pivot.
    """
    logger.debug("GEMS_KUNLUNXIN LINALG_SOLVE_EX")

    if not left:
        raise NotImplementedError("right=True (XA = B) is not yet supported")
    assert A.dtype in _TRITON_DTYPES, (
        f"linalg_solve_ex requires float32/float64, got {A.dtype}"
    )
    if A.ndim < 2 or B.ndim < 2:
        raise ValueError("A and B must be at least 2D")
    if A.shape[-1] != A.shape[-2]:
        raise ValueError("A must be a square matrix")
    n = A.shape[-1]
    if B.shape[-2] != n:
        raise ValueError("B must have compatible dimensions with A")

    batch_shape = A.shape[:-2]
    if A.numel() == 0 or B.numel() == 0:
        info = torch.zeros(batch_shape, dtype=torch.int32, device=A.device)
        return B.clone(), info

    batch = 1
    for dim in batch_shape:
        batch *= dim
    k = B.shape[-1]

    A_flat = A.contiguous().reshape(batch, n, n)
    B_flat = B.contiguous().reshape(batch, n, k)
    out = torch.empty(B.shape, dtype=B.dtype, device=B.device)
    out_flat = out.reshape(batch, n, k)
    info = torch.empty(batch, dtype=torch.int32, device=A.device)

    if _use_one_pass(batch, n):
        _run_one_pass(A_flat, B_flat, out_flat, batch, n, k, info)
    else:
        _run_per_step(A_flat, B_flat, out_flat, batch, n, k, info)

    if check_errors:
        flag = torch.empty(1, dtype=torch.int32, device=A.device)
        bb = max(1, triton.next_power_of_2(min(batch, 1024)))
        _inv_ex_flag_kernel[(1,)](info, flag, batch, BLOCK_B=bb, num_warps=1)
        if flag.item() != 0:
            raise torch.linalg.LinAlgError(
                "linalg.solve_ex: The diagonal element of the LU decomposition "
                "is zero."
            )

    return LinalgSolveExResult(out, info.view(batch_shape))

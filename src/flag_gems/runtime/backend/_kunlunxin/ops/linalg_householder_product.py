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

"""Kunlunxin/XPU implementation of ``torch.linalg.householder_product``.

Q = H(0) * H(1) * ... * H(K-1),  H(i) = I - tau[i] * v(i) * v(i)^T,
v(i) = [0 .. 0 1 A[i+1, i] .. A[M-1, i]]   (output of torch.geqrf).

Why the generic implementation does not compile on XPU
------------------------------------------------------
The generic ``flag_gems.ops.linalg_householder_product`` keeps a
[BLOCK_M, BLOCK_N] 2-D register tile and computes
``dots = tl.sum(v[:, None] * q_block, axis=0)``.  ``TritonXPULegalize``
rejects every 2D+ ``tt.reduce`` on axis 0 with ``shape[0] > 1``
("axis must not be 0 for 2D+ shapes", triton
``third_party/xpu/lib/Dialect/TritonXPU/Transforms/Legalize.cpp``) and the
suggested workaround (``tt.trans``) cannot be lowered on XPU at all.  On top
of that, the generic kernel's K-loop is an ``scf.for`` whose body stores the
Q tile to global and loads it back on the next iteration; a dynamic loop
that loads and stores the same buffer is miscompiled by the XPU backend
(verified in isolation: plain ``Q = Q*2+1`` outside a dynamic loop is exact,
inside one the results are garbage).

Implementation strategy
-----------------------
1. **Column-per-program one-shot kernel** (M <= _ONE_SHOT_M): each program
   owns one output column ``Q[:, col]`` as a 1-D register vector.  The only
   reduction ``dots = tl.sum(v * q)`` is a 1-D (scalar) reduction, the update
   ``q -= tau_k * dots * v`` is elementwise, and Q is written exactly once
   (a single masked store).  A and tau are only *read*; Q is never read back,
   so the store/load miscompilation cannot occur.
2. **3-kernel-per-k fallback** (M > _ONE_SHOT_M): the m dimension is split
   into NBLK blocks of BLOCK_M lanes (a single program cannot hold more than
   _ONE_SHOT_M lanes of a column).  For each k: kernel A computes per-block
   partial dot products (loads only), kernel B reduces them to the scalar
   ``dots``, kernel C updates and stores the Q block (load-then-store, no
   store-before-load within a program).  Q is initialised with ``torch.eye``
   on the host; K launches per batch element.

XPU-specific details
--------------------
- Loads use clamped row indices (``tl.minimum(rows, M - 1)``) instead of
  ``mask=``: a masked load still issues the memory access on XPU, so the
  address itself must be legal; the out-of-range contribution is zeroed by a
  ``tl.where`` (the ``v`` selector and the ``tl.where(msk, qb, 0.0)``).
- Stores keep an affine ``mask=`` (row < M): affine masked stores are handled
  correctly on XPU (only data-dependent/gather addresses silently drop the
  mask), and this keeps the non-power-of-two M tails in place.
- ``isCloseUnrollControl=True`` skips ``TritonXPUUnrollControl``, which
  tiles the K-loop body without re-typing the loop-carried register tensor
  (``arith.mulf``/``arith.cmpi`` verification failure) — same fix as
  ``searchsorted.py``.
- ``torch.geqrf`` on XPU returns ``h`` in column-major layout
  (``stride == (1, m)``); the launcher always materialises a contiguous
  row-major copy (``A.unsqueeze(0).contiguous()`` as the generic does), so
  ``stride_row == n`` and all kernel indexing is row-major.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

# One-shot path: M <= _ONE_SHOT_M keeps the whole Q column in registers
# (verified correct on XPU for BLOCK_M up to 8192; beyond that the kernel
# miscompiles).  Above it the 3-kernel-per-k fallback takes over.
_ONE_SHOT_M = 8192
_BLOCKED_BLOCK_M = 512


@triton.jit
def _householder_product_column_kernel(
    A,
    tau,
    Q,
    M,
    N,
    K,
    a_batch_stride,
    q_batch_stride,
    tau_batch_stride,
    stride_row,
    BLOCK_M: tl.constexpr,
):
    """One program per (batch, column); the Q column lives in registers.

    ``BLOCK_M`` is a power of two >= M (the launcher guarantees it); lanes
    [M, BLOCK_M) are initialised to 0, never selected into ``v``, and masked
    out of the final store.
    """
    pid_batch = tl.program_id(0)
    col = tl.program_id(1)

    rows = tl.arange(0, BLOCK_M)
    row_mask = rows < M
    safe_rows = tl.minimum(rows, M - 1)

    a_base = A + pid_batch * a_batch_stride
    q_base = Q + pid_batch * q_batch_stride
    tau_base = tau + pid_batch * tau_batch_stride

    # Q[:, col] starts as I[:, col] = (rows == col ? 1 : 0).
    q_vec = tl.where(rows == col, 1.0, 0.0).to(A.dtype.element_ty)

    # Only reflectors k <= col affect column col: v_k[col] == 0 exactly for
    # k > col (v_k is zero above its index), so H_k e_col = e_col exactly and
    # the trailing no-ops can be dropped.  This halves the k-iterations.
    k_upper = tl.minimum(col, K - 1)
    for k_idx in range(k_upper, -1, -1):
        tau_k = tl.load(tau_base + k_idx)
        # v(i) = e_i + sum_{j>i} A[j, i] * e_j; rows >= M contribute 0.
        a_col = tl.load(a_base + safe_rows * stride_row + k_idx)
        v = tl.where(
            rows == k_idx,
            1.0,
            tl.where(row_mask & (rows > k_idx), a_col, 0.0),
        )
        dots = tl.sum(v * q_vec)  # 1-D (scalar) reduction: XPU-safe
        q_vec = q_vec - tau_k * (dots * v)

    q_ptrs = q_base + rows * stride_row + col
    tl.store(q_ptrs, q_vec, mask=row_mask)


@triton.jit
def _lhp_partial_dot_kernel(
    A,
    Q,
    PD,
    M,
    NBLK,
    a_batch_stride,
    q_batch_stride,
    k_idx,
    stride_row,
    col_stride,
    BLOCK_M: tl.constexpr,
):
    """Kernel A (large-M path): per (batch, m-block, column) partial dots.

    Loads only (A column k and the current Q block), writes a single scalar
    to the partial-dot buffer.  Grid: (batch, NBLK, n).
    """
    pid_batch = tl.program_id(0)
    mb = tl.program_id(1)
    col = tl.program_id(2)
    rows = tl.arange(0, BLOCK_M)
    r = mb * BLOCK_M + rows
    msk = r < M
    safe_r = tl.minimum(r, M - 1)

    a_base = A + pid_batch * a_batch_stride
    q_base = Q + pid_batch * q_batch_stride
    a_col = tl.load(a_base + safe_r * stride_row + k_idx)
    v = tl.where(r == k_idx, 1.0, tl.where(msk & (r > k_idx), a_col, 0.0))
    qb = tl.load(q_base + safe_r * stride_row + col)
    qb = tl.where(msk, qb, 0.0)
    s = tl.sum(v * qb)
    tl.store(PD + (pid_batch * NBLK + mb) * col_stride + col, s)


@triton.jit
def _lhp_reduce_kernel(
    PD,
    D,
    NBLK,
    col_stride,
    BLOCK_N: tl.constexpr,
):
    """Kernel B (large-M path): dots = sum over m-blocks of the partial dots.

    Grid: (batch, n).  BLOCK_N is a power of two >= NBLK.
    """
    pid_batch = tl.program_id(0)
    col = tl.program_id(1)
    offs = tl.arange(0, BLOCK_N)
    p = tl.load(
        PD + pid_batch * NBLK * col_stride + offs * col_stride + col,
        mask=offs < NBLK,
        other=0.0,
    )
    s = tl.sum(p)
    tl.store(D + pid_batch * col_stride + col, s)


@triton.jit
def _lhp_update_kernel(
    A,
    tau,
    Q,
    D,
    M,
    NBLK,
    a_batch_stride,
    tau_batch_stride,
    q_batch_stride,
    k_idx,
    stride_row,
    col_stride,
    BLOCK_M: tl.constexpr,
):
    """Kernel C (large-M path): Q block -= tau_k * dots * v block.

    Load-then-store of the same block (no store-before-load of the same
    address within the program, which the XPU backend miscompiles).
    Grid: (batch, NBLK, n).
    """
    pid_batch = tl.program_id(0)
    mb = tl.program_id(1)
    col = tl.program_id(2)
    rows = tl.arange(0, BLOCK_M)
    r = mb * BLOCK_M + rows
    msk = r < M
    safe_r = tl.minimum(r, M - 1)

    a_base = A + pid_batch * a_batch_stride
    q_base = Q + pid_batch * q_batch_stride
    tau_base = tau + pid_batch * tau_batch_stride
    tau_k = tl.load(tau_base + k_idx)
    a_col = tl.load(a_base + safe_r * stride_row + k_idx)
    v = tl.where(r == k_idx, 1.0, tl.where(msk & (r > k_idx), a_col, 0.0))
    qb = tl.load(q_base + safe_r * stride_row + col)
    qb = tl.where(msk, qb, 0.0)
    dots = tl.load(D + pid_batch * col_stride + col)
    qb = qb - tau_k * (dots * v)
    tl.store(q_base + r * stride_row + col, qb, mask=msk)


def _lhp_large_m(
    A_work,
    tau_work,
    m,
    n,
    k,
    batch_size,
    a_batch_stride,
    tau_batch_stride,
    stride_row,
):
    """Large-M (M > _ONE_SHOT_M) fallback: 3 launches per k, see module docstring."""
    Q = (
        torch.eye(m, n, dtype=A_work.dtype, device=A_work.device)
        .unsqueeze(0)
        .expand(batch_size, m, n)
        .contiguous()
    )
    nblk = triton.cdiv(m, _BLOCKED_BLOCK_M)
    col_stride = n
    q_batch_stride = m * n  # Q is (batch, m, n) contiguous
    pd = torch.empty(batch_size * nblk * n, dtype=A_work.dtype, device=A_work.device)
    dots = torch.empty(batch_size * n, dtype=A_work.dtype, device=A_work.device)
    block_n = triton.next_power_of_2(nblk)
    for k_idx in range(k - 1, -1, -1):
        _lhp_partial_dot_kernel[(batch_size, nblk, n)](
            A_work,
            Q,
            pd,
            m,
            nblk,
            a_batch_stride,
            q_batch_stride,
            k_idx,
            stride_row,
            col_stride,
            BLOCK_M=_BLOCKED_BLOCK_M,
            isCloseUnrollControl=True,
        )
        _lhp_reduce_kernel[(batch_size, n)](
            pd,
            dots,
            nblk,
            col_stride,
            BLOCK_N=block_n,
            isCloseUnrollControl=True,
        )
        _lhp_update_kernel[(batch_size, nblk, n)](
            A_work,
            tau_work,
            Q,
            dots,
            m,
            nblk,
            a_batch_stride,
            tau_batch_stride,
            q_batch_stride,
            k_idx,
            stride_row,
            col_stride,
            BLOCK_M=_BLOCKED_BLOCK_M,
            isCloseUnrollControl=True,
        )
    return Q


def linalg_householder_product(A, tau):
    """Computes the product of Householder matrices (orgqr).

    Given the output of torch.geqrf (Householder vectors A and scalars tau),
    computes Q = H(0) * H(1) * ... * H(k-1) where H(i) = I - tau[i] * v[i] * v[i]^T.
    """
    logger.debug("GEMS_KUNLUNXIN LINALG_HOUSEHOLDER_PRODUCT")

    # Only float32 and float64 supported: householder_product requires real floating point
    assert A.dtype in (
        torch.float32,
        torch.float64,
    ), f"linalg_householder_product only supports float32 and float64, got {A.dtype}"

    shape = A.shape
    if len(shape) < 2:
        raise ValueError("A must be at least 2D")

    m = shape[-2]
    n = shape[-1]
    k = tau.shape[-1]

    # Handle batch dimensions
    if len(shape) == 2:
        batch_size = 1
        A_work = A.unsqueeze(0).contiguous()
        tau_work = tau.unsqueeze(0).contiguous()
    else:
        batch_size = 1
        for d in shape[:-2]:
            batch_size *= d
        A_work = A.reshape(batch_size, m, n).contiguous()
        tau_work = tau.reshape(batch_size, k).contiguous()

    # Allocate output Q
    Q = torch.empty_like(A_work)
    if m == 0 or n == 0 or batch_size == 0:
        # Degenerate/empty output; the kernels below assume 0 < M and 0 < N.
        return Q.reshape(shape)

    a_batch_stride = A_work.stride(0)
    q_batch_stride = Q.stride(0)
    tau_batch_stride = tau_work.stride(0)
    stride_row = A_work.stride(1)  # n (columns per row)

    if m > _ONE_SHOT_M:
        Q = _lhp_large_m(
            A_work,
            tau_work,
            m,
            n,
            k,
            batch_size,
            a_batch_stride,
            tau_batch_stride,
            stride_row,
        )
        return Q.reshape(shape)

    grid = (batch_size, n)
    BLOCK_M = triton.next_power_of_2(m)
    _householder_product_column_kernel[grid](
        A_work,
        tau_work,
        Q,
        m,
        n,
        k,
        a_batch_stride,
        q_batch_stride,
        tau_batch_stride,
        stride_row,
        BLOCK_M=BLOCK_M,
        # The K-loop is an scf.for with a loop-carried register tensor
        # (q_vec); TritonXPUUnrollControl tiles the body without re-typing
        # that iter_arg and the module fails MLIR verification
        # ('arith.mulf'/'arith.cmpi' type mismatch). Skipping the pass for
        # this kernel only is the standard fix (see searchsorted.py).
        isCloseUnrollControl=True,
    )
    return Q.reshape(shape)
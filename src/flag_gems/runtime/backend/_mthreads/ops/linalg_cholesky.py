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

"""Triton implementation of torch.linalg.cholesky (lower/upper factor).

Three paths, each one program per matrix in the batch and a single launch:

* ``_chol_tile_kernel`` (n <= 64): the whole lower triangle lives in one
  register tile; each column is extracted by a masked reduction along the
  contiguous axis, its factor column is written straight to the output, and the
  trailing sub-tile is updated by a mask-free rank-1 update.
* ``_chol_blocked_kernel`` (n >= 128, power-of-two n divisible by 64):
  left-looking blocked Cholesky. Every correction against previously computed
  block columns is a ``tl.dot`` matrix multiply; only the diagonal-block
  factorization and the short panel substitution stay sequential.
* ``_chol_lower_kernel`` (fallback): columns are processed sequentially and the
  accumulation is vectorized over rows, reading the factor back from the
  output buffer.

The strict upper triangle is written as zeros, matching the reference, which
zeroes the unused triangle on this device.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.linalg_cholesky import linalg_cholesky as default_linalg_cholesky
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# PyTorch's linalg_cholesky only accepts fp32/fp64, and Moore Threads hardware
# does not support fp64 compute, so only fp32 is specialized; fp64 falls back
# to the generic implementation.
_SUPPORTED_DTYPES = {torch.float32}


@triton.autotune(
    configs=[triton.Config({}, num_warps=w) for w in (1, 2, 4, 8)],
    key=["n", "esize"],
)
@triton.jit
def _chol_tile_kernel(
    A,
    L,
    n,
    esize,
    sa_b,
    sa_r,
    sl_b,
    sl_r,
    BLOCK_N: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_N)
    a_ptr = A + pid.to(tl.int64) * sa_b + rows[:, None] * sa_r + cols[None, :]

    if EVEN:
        tile = tl.load(a_ptr, mask=rows[:, None] >= cols[None, :], other=0.0)
    else:
        tile = tl.load(
            a_ptr,
            mask=(rows[:, None] < n)
            & (cols[None, :] < n)
            & (rows[:, None] >= cols[None, :]),
            other=0.0,
        )

    l_col = L + pid.to(tl.int64) * sl_b + rows * sl_r
    for k in range(0, n):
        # Column k of the working matrix, indexed by row.
        colk = tl.sum(tl.where(cols[None, :] == k, tile, 0.0), axis=1)
        d = tl.sqrt(tl.sum(tl.where(rows == k, colk, 0.0), axis=0))
        lu = tl.where(rows > k, colk / d, 0.0)
        lv = tl.where(rows == k, d, lu)
        if EVEN:
            tl.store(l_col + k, lv)
        else:
            tl.store(l_col + k, lv, mask=rows < n)
        tile = tile - lu[:, None] * lu[None, :]


@triton.autotune(
    configs=[
        triton.Config({"BK": 64}, num_warps=16),
        triton.Config({"BK": 64}, num_warps=8),
        triton.Config({"BK": 32}, num_warps=8),
        triton.Config({"BK": 128}, num_warps=16),
    ],
    key=["n", "esize"],
)
@triton.jit
def _chol_blocked_kernel(
    A,
    L,
    n,
    esize,
    sa_b,
    sa_r,
    sl_b,
    sl_r,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    """Left-looking blocked Cholesky for one matrix (n a multiple of BK).

    For every block column the diagonal block is corrected by dots against all
    previously computed block columns and factored sequentially; the whole
    panel below it is corrected by one dot per previous block column and then
    solved by a single BK-step substitution over a (BM x BK) register tile.
    """
    pid = tl.program_id(0)
    lp = L + pid.to(tl.int64) * sl_b
    ap = A + pid.to(tl.int64) * sa_b
    rows = tl.arange(0, BM)
    cc = tl.arange(0, BK)

    # Zero the strict upper triangle; the lower triangle is fully written below.
    for jb in range(0, n, BK):
        for ib in range(0, jb, BK):
            tl.store(
                lp + (ib + cc)[:, None] * sl_r + (jb + cc)[None, :],
                tl.zeros((BK, BK), dtype=A.dtype.element_ty),
            )
        tl.store(
            lp + (jb + cc)[:, None] * sl_r + (jb + cc)[None, :],
            0.0,
            mask=cc[:, None] < cc[None, :],
        )

    for jb in range(0, n, BK):
        # ---- diagonal block ----
        T11 = tl.load(
            ap + (jb + cc)[:, None] * sa_r + (jb + cc)[None, :],
            mask=cc[:, None] >= cc[None, :],
            other=0.0,
        )
        for kc in range(0, jb, BK):
            Lk = tl.load(lp + (jb + cc)[:, None] * sl_r + (kc + cc)[None, :])
            T11 = T11 - tl.dot(Lk, tl.trans(Lk), input_precision="ieee")
        for k in range(0, BK):
            colk = tl.sum(tl.where(cc[None, :] == k, T11, 0.0), axis=1)
            d = tl.sqrt(tl.sum(tl.where(cc == k, colk, 0.0), axis=0))
            lu = tl.where(cc > k, colk / d, 0.0)
            lv = tl.where(cc == k, d, lu)
            tl.store(lp + (jb + cc) * sl_r + (jb + k), lv)
            T11 = T11 - lu[:, None] * lu[None, :]
        tl.debug_barrier()

        # ---- panel: one (BM x BK) tile covering every row below the block ----
        if jb + BK < n:
            pmask = rows[:, None] >= jb + BK
            T = tl.load(
                ap + rows[:, None] * sa_r + (jb + cc)[None, :], mask=pmask, other=0.0
            )
            for kc in range(0, jb, BK):
                Li = tl.load(lp + rows[:, None] * sl_r + (kc + cc)[None, :])
                Lj = tl.load(lp + (jb + cc)[:, None] * sl_r + (kc + cc)[None, :])
                T = T - tl.dot(Li, tl.trans(Lj), input_precision="ieee")
            P = tl.zeros((BM, BK), dtype=A.dtype.element_ty)
            for c in range(0, BK):
                tcol = tl.sum(tl.where(cc[None, :] == c, T, 0.0), axis=1)
                lrow = tl.load(lp + (jb + c) * sl_r + (jb + cc), mask=cc < c, other=0.0)
                acc = tl.sum(P * lrow[None, :], axis=1)
                dg = tl.load(lp + (jb + c) * sl_r + (jb + c))
                pcol = (tcol - acc) / dg
                P = tl.where(cc[None, :] == c, pcol[:, None], P)
            tl.store(lp + rows[:, None] * sl_r + (jb + cc)[None, :], P, mask=pmask)
            tl.debug_barrier()


@triton.jit
def _chol_lower_kernel(
    A,
    L,
    n,
    sa_b,
    sa_r,
    sl_b,
    sl_r,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    a_base = pid.to(tl.int64) * sa_b
    l_base = pid.to(tl.int64) * sl_b
    rows = tl.arange(0, BLOCK_N)
    ks = tl.arange(0, BLOCK_K)

    a_rows = A + a_base + rows * sa_r
    l_rows = L + l_base + rows * sl_r
    if EVEN:
        rmask = rows < BLOCK_N
    else:
        rmask = rows < n

    for j in range(0, n):
        acc = tl.zeros((BLOCK_N,), dtype=A.dtype.element_ty)
        for kc in range(0, j, BLOCK_K):
            kk = kc + ks
            kmask = kk < j
            ljk = tl.load(L + l_base + j * sl_r + kk, mask=kmask, other=0.0)
            lik = tl.load(
                l_rows[:, None] + kk[None, :],
                mask=rmask[:, None] & kmask[None, :],
                other=0.0,
            )
            acc += tl.sum(lik * ljk[None, :], axis=1)

        a_diag = tl.load(A + a_base + j * sa_r + j)
        a_col = tl.load(a_rows + j, mask=rmask, other=0.0)
        diag_acc = tl.sum(tl.where(rows == j, acc, 0.0), axis=0)
        d = tl.sqrt(a_diag - diag_acc)
        val = tl.where(rows < j, 0.0, tl.where(rows == j, d, (a_col - acc) / d))
        if EVEN:
            tl.store(l_rows + j, val)
        else:
            tl.store(l_rows + j, val, mask=rmask)
        # Column j is read by every later column, possibly from other lanes.
        tl.debug_barrier()


def _use_triton_kernel(A):
    """Whether ``A`` can be handled by the MUSA kernels."""
    if not isinstance(A, torch.Tensor):
        return False
    if A.device.type != "musa" or A.dtype not in _SUPPORTED_DTYPES:
        return False
    if A.is_complex():
        return False
    if A.dim() < 2 or A.shape[-1] != A.shape[-2]:
        return False
    return True


def linalg_cholesky(A, upper=False, *, out=None):
    logger.debug("GEMS_MTHREADS LINALG_CHOLESKY")

    if not _use_triton_kernel(A):
        return default_linalg_cholesky(A, upper)

    n = A.shape[-1]
    if out is not None:
        L = out
    else:
        L = torch.empty_like(A)

    if A.numel() == 0:
        return L.transpose(-2, -1) if upper else L

    if A.dim() == 2:
        Ab = A
        Lb = L
        sa_b = 0
        sl_b = 0
        batch = 1
    else:
        Ab = A.reshape(-1, n, n)
        Lb = L.reshape(-1, n, n)
        sa_b = Ab.stride(0)
        sl_b = Lb.stride(0)
        batch = Ab.shape[0]

    block_n = triton.next_power_of_2(n)
    even = block_n == n
    with torch_device_fn.device(A.device):
        _launch_cholesky(A, Ab, Lb, n, block_n, even, sa_b, sl_b, batch)

    if upper:
        return L.transpose(-2, -1)
    return L


def _launch_cholesky(A, Ab, Lb, n, block_n, even, sa_b, sl_b, batch):
    if block_n <= 64:
        _chol_tile_kernel[(batch,)](
            Ab,
            Lb,
            n,
            A.element_size(),
            sa_b,
            Ab.stride(-2),
            sl_b,
            Lb.stride(-2),
            BLOCK_N=block_n,
            EVEN=even,
        )
    elif even and n % 64 == 0:
        _chol_blocked_kernel[(batch,)](
            Ab,
            Lb,
            n,
            A.element_size(),
            sa_b,
            Ab.stride(-2),
            sl_b,
            Lb.stride(-2),
            BM=n,
        )
    else:
        _chol_lower_kernel[(batch,)](
            Ab,
            Lb,
            n,
            sa_b,
            Ab.stride(-2),
            sl_b,
            Lb.stride(-2),
            BLOCK_N=block_n,
            BLOCK_K=64,
            EVEN=even,
            num_warps=8,
            num_stages=1,
        )

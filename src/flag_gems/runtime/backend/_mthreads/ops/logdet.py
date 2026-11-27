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

"""logdet: logarithm of the determinant of (batches of) square matrices.

One Triton program owns one matrix and performs a partial-pivoted (LAPACK-style
row pivoting) LU factorization entirely in registers, accumulating
sum_i log|U_ii| and the sign of det(A). The result mirrors torch.logdet
(= log(sign) + logabsdet):

    out = logabsdet          if sign > 0
        = NaN                if sign < 0
        = -inf               if sign == 0 (exactly singular)

Instead of physically swapping rows, the elimination walks a permutation of the
rows (pivot row chosen among the not-yet-used rows). This removes the row
extraction/rewrite pair from every step: the sign of the permutation is the
inversion count of the pivot sequence, evaluated once after the loop, and
det(A) = sign(permutation) * prod(U_ii).
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.logdet import logdet as default_logdet
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64, so only fp32 is specialized;
# every other dtype falls back to the generic implementation.
_SUPPORTED_DTYPES = {torch.float32}

_MAX_BATCH_DIMS = 4
_REGISTER_TILE_LIMIT = 32


@triton.jit
def _pivot_combine(a_cand, a_val, a_idx, b_cand, b_val, b_idx):
    """Pick the larger magnitude (left wins ties), carrying signed value and row."""
    take_a = a_cand >= b_cand
    return (
        tl.where(take_a, a_cand, b_cand),
        tl.where(take_a, a_val, b_val),
        tl.where(take_a, a_idx, b_idx),
    )


@triton.jit
def _logdet_lu_kernel(
    inp_ptr,
    out_ptr,
    n,
    stride_m,
    stride_n,
    s0,
    s1,
    s2,
    s3,
    b0,
    b1,
    b2,
    b3,
    NB: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP64: tl.constexpr,
):
    pid = tl.program_id(0)

    # Decompose the flat batch index into the collapsed batch dimensions,
    # most-minor dimension first (b0/s0).
    base = 0
    rem = pid
    if NB >= 1:
        base += (rem % b0) * s0
        rem = rem // b0
    if NB >= 2:
        base += (rem % b1) * s1
        rem = rem // b1
    if NB >= 3:
        base += (rem % b2) * s2
        rem = rem // b2
    if NB >= 4:
        base += (rem % b3) * s3
        rem = rem // b3

    rows = tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < n) & (cols[None, :] < n)
    offs = base + rows[:, None] * stride_m + cols[None, :] * stride_n
    A = tl.load(inp_ptr + offs, mask=mask, other=0.0)

    if FP64:
        A = A.to(tl.float64)
        diag = tl.full((BLOCK_N,), 1.0, tl.float64)
    else:
        A = A.to(tl.float32)
        diag = tl.full((BLOCK_N,), 1.0, tl.float32)

    # Padding must behave as the identity so padded rows/cols are inert.
    pad_diag = (rows[:, None] == cols[None, :]) & (~mask)
    A = tl.where(pad_diag, 1.0, A)

    parity = tl.zeros((), dtype=tl.int32)
    singular = tl.zeros((), dtype=tl.int32)
    unused = rows < n
    piv = rows

    for k in range(0, n):
        # column k of the current tile
        col = tl.sum(tl.where(cols[None, :] == k, A, 0.0), axis=1)

        # pivot: not-yet-used row holding the maximum |A[i, k]|. One tuple
        # reduction also returns the signed pivot value and the row index.
        cand = tl.where(unused & (rows < n), tl.abs(col), -1.0)
        pabs, akk, p = tl.reduce((cand, col, rows), 0, _pivot_combine)
        piv = tl.where(rows == k, p, piv)

        rowp = tl.sum(tl.where(rows[:, None] == p, A, 0.0), axis=0)
        # guarded divisor: a zero pivot yields mult == 0 instead of inf/NaN
        safe = tl.where(akk == 0.0, 1.0, akk)
        mult = tl.where(unused & (rows != p), col / safe, 0.0)

        # rank-1 Schur complement update of the trailing submatrix
        A = tl.where(
            unused[:, None] & (rows[:, None] != p) & (cols[None, :] > k),
            A - mult[:, None] * rowp[None, :],
            A,
        )

        unused = unused & (rows != p)
        diag = tl.where(rows == k, akk, diag)
        parity += (akk < 0.0).to(tl.int32)
        singular += (akk == 0.0).to(tl.int32)

    # log|det| = sum_i log|U_ii|: one vectorized log instead of one per step
    logabs = tl.sum(tl.log(tl.abs(diag)), axis=0)

    # sign of the row permutation: inversion count of the pivot sequence
    upper = (rows[:, None] < rows[None, :]) & (rows[None, :] < n)
    parity += tl.sum(tl.where(upper & (piv[:, None] > piv[None, :]), 1, 0))

    nan = float("nan")
    result = tl.where(parity % 2 != 0, nan, logabs)
    result = tl.where(singular != 0, float("-inf"), result)
    tl.store(out_ptr + pid, result.to(out_ptr.dtype.element_ty))


def _batch_dims(inp):
    """Collapse the batch axes into packed (size, stride) runs, minor axis first."""
    sizes = []
    strides = []
    for d in range(inp.dim() - 2):
        sz = int(inp.shape[d])
        if sz == 1:
            continue
        sizes.append(sz)
        strides.append(int(inp.stride(d)))

    cs = []
    cst = []
    for sz, st in zip(sizes, strides):
        if cst and cst[-1] == st * sz:
            # outer stride continues the inner run: merge onto the inner stride
            cs[-1] *= sz
            cst[-1] = st
        else:
            cs.append(sz)
            cst.append(st)

    b = [1, 1, 1, 1]
    s = [0, 0, 0, 0]
    for i in range(len(cs)):
        b[i] = cs[len(cs) - 1 - i]
        s[i] = cst[len(cs) - 1 - i]
    return len(cs), b, s


def _use_triton_kernel(inp):
    """Whether ``inp`` can be handled by the MUSA register-tile kernel."""
    if not isinstance(inp, torch.Tensor):
        return False
    if inp.device.type != "musa" or inp.dtype not in _SUPPORTED_DTYPES:
        return False
    if inp.is_complex() or inp.requires_grad:
        return False
    if inp.dim() < 2 or inp.shape[-1] != inp.shape[-2]:
        return False
    if int(inp.shape[-1]) > _REGISTER_TILE_LIMIT:
        return False
    if inp.dim() - 2 > _MAX_BATCH_DIMS:
        return False
    return True


def logdet(inp):
    logger.debug("GEMS_MTHREADS LOGDET")

    if not isinstance(inp, torch.Tensor):
        raise TypeError("logdet: input must be a torch.Tensor")
    if not _use_triton_kernel(inp):
        return default_logdet(inp)

    n = int(inp.shape[-1])
    batch_shape = inp.shape[:-2]
    batch_count = 1
    for sz in batch_shape:
        batch_count *= int(sz)

    if batch_count == 0:
        return torch.empty(batch_shape, dtype=inp.dtype, device=inp.device)

    out = torch.empty(batch_shape, dtype=inp.dtype, device=inp.device)

    nb, b, s = _batch_dims(inp)
    if nb > _MAX_BATCH_DIMS:
        return default_logdet(inp)

    # Tile width follows the matrix size: the per-step where-tree and the
    # tile-wide reductions then cover only the live elements.
    block_n = 4
    while block_n < n:
        block_n *= 2
    num_warps = 1 if block_n <= 16 else 2

    with torch_device_fn.device(out.device):
        _logdet_lu_kernel[(batch_count,)](
            inp,
            out,
            n,
            inp.stride(-2),
            inp.stride(-1),
            s[0],
            s[1],
            s[2],
            s[3],
            b[0],
            b[1],
            b[2],
            b[3],
            NB=nb,
            BLOCK_N=block_n,
            FP64=False,
            num_warps=num_warps,
        )
    return out

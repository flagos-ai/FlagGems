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

"""Kunlunxin (XPU) lu_unpack backend override.

Two backend-local deviations from the generic
``flag_gems/ops/lu_unpack.py`` implementation:

1. Empty-input short-circuit. When ``m == 0`` or ``n == 0`` (so ``k == 0``),
   ``triton.next_power_of_2(0) == 0`` makes ``tl.arange(0, BLOCK)`` fail to
   compile (assert arange end > start). The short-circuit mirrors the ATen
   native semantics for empty inputs:

   - P: ``(..., m, m)`` identity (no swap with k == 0); when m == 0 it is
     ``(0, 0)`` empty.
   - L: ``(..., m, k)`` empty; U: ``(..., k, n)`` empty.

2. Permutation-matrix (P) construction for ``m > 512``. The generic
   ``lu_unpack_p_kernel_large`` launches one program per row and replays the
   whole ``k``-step pivot loop inside every one of them, i.e. ``O(m * k)``
   serialized scalar pivot loads. On XPU that single kernel is ~82% of the
   whole operator at 1024x1024 / 4096x4096. Here the permutation is built
   once in ``O(k)`` (a single program applying the LAPACK ``ipiv`` swaps to a
   scratch index vector held in global memory) and then materialized with an
   ``O(m)`` vectorized scatter.

Everything else (L / U extraction, and the whole ``m <= 512`` P path) is
delegated to the generic implementation - same Triton kernels, no
CPU/ATen/native fallback involved.
"""

import torch
import triton
import triton.language as tl

from flag_gems.ops.lu_unpack import lu_unpack as _general_lu_unpack
from flag_gems.ops.lu_unpack import lu_unpack_out as _general_lu_unpack_out
from flag_gems.utils import libentry

# The generic vectorized P kernel (one program per batch, BLOCK_M lanes) is
# already launch-floor fast for m <= 512; only the per-row large path is
# replaced.
_P_SMALL_M = 512
# Tile width for the index-vector init / scatter kernels. Plain masked
# load+store only (no reduction), i.e. the pattern the generic L/U kernels
# already rely on.
_P_TILE = 256


@libentry()
@triton.jit
def _lu_perm_init_kernel(
    s_ptr,
    m,
    s_stride_b,
    TILE: tl.constexpr,
):
    """s[b, v] = v -- identity index vector, one tile per program."""
    batch_id = tl.program_id(0)
    offs = tl.program_id(1) * TILE + tl.arange(0, TILE)
    mask = offs < m
    tl.store(s_ptr + batch_id * s_stride_b + offs, offs.to(tl.int32), mask=mask)


@libentry()
@triton.jit
def _lu_perm_swap_kernel(
    pivots_ptr,
    s_ptr,
    k,
    pivots_stride_b,
    pivots_stride_k,
    s_stride_b,
):
    """Apply the LAPACK ipiv row interchanges to s in place, one program per
    batch element.

    ``s`` starts as the identity and ends up mapping *column* v of P to the
    row that carries its 1.0, i.e. ``P[s[v], v] = 1``. This is the inverse of
    the ``perm`` vector the generic kernels track (they swap the *values* i
    and j of ``perm``, which is the same as swapping the *positions* i and j
    of its inverse).

    The swap is branchless: for ``j == i`` both stores write the same value
    to the same address.

    ``tl.debug_barrier()`` is mandatory on this backend: TritonXPU stages
    global accesses through local memory (gm2lm / lm2gm), so without an
    explicit flush a later iteration reads a stale copy of ``s`` and the
    permutation silently comes out wrong (measured: only ~285 of 600 indices
    distinct). See the solution note for the isolated probe.
    """
    batch_id = tl.program_id(0)
    pivots_base = batch_id * pivots_stride_b
    s_base = batch_id * s_stride_b
    for i in range(k):
        j = tl.load(pivots_ptr + pivots_base + i * pivots_stride_k) - 1
        a = tl.load(s_ptr + s_base + i)
        b = tl.load(s_ptr + s_base + j)
        tl.debug_barrier()
        tl.store(s_ptr + s_base + i, b)
        tl.store(s_ptr + s_base + j, a)
        tl.debug_barrier()


@libentry()
@triton.jit
def _lu_perm_scatter_kernel(
    s_ptr,
    p_ptr,
    m,
    s_stride_b,
    p_stride_b,
    p_stride_m,
    p_stride_n,
    TILE: tl.constexpr,
):
    """Scatter P[s[v], v] = 1.0 into the pre-zeroed P, one tile per program."""
    batch_id = tl.program_id(0)
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    mask = cols < m
    rows = tl.load(s_ptr + batch_id * s_stride_b + cols, mask=mask, other=0)
    offsets = batch_id * p_stride_b + rows * p_stride_m + cols * p_stride_n
    tl.store(p_ptr + offsets, tl.full([TILE], 1.0, dtype=tl.float32), mask=mask)


def _lu_unpack_permutation(LU_data, LU_pivots, batch_dims, batch_size, m, k):
    P = torch.zeros(*batch_dims, m, m, device=LU_data.device, dtype=LU_data.dtype)
    s = torch.empty(batch_size, m, device=LU_data.device, dtype=torch.int32)
    num_tiles = triton.cdiv(m, _P_TILE)
    grid = (batch_size, num_tiles)

    _lu_perm_init_kernel[grid](s, m, s.stride(0), _P_TILE)
    _lu_perm_swap_kernel[(batch_size,)](
        LU_pivots,
        s,
        k,
        LU_pivots.stride(-2) if LU_pivots.dim() > 1 else 0,
        LU_pivots.stride(-1),
        s.stride(0),
    )
    _lu_perm_scatter_kernel[grid](
        s,
        P,
        m,
        s.stride(0),
        P.stride(-3) if len(batch_dims) > 0 else 0,
        P.stride(-2),
        P.stride(-1),
        _P_TILE,
    )
    return P


def _empty_lu_unpack_result(LU_data, LU_pivots, unpack_data, unpack_pivots):
    m, n = LU_data.shape[-2], LU_data.shape[-1]
    batch_dims = LU_data.shape[:-2]
    device = LU_data.device
    dtype = LU_data.dtype
    k = min(m, n)
    if unpack_pivots:
        # k == 0: no pivot swap, P is the identity permutation
        P = torch.zeros(*batch_dims, m, m, device=device, dtype=dtype)
        if m > 0:
            P.diagonal(dim1=-2, dim2=-1).fill_(1.0)
    else:
        P = torch.empty(0, device=device, dtype=dtype)
    if unpack_data:
        L = torch.empty(*batch_dims, m, k, device=device, dtype=dtype)
        U = torch.empty(*batch_dims, k, n, device=device, dtype=dtype)
    else:
        L = torch.empty(0, device=device, dtype=dtype)
        U = torch.empty(0, device=device, dtype=dtype)
    return (P, L, U)


def lu_unpack(LU_data, LU_pivots, unpack_data=True, unpack_pivots=True):
    m, n = LU_data.shape[-2], LU_data.shape[-1]
    if m == 0 or n == 0:
        return _empty_lu_unpack_result(LU_data, LU_pivots, unpack_data, unpack_pivots)
    if unpack_pivots and m > _P_SMALL_M:
        batch_dims = LU_data.shape[:-2]
        batch_size = 1
        for dim in batch_dims:
            batch_size *= dim
        P = _lu_unpack_permutation(
            LU_data, LU_pivots, batch_dims, batch_size, m, min(m, n)
        )
        # L / U still come from the generic Triton kernels; unpack_pivots is
        # switched off there so its per-row P kernel is never launched.
        _, L, U = _general_lu_unpack(LU_data, LU_pivots, unpack_data, False)
        return (P, L, U)
    return _general_lu_unpack(LU_data, LU_pivots, unpack_data, unpack_pivots)


def lu_unpack_out(
    LU_data, LU_pivots, unpack_data=True, unpack_pivots=True, *, P=None, L=None, U=None
):
    m, n = LU_data.shape[-2], LU_data.shape[-1]
    if m == 0 or n == 0 or (unpack_pivots and m > _P_SMALL_M):
        if m == 0 or n == 0:
            P_result, L_result, U_result = _empty_lu_unpack_result(
                LU_data, LU_pivots, unpack_data, unpack_pivots
            )
        else:
            P_result, L_result, U_result = lu_unpack(
                LU_data, LU_pivots, unpack_data, unpack_pivots
            )
        # Write back through the raw native strided-copy engine
        # (``aten::_copy_from``) instead of the gems-registered ``copy_``
        # to avoid a nested dispatch through the overridden operator.
        if P is not None and P_result.numel() > 0:
            torch.ops.aten._copy_from(P_result, P, False)
        else:
            P = P_result
        if L is not None and L_result.numel() > 0:
            torch.ops.aten._copy_from(L_result, L, False)
        else:
            L = L_result
        if U is not None and U_result.numel() > 0:
            torch.ops.aten._copy_from(U_result, U, False)
        else:
            U = U_result
        return (P, L, U)
    return _general_lu_unpack_out(
        LU_data, LU_pivots, unpack_data, unpack_pivots, P=P, L=L, U=U
    )

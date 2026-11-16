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
"""Kunlunxin XPU3 override for ``aten::_gather_sparse_backward``.

``_gather_sparse_backward(self, dim, index, grad)`` builds the *uncoalesced*
sparse COO gradient of ``torch.gather``: one non-zero per element of ``grad``,
``values == grad.reshape(-1)``, ``indices[dim] == index.reshape(-1)`` and every
other index row is the flat non-zero id unravelled over ``grad.shape``.

Why the generic implementation is slow on XPU3
----------------------------------------------
The generic ``flag_gems/ops/_gather_sparse_backward.py`` emits a single masked
kernel that is otherwise already affine (both output rows are contiguous stores
and the index load is unit stride).  Its cost is entirely ALU/mask overhead:

* ``offs`` is int64 and every coordinate uses ``%`` / ``//`` by a *runtime*
  scalar, which XPU3 lowers to long software sequences (~30 instructions per
  op); int64 makes it worse still.
* the store/load masks are always emitted, even when ``nnz % BLOCK == 0``, and
  the masked-memory path is ~2x the unmasked one on this backend
  (``KERNEL_OPT_EXPERIENCE.md`` section 3.0/3.6).
* ``BLOCK_SIZE = 1024`` leaves tens of thousands of programs on the large
  cases; each program costs ~60 ns of fixed launch/setup
  (``pattern-xpu3-gather-wall-vs-block-dma``).

Measured pre-fix (``harness/perf_ir/prof_gather_sparse_backward_pre.log``): one
``_gather_sparse_backward_kernel`` per call, 100 % of device time, effective
~198 GB/s of a ~1.2 TB/s HBM budget => 0.155-0.42x of the vendor reference.

Fast path (all benchmark shapes: powers of two)
-----------------------------------------------
* int32 flat offsets (``nnz < 2**31``) and power-of-two dim sizes turn the
  unravelling into one shift + one mask per row: ``coord_d = (offs >> S_d) &
  (size_d - 1)``, with ``S_d`` the cumulative log2 of the sizes below ``d``.
* ``EVEN`` constexpr branch drops the mask entirely when ``nnz % BLOCK == 0``.
* ``BLOCK = 16384`` keeps the program count low.
* For ``nnz >= 4M`` the index row (a plain int64 contiguous copy) is moved by
  ``tle_copy`` (TMA tile DMA) while a second launch only generates the
  coordinate rows; probing showed 1.7x over the fused form at rank 3
  (``harness/perf_ir/probe_gsb_round4.log``).

Fallbacks (device kernels only, no aten/CPU path)
-------------------------------------------------
* non-power-of-two extents or ``nnz >= 2**31`` -> same kernels with runtime
  int64 ``prod-below`` / ``size`` pairs (the generic div/mod semantics).
* ``rank > 4`` -> the generic Triton implementation, unchanged behaviour.
* ``tle_copy`` refusing the layout -> an in-module Triton row-copy kernel.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

from ..utils.tle_copy import tle_copy

logger = logging.getLogger(__name__)

_BLOCK = 16384
_GEN_BLOCK = 8192
# Below this nnz the extra launch of the split path costs more than it saves
# (measured crossover between nnz 0.5M and 8.4M).
_SPLIT_MIN_NNZ = 1 << 22
_MAX_RANK = 4
_INT32_MAX = (1 << 31) - 1


@triton.jit
def _gsb_store(ptr, offs, val, mask, EVEN: tl.constexpr):
    if EVEN:
        tl.store(ptr + offs, val)
    else:
        tl.store(ptr + offs, val, mask=mask)


@triton.jit
def _gsb_load_idx(idx_ptr, offs, mask, EVEN: tl.constexpr):
    if EVEN:
        return tl.load(idx_ptr + offs)
    return tl.load(idx_ptr + offs, mask=mask, other=0)


@triton.jit
def _gsb_pow2_kernel(
    idx_ptr,
    out_ptr,
    nnz,
    skip,
    s0,
    m0,
    s1,
    m1,
    s2,
    m2,
    s3,
    m3,
    RANK: tl.constexpr,
    DIM: tl.constexpr,
    EVEN: tl.constexpr,
    COORDS: tl.constexpr,
    IDX_ROW: tl.constexpr,
    BCAST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Power-of-two extents.

    Coordinates are unravelled right-to-left in one chained shift/mask (the
    *independent* ``(offs >> cum_shift) & mask`` form trips the XPU3 MLIR
    ``UnrollControl`` pass -- see the wiki entry).  Rows are emitted from the
    last written row down so ``cur`` is shifted only once per dimension.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < nnz
    if IDX_ROW:
        if BCAST:
            # A single index element broadcasts to every non-zero.
            idx_v = tl.zeros([BLOCK], dtype=tl.int64) + tl.load(idx_ptr)
        else:
            idx_v = _gsb_load_idx(idx_ptr, offs, mask, EVEN)

    cur = offs >> skip
    if RANK > 3:
        if DIM == 3:
            if IDX_ROW:
                _gsb_store(out_ptr + 3 * nnz, offs, idx_v, mask, EVEN)
        else:
            if COORDS:
                _gsb_store(out_ptr + 3 * nnz, offs, (cur & m3).to(tl.int64), mask, EVEN)
        cur = cur >> s3
    if RANK > 2:
        if DIM == 2:
            if IDX_ROW:
                _gsb_store(out_ptr + 2 * nnz, offs, idx_v, mask, EVEN)
        else:
            if COORDS:
                _gsb_store(out_ptr + 2 * nnz, offs, (cur & m2).to(tl.int64), mask, EVEN)
        cur = cur >> s2
    if RANK > 1:
        if DIM == 1:
            if IDX_ROW:
                _gsb_store(out_ptr + nnz, offs, idx_v, mask, EVEN)
        else:
            if COORDS:
                _gsb_store(out_ptr + nnz, offs, (cur & m1).to(tl.int64), mask, EVEN)
        cur = cur >> s1
    if DIM == 0:
        if IDX_ROW:
            _gsb_store(out_ptr, offs, idx_v, mask, EVEN)
    else:
        if COORDS:
            _gsb_store(out_ptr, offs, (cur & m0).to(tl.int64), mask, EVEN)


@triton.jit
def _gsb_general_kernel(
    idx_ptr,
    out_ptr,
    nnz,
    skip,
    z0,
    z1,
    z2,
    z3,
    RANK: tl.constexpr,
    DIM: tl.constexpr,
    EVEN: tl.constexpr,
    COORDS: tl.constexpr,
    IDX_ROW: tl.constexpr,
    BCAST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Arbitrary extents: the same chain with ``//`` / ``%`` per dimension."""
    pid = tl.program_id(0)
    offs = (pid * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    mask = offs < nnz
    if IDX_ROW:
        if BCAST:
            idx_v = tl.zeros([BLOCK], dtype=tl.int64) + tl.load(idx_ptr)
        else:
            idx_v = _gsb_load_idx(idx_ptr, offs, mask, EVEN)

    cur = offs // skip
    if RANK > 3:
        if DIM == 3:
            if IDX_ROW:
                _gsb_store(out_ptr + 3 * nnz, offs, idx_v, mask, EVEN)
        else:
            if COORDS:
                _gsb_store(out_ptr + 3 * nnz, offs, cur % z3, mask, EVEN)
        cur = cur // z3
    if RANK > 2:
        if DIM == 2:
            if IDX_ROW:
                _gsb_store(out_ptr + 2 * nnz, offs, idx_v, mask, EVEN)
        else:
            if COORDS:
                _gsb_store(out_ptr + 2 * nnz, offs, cur % z2, mask, EVEN)
        cur = cur // z2
    if RANK > 1:
        if DIM == 1:
            if IDX_ROW:
                _gsb_store(out_ptr + nnz, offs, idx_v, mask, EVEN)
        else:
            if COORDS:
                _gsb_store(out_ptr + nnz, offs, cur % z1, mask, EVEN)
        cur = cur // z1
    if DIM == 0:
        if IDX_ROW:
            _gsb_store(out_ptr, offs, idx_v, mask, EVEN)
    else:
        if COORDS:
            _gsb_store(out_ptr, offs, cur % z0, mask, EVEN)


def _prod(seq):
    prod = 1
    for v in seq:
        prod *= v
    return prod


def _log2(n):
    return n.bit_length() - 1


def _next_pow2(n):
    return 1 << (n - 1).bit_length() if n > 1 else 1


def _launch(
    index_flat,
    out,
    nnz,
    dim,
    rank,
    sizes,
    bcast,
    pow2,
    coords=True,
    idx_row=True,
):
    """Launch the coordinate / index-row kernel for one output."""
    if rank > _MAX_RANK or len(sizes) > _MAX_RANK:
        return False

    # Rows beyond ``rank`` only contribute to the flat coordinate, so the chain
    # starts at ``offs >> skip`` / ``offs // skip``.
    trailing = sizes[rank:]

    if pow2:
        block = _BLOCK if nnz >= _BLOCK else max(64, _next_pow2(nnz))
        skip = _log2(_prod(trailing)) if trailing else 0
        scal = (
            _log2(sizes[0]) if rank > 0 else 0,
            sizes[0] - 1 if rank > 0 else 0,
            _log2(sizes[1]) if rank > 1 else 0,
            sizes[1] - 1 if rank > 1 else 0,
            _log2(sizes[2]) if rank > 2 else 0,
            sizes[2] - 1 if rank > 2 else 0,
            _log2(sizes[3]) if rank > 3 else 0,
            sizes[3] - 1 if rank > 3 else 0,
        )
        kernel = _gsb_pow2_kernel
    else:
        block = _GEN_BLOCK if nnz >= _GEN_BLOCK else max(64, _next_pow2(nnz))
        skip = _prod(trailing) if trailing else 1
        scal = (
            sizes[0] if rank > 0 else 1,
            sizes[1] if rank > 1 else 1,
            sizes[2] if rank > 2 else 1,
            sizes[3] if rank > 3 else 1,
        )
        kernel = _gsb_general_kernel

    grid = ((nnz + block - 1) // block,)
    kernel[grid](
        index_flat,
        out,
        nnz,
        skip,
        *scal,
        rank,
        dim,
        nnz % block == 0,
        coords,
        idx_row,
        bcast,
        block,
    )
    return True


def _gather_sparse_backward(self, dim, index, grad):
    logger.debug("GEMS _GATHER_SPARSE_BACKWARD")

    ndim = self.ndim

    # The native op has two dedicated scalar paths that are *not* symmetric:
    # a 0-dim ``self`` ignores ``index``/``dim`` and keeps every grad value with
    # a ``(0, grad.numel())`` index tensor (sparse_dim 0), while a 0-dim ``grad``
    # on a non-scalar ``self`` keeps the real ``index.view(1, 1)``.  Both run
    # before the equal-rank check.  They are metadata-only, so they stay as in
    # the generic implementation.
    if ndim == 0:
        sparse_indices = torch.empty(
            (0, grad.numel()), dtype=torch.int64, device=index.device
        )
        return torch.sparse_coo_tensor(
            sparse_indices, grad, self.size(), device=self.device
        )

    if grad.ndim == 0:
        sparse_indices = index.reshape(1, 1)
        if sparse_indices.dtype != torch.int64:
            sparse_indices = sparse_indices.to(torch.int64)
        return torch.sparse_coo_tensor(
            sparse_indices, grad, self.size(), device=self.device
        )

    nnz = grad.numel()

    # The sparse values are simply the flattened gradient.
    values = grad.reshape(-1)

    # An empty grad gives nnz == 0; the row count is still ``self.ndim``.
    if nnz == 0:
        sparse_indices = torch.empty((ndim, 0), dtype=torch.int64, device=index.device)
        return torch.sparse_coo_tensor(
            sparse_indices, values, self.size(), device=self.device
        )

    # ``grad`` supplies the non-zero count and the extents the flat coordinate is
    # unravelled over; a lower-rank ``grad`` indexes past its own extents.
    if grad.ndim < ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-grad.ndim}, {grad.ndim - 1}], but got {grad.ndim})"
        )

    index_flat = index.reshape(-1).contiguous()
    # A single index element broadcasts to all nnz entries; otherwise it must
    # line up one-to-one with the gradient.
    numel = index_flat.numel()
    if numel == 1:
        bcast = True
    elif numel == nnz:
        bcast = False
    else:
        raise IndexError(
            f"index must have 1 or {nnz} elements to match the gradient, "
            f"got {numel}"
        )

    # The native op validates nothing about ``dim``: a dim outside [0, ndim)
    # simply never matches, so every coordinate stays unravelled.
    dim = dim if dim >= 0 else dim + ndim

    if index_flat.dtype != torch.int64:
        index_flat = index_flat.to(torch.int64)

    sparse_indices = torch.empty((ndim, nnz), dtype=torch.int64, device=index.device)

    sizes = tuple(int(s) for s in grad.shape)
    pow2 = nnz <= _INT32_MAX and all(
        s > 0 and (s & (s - 1)) == 0 for s in sizes
    )
    if ndim > _MAX_RANK or len(sizes) > _MAX_RANK:
        # Deeper tensors keep the generic implementation, which is the same
        # device-side unravelling kernel (no aten/CPU path involved).
        from flag_gems.ops._gather_sparse_backward import (
            _gather_sparse_backward as _generic,
        )

        return _generic(self, dim, index, grad)

    with torch_device_fn.device(index.device):
        if pow2 and nnz >= _SPLIT_MIN_NNZ and not bcast:
            # Split: one launch generates the coordinate rows, the index row is
            # a plain contiguous int64 copy moved by the TMA tile kernel.  The
            # fused form is 1.7x slower at rank 3 (probe_gsb_round4.log).
            if ndim > 1:
                _launch(
                    index_flat,
                    sparse_indices,
                    nnz,
                    dim,
                    ndim,
                    sizes,
                    bcast,
                    pow2,
                    coords=True,
                    idx_row=False,
                )
            if not tle_copy(index_flat, sparse_indices[dim]):
                # Layout tle cannot express (e.g. an odd nnz leaves the row
                # base 8-byte aligned only): fall back to our own row kernel.
                _launch(
                    index_flat,
                    sparse_indices,
                    nnz,
                    dim,
                    ndim,
                    sizes,
                    bcast,
                    pow2,
                    coords=False,
                    idx_row=True,
                )
        else:
            _launch(index_flat, sparse_indices, nnz, dim, ndim, sizes, bcast, pow2)

    return torch.sparse_coo_tensor(
        sparse_indices, values, self.size(), device=self.device
    )

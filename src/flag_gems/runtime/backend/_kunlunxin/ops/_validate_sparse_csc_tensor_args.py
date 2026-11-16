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

# Kunlunxin (XPU) override of flag_gems.ops._validate_sparse_csc_tensor_args.
#
# Root cause on XPU3: the generic kernel finishes the per-segment check with
#   tl.device_assert(tl.max(bad.to(tl.int32), 0) == 0, "...")
# The XPU backend cannot lower `tl.device_assert`:
#   * `tt.reduce` feeding a `tt.assert` fails to legalize in
#     ConvertTritonXPUToLLVM ("failed to legalize operation 'tt.reduce'"), and
#   * a bare `tt.assert` lowers to an undefined `__assert_fail` symbol, so the
#     kernel never links.
# Both make the operator uncompilable (every functional/benchmark case fails).
#
# Fix (same shape as _kunlunxin/ops/assert_async.py): keep the whole index
# check on the XPU device, but instead of aborting on-device, each program
# reduces its own `bad` lanes to one int32 verdict and stores it to a small
# device buffer.  The host reads the verdict buffer back (an implicit stream
# sync + tiny D2H copy, not a torch numeric op) and raises the same RuntimeError
# ATen raises.  The validated index data is never touched by torch; only the
# already-computed boolean verdict crosses to the host.
#
# The original per-segment kernel also cost ~10 us per compressed segment on
# XPU3 (one program each): the 4096-row benchmark case spent ~0.9 ms just in
# program scheduling.  This override instead uses two *vectorised* flat passes
# and never loops per segment:
#   1. an endpoint pass over compressed rows (TILE rows/program) validates the
#      `comp[r]` endpoints and scatters a "segment start" flag per entry;
#   2. an entry pass over plain entries (TILE entries/program) validates
#      `0 <= v < dim` and the strictly-increasing-within-segment order, using
#      the start flags to skip the (legitimately unordered) segment boundaries.
# Loads keep affine `...+lanes` addresses (block DMA, not per-lane gather).
# Every executed masked load has lane 0 active because each grid is exact
# `ceil`, so none is all-false (an all-false masked load aborts a program on
# XPU3, see wiki/patterns/xpu3-all-false-mask-load-aborts); the one-entry case
# is compiled out through the `HAS_PREV` constexpr.

import logging
import operator

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# Flat tile widths for the two vectorised passes.
_ROWS_TILE = 256
_ENTRIES_TILE = 256

_ERR_MSG = (
    "invalid sparse compressed indices: bounds, endpoints, or sorted distinct "
    "entries"
)


@triton.jit
def _endpoint_kernel(
    comp,
    flag,
    eperr,
    cdim,
    dim,
    nnz,
    cs,
    BSHAPE: tl.constexpr,
    CSTRIDES: tl.constexpr,
    TILE: tl.constexpr,
    CDIM_ZERO: tl.constexpr,
):
    pid = tl.program_id(0)
    bidx = tl.program_id(1)
    batch = bidx.to(tl.int64)
    cb = tl.full((), 0, tl.int64)
    for d in tl.static_range(len(BSHAPE) - 1, -1, -1):
        coord = batch % BSHAPE[d]
        batch = batch // BSHAPE[d]
        cb += coord * CSTRIDES[d]
    if CDIM_ZERO:
        # No compressed rows; ATen requires comp[0] == 0 and nnz == 0.
        lo = tl.load(comp + cb).to(tl.int32)
        tl.store(eperr + bidx, (~((lo == 0) & (nnz == 0))).to(tl.int32))
    else:
        rows = pid * TILE + tl.arange(0, TILE)
        m = rows < cdim
        # `other=0` is unreliable on XPU3; neutralise inactive lanes with
        # `tl.where` so they cannot pollute the endpoint verdict or the scatter.
        lo = tl.where(
            m, tl.load(comp + cb + rows * cs, mask=m, other=0).to(tl.int32), 0
        )
        hi = tl.where(
            m, tl.load(comp + cb + (rows + 1) * cs, mask=m, other=0).to(tl.int32), 0
        )
        valid = (lo >= 0) & (hi >= lo) & (hi <= nnz) & (hi - lo <= dim)
        valid &= (rows != 0) | (lo == 0)
        valid &= (rows != cdim - 1) | (hi == nnz)
        bad = m & ~valid
        tl.store(
            eperr + bidx * tl.num_programs(0) + pid, tl.max(bad.to(tl.int32), 0)
        )
        # Mark segment starts.  Inactive lanes carry lo == 0 so they scatter to
        # index 0 (always a start) instead of an out-of-bounds address.
        addr = tl.minimum(tl.maximum(lo, 0), nnz)
        tl.store(flag + bidx * (nnz + 1) + addr, 1)


@triton.jit
def _entries_kernel(
    plain,
    flag,
    enerr,
    dim,
    nnz,
    ps,
    BSHAPE: tl.constexpr,
    PSTRIDES: tl.constexpr,
    TILE: tl.constexpr,
    HAS_PREV: tl.constexpr,
):
    pid = tl.program_id(0)
    bidx = tl.program_id(1)
    batch = bidx.to(tl.int64)
    pb = tl.full((), 0, tl.int64)
    for d in tl.static_range(len(BSHAPE) - 1, -1, -1):
        coord = batch % BSHAPE[d]
        batch = batch // BSHAPE[d]
        pb += coord * PSTRIDES[d]
    lanes = tl.arange(0, TILE)
    idx = pid * TILE + lanes
    m = idx < nnz
    val = tl.load(plain + pb + idx * ps, mask=m, other=0).to(tl.int32)
    bad = m & ((val < 0) | (val >= dim))
    if HAS_PREV:
        # `idx == 0` is never a boundary-with-predecessor; `flag == 1` marks a
        # segment start whose first element has no in-segment predecessor.
        pm = m & (idx > 0)
        prev = tl.load(plain + pb + (idx - 1) * ps, mask=pm, other=0).to(tl.int32)
        start = tl.load(flag + bidx * (nnz + 1) + idx, mask=m, other=1).to(tl.int32)
        bad |= pm & (start == 0) & (prev >= val)
    tl.store(enerr + bidx * tl.num_programs(0) + pid, tl.max(bad.to(tl.int32), 0))



def _check_indices(is_crow, compressed_idx, plain_idx, cdim, dim, nnz):
    if compressed_idx.ndim == 0 or plain_idx.ndim == 0:
        raise IndexError("Dimension specified as -1 but tensor has no dimensions")
    cname, pname, dname = (
        ("crow_indices", "col_indices", "nrows")
        if is_crow
        else ("ccol_indices", "row_indices", "ncols")
    )
    if compressed_idx.shape[-1] != cdim + 1 or cdim < 0:
        raise RuntimeError(
            f"{cname} have wrong shape: {cname}.shape[-1] = {compressed_idx.shape[-1]} "
            f"is not equal to {dname} + 1 = {cdim + 1}"
        )
    if plain_idx.shape[-1] != nnz or nnz < 0:
        raise RuntimeError(
            f"{pname} have wrong shape: {pname}.shape[-1] = {plain_idx.shape[-1]} is not equal to nnz = {nnz}"
        )
    if compressed_idx.dtype != plain_idx.dtype or plain_idx.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise RuntimeError(
            "compressed and plain indices must have the same int32 or int64 dtype"
        )
    if compressed_idx.device != plain_idx.device:
        raise RuntimeError("compressed and plain indices must be on the same device")
    if compressed_idx.shape[:-1] != plain_idx.shape[:-1]:
        raise RuntimeError(
            "compressed and plain indices must have the same batch dimensions"
        )
    # ATen checks that all three dimensions fit the index scalar type.
    limit = torch.iinfo(plain_idx.dtype)
    for name, value in (
        ("plain dimension", dim),
        ("compressed dimension", cdim),
        ("nnz", nnz),
    ):
        if not limit.min <= value <= limit.max:
            raise RuntimeError(f"{limit.bits}-bit integer overflow in {name} = {value}")
    batch_shape = compressed_idx.shape[:-1]
    count = 1
    for extent in batch_shape:
        count *= extent
    if count:
        nrows = 1 if cdim == 0 else (cdim + _ROWS_TILE - 1) // _ROWS_TILE
        nentries = 0 if nnz == 0 else (nnz + _ENTRIES_TILE - 1) // _ENTRIES_TILE
        flag_len = count * (nnz + 1)
        # `torch.zeros` only initialises the start-flag / verdict scratch; the
        # validation itself runs entirely inside the XPU kernels.
        scratch = torch.zeros(
            flag_len + count * (nrows + nentries),
            dtype=torch.int32,
            device=compressed_idx.device,
        )
        flag = scratch[:flag_len]
        err = scratch[flag_len:]
        with torch_device_fn.device(compressed_idx.device):
            _endpoint_kernel[(nrows, count)](
                compressed_idx,
                flag,
                err,
                cdim,
                dim,
                nnz,
                compressed_idx.stride(-1),
                BSHAPE=tuple(batch_shape),
                CSTRIDES=compressed_idx.stride()[:-1],
                TILE=_ROWS_TILE,
                CDIM_ZERO=(cdim == 0),
                num_warps=4,
            )
            if nentries:
                enerr = err.narrow(0, count * nrows, count * nentries)
                _entries_kernel[(nentries, count)](
                    plain_idx,
                    flag,
                    enerr,
                    dim,
                    nnz,
                    plain_idx.stride(-1),
                    BSHAPE=tuple(batch_shape),
                    PSTRIDES=plain_idx.stride()[:-1],
                    TILE=_ENTRIES_TILE,
                    HAS_PREV=(nnz > 1),
                    num_warps=4,
                )
        # Host-side aggregation of the on-device verdict.  `tolist` copies the
        # small verdict buffer (boolean flags, never the validated index data)
        # back to the host; the Python `any` decides whether to raise.
        if any(err.tolist()):
            raise RuntimeError(_ERR_MSG)


def _validate_metadata(
    compressed_indices, plain_indices, values, size, layout, check_pinning
):
    if layout not in (
        torch.sparse_csr,
        torch.sparse_csc,
        torch.sparse_bsr,
        torch.sparse_bsc,
    ):
        raise RuntimeError(
            "layout must be SparseCsr, SparseCsc, SparseBsr or SparseBsc"
        )
    row_major = layout in (torch.sparse_csr, torch.sparse_bsr)
    blocked = layout in (torch.sparse_bsr, torch.sparse_bsc)
    cname, pname = (
        ("crow_indices", "col_indices")
        if row_major
        else ("ccol_indices", "row_indices")
    )
    for tensor, name in (
        (plain_indices, pname),
        (compressed_indices, cname),
        (values, "values"),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a Tensor")
        if tensor.layout != torch.strided:
            raise RuntimeError(f"expected {name} to be a strided tensor")
    batch_ndim = compressed_indices.ndim - 1
    block_ndim = 2 if blocked else 0
    dense_ndim = values.ndim - batch_ndim - block_ndim - 1
    if batch_ndim < 0:
        raise RuntimeError(f"{cname} must have dimensionality >= 1")
    if compressed_indices.ndim != plain_indices.ndim:
        raise RuntimeError(f"{cname} and {pname} dimensionalities must be equal")
    if dense_ndim < 0:
        raise RuntimeError(
            "values must have dimensionality > sum of batch and block dimensionalities"
        )
    if compressed_indices.stride(-1) != 1 or plain_indices.stride(-1) != 1:
        raise RuntimeError("indices must be contiguous per batch")
    size = tuple(operator.index(extent) for extent in size)
    if len(size) != batch_ndim + 2 + dense_ndim:
        raise RuntimeError(
            "tensor dimensionality must be sum of batch, base, and dense dimensionalities"
        )
    batch_shape = size[:batch_ndim]
    if any(
        t.shape[:batch_ndim] != batch_shape
        for t in (compressed_indices, plain_indices, values)
    ):
        raise RuntimeError(
            "all batch dimensions must be equal to tensor batch dimensions"
        )
    br = max(1, values.shape[batch_ndim + 1]) if blocked else 1
    bc = max(1, values.shape[batch_ndim + 2]) if blocked else 1
    rows, cols = size[batch_ndim : batch_ndim + 2]
    if rows < 0 or cols < 0 or rows % br or cols % bc:
        raise RuntimeError(
            "tensor shape must be nonnegative and divisible by the values blocksize"
        )
    cdim, dim = (rows // br, cols // bc) if row_major else (cols // bc, rows // br)
    nnz = values.shape[batch_ndim]
    if compressed_indices.shape[-1] != cdim + 1:
        raise RuntimeError(f"{cname}.shape[-1] must equal the compressed dimension + 1")
    if plain_indices.shape[-1] != nnz:
        raise RuntimeError(f"{pname}.shape[-1] must equal nnz as defined by values")
    if compressed_indices.dtype != plain_indices.dtype or plain_indices.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise RuntimeError("indices must have the same int32 or int64 dtype")
    if (
        compressed_indices.device != plain_indices.device
        or compressed_indices.device != values.device
    ):
        raise RuntimeError("device of indices must match device of values")
    if check_pinning is not False:
        if (
            compressed_indices.is_pinned() != plain_indices.is_pinned()
            or compressed_indices.is_pinned() != values.is_pinned()
        ):
            raise RuntimeError(
                "memory pinning of indices must match memory pinning of values"
            )
    if compressed_indices.is_meta:
        if nnz:
            raise RuntimeError("expected nnz to be 0 for sparse meta tensor")
        return None
    _check_indices(row_major, compressed_indices, plain_indices, cdim, dim, nnz)
    return None


def _validate_sparse_csc_tensor_args(
    ccol_indices, row_indices, values, size, check_pinning=None
):
    logger.debug("GEMS_KUNLUNXIN _VALIDATE_SPARSE_CSC_TENSOR_ARGS")
    return _validate_metadata(
        ccol_indices, row_indices, values, size, torch.sparse_csc, check_pinning
    )

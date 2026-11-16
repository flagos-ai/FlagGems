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

"""Kunlunxin (TritonXPU) specialization of ``aten::unique_dim_consecutive``.

Why this file exists
--------------------
The generic implementation (``src/flag_gems/ops/unique_dim_consecutive.py``)
marks the first row of every consecutive group with
``_unique_dim_consecutive_first_kernel`` and
``BLOCK_SIZE = min(1024, next_power_of_2(row_len))``.  For a 1-column row
(``row_len == 1``: every 1-D input, ``(N, 1)``, and any ``dim`` whose slice has
a single element) that lowers to ``tl.arange(0, 1)`` + a masked load.  A
standalone repro on XPU3 (``harness/perf_ir/repro_unique_dim_consecutive_rowlen1.py``, mirrored in the solution
README) shows that exact kernel is *non-deterministic*: repeated launches on
``[0,0,0,0]`` return ``[1,0,0,0]``, ``[1,1,0,0]``, ``[1,0,1,1]`` ...  The same
kernel with ``BLOCK_SIZE >= 2`` or a plain scalar load is deterministic.
This is the backend's documented "tiles narrower than 64 lanes mis-lower"
hazard.  On HEAD it produces 64/317 accuracy failures in
``tests/test_unique_dim_consecutive.py``.

Implementation notes
--------------------
* Only layout metadata differs from ``unique_dim``: consecutive mode never
  sorts, so the "sorted permutation" is the identity.  Everything here is
  therefore indexed in original row order.
* Every helper buffer is padded to whole tiles and every load/store is
  unmasked; out-of-range lanes are redirected with ``tl.where`` (a masked
  discrete access is *not* honoured on TritonXPU, and ``other=`` is silently
  ignored).  Reductions are always unconditional, never inside a runtime
  ``if`` arm.
* No ATen copy-from primitive anywhere: the movedim materialisation and the
  padded copies are real Triton kernels (``_perm_copy_kernel`` /
  ``_copy_pad_kernel``).
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.libentry import libentry

from .unique_dim import (
    _COMPARE_BLOCK,
    _MIN_TILE,
    _ROW_BLOCK,
    _counts_kernel,
    _inverse,
    _num_warps,
    _pad_len,
    _row_diff_reduce_kernel,
    _row_pad_len,
)

logger = logging.getLogger(__name__)

_COPY_BLOCK = 1024


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


@libentry()
@triton.jit
def _copy_pad_kernel(
    src_ptr: tl.tensor,
    dst_ptr: tl.tensor,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy ``src[:n]`` into a whole-tile destination, unmasked.

    ``dst`` is allocated with ``ceil(n / BLOCK) * BLOCK`` elements, so the tail
    lanes have a real in-range slot to write; their *source* index is redirected
    to 0 instead of relying on a ignored load mask.
    """
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    live = offsets < n
    src = tl.load(src_ptr + tl.where(live, offsets, 0))
    tl.store(dst_ptr + offsets, src)


@libentry()
@triton.jit
def _perm_copy_kernel(
    src_ptr: tl.tensor,
    dst_ptr: tl.tensor,
    meta_ptr: tl.tensor,
    row_len,
    stride_dim,
    n_other,
    total,
    BLOCK_SIZE: tl.constexpr,
):
    """Materialise ``input.movedim(dim, 0)`` into a contiguous buffer.

    ``meta`` holds ``(size, stride)`` pairs for every dimension except ``dim``,
    in original order (so a linear column index decodes row-major, exactly the
    order of ``movedim(dim, 0).reshape(size_dim, -1)``).  ``dst`` is padded to a
    whole tile so every lane writes in range.
    """
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    live = offsets < total
    o = tl.where(live, offsets, 0)
    row = o // row_len
    rem = o - row * row_len
    src_off = row * stride_dim
    # Decode the column row-major over the remaining dims: the *last* remaining
    # dim varies fastest, so walk the meta pairs in reverse.
    for k in range(n_other):
        j = n_other - 1 - k
        size = tl.load(meta_ptr + 2 * j)
        stride = tl.load(meta_ptr + 2 * j + 1)
        coord = rem % size
        rem = rem // size
        src_off += coord * stride
    tl.store(dst_ptr + offsets, tl.load(src_ptr + src_off))


@libentry()
@triton.jit
def _first_single_kernel(
    flat_ptr: tl.tensor,
    is_first_ptr: tl.tensor,
    row_len,
    BLOCK_SIZE: tl.constexpr,
):
    """Single-tile consecutive-diff flag (identity order, unmasked)."""
    row = ext.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    live = offsets < row_len
    safe = tl.where(live, offsets, 0)
    prev_row = tl.where(row == 0, 0, row - 1)
    cur = tl.load(flat_ptr + row * row_len + safe)
    prev = tl.load(flat_ptr + prev_row * row_len + safe)
    neq = (cur != prev) & live
    diff = tl.sum(neq.to(tl.int32), axis=0) != 0
    tl.store(is_first_ptr + row, (row == 0) | diff)


@libentry()
@triton.jit
def _first_chunk_kernel(
    flat_ptr: tl.tensor,
    chunk_diff_ptr: tl.tensor,
    row_len,
    n_chunks,
    BLOCK_SIZE: tl.constexpr,
):
    """Per-(row, column-chunk) consecutive-diff flag (identity order)."""
    row = ext.program_id(0)
    chunk = ext.program_id(1)
    offsets = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    live = offsets < row_len
    safe = tl.where(live, offsets, 0)
    prev_row = tl.where(row == 0, 0, row - 1)
    cur = tl.load(flat_ptr + row * row_len + safe)
    prev = tl.load(flat_ptr + prev_row * row_len + safe)
    neq = (cur != prev) & live
    diff = (tl.sum(neq.to(tl.int32), axis=0) != 0).to(tl.int32)
    out = tl.where(row == 0, (chunk == 0).to(tl.int32), diff)
    tl.store(chunk_diff_ptr + row * n_chunks + chunk, out)


@libentry()
@triton.jit
def _gather_flat_kernel(
    flat_ptr: tl.tensor,
    index_ptr: tl.tensor,
    out_ptr: tl.tensor,
    n,
    row_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """``out[r * row_len + c] = flat[index[r] * row_len + c]`` (padded out).

    Formulated over the flat output so the source address has no ``tl.where``
    lane redirect: a ``where``-redirected *discrete* address mis-lowers on this
    backend (verified: it broadcasts a single element per program for row_len
    2500).  ``row_len`` is ``constexpr`` so the div/mod lower to shifts/mul.
    """
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    live = offsets < n
    o = tl.where(live, offsets, 0)
    row = o // row_len
    col = o - row * row_len
    src_row = tl.load(index_ptr + row)
    val = tl.load(flat_ptr + src_row * row_len + col)
    tl.store(out_ptr + offsets, val)


@libentry()
@triton.jit
def _gather_1d_kernel(
    values_ptr: tl.tensor,
    index_ptr: tl.tensor,
    out_ptr: tl.tensor,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    """``out[i] = values[index[i]]`` for ``i < n`` (padded buffers)."""
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    live = offsets < n
    idx = tl.load(index_ptr + offsets)
    val = tl.load(values_ptr + tl.where(live, idx, 0))
    tl.store(out_ptr + offsets, val)


# ---------------------------------------------------------------------------
# host helpers
# ---------------------------------------------------------------------------


def _copy_pad(src: torch.Tensor, extra: int = 0):
    """Triton copy of ``src`` into a whole-tile padded buffer.

    The destination is ``ceil((n + extra) / BLOCK) * BLOCK + 1`` elements: the
    grid covers all padded lanes (unmasked stores) and the trailing ``+1`` lets
    a consumer do a raw ``ptr[offset + 1]`` load on the final lane.
    """
    n = src.numel()
    pad = triton.cdiv(_pad_len(n) + extra, _COPY_BLOCK) * _COPY_BLOCK
    dst = torch.empty(pad + 1, dtype=src.dtype, device=src.device)
    if n:
        grid = (triton.cdiv(pad, _COPY_BLOCK), 1, 1)
        with torch_device_fn.device(src.device.index):
            _copy_pad_kernel[grid](
                src, dst, n, BLOCK_SIZE=_COPY_BLOCK, num_warps=4
            )
    return dst


def _to_row_padded(src: torch.Tensor, num_rows: int):
    """Triton copy of ``src`` into a scan-tile-safe ``_row_pad_len`` buffer."""
    pad = _row_pad_len(num_rows)
    dst = torch.empty(pad, dtype=src.dtype, device=src.device)
    if num_rows:
        grid = (triton.cdiv(pad, _COPY_BLOCK), 1, 1)
        with torch_device_fn.device(src.device.index):
            _copy_pad_kernel[grid](
                src, dst, num_rows, BLOCK_SIZE=_COPY_BLOCK, num_warps=4
            )
    return dst


def _moved_contiguous(input: torch.Tensor, dim: int) -> torch.Tensor:
    """``input.movedim(dim, 0)`` made contiguous with a Triton copy kernel."""
    view = input.movedim(dim, 0)
    if view.is_contiguous():
        return view
    shape = input.shape
    stride = input.stride()
    others = [d for d in range(len(shape)) if d != dim]
    meta = []
    for d in others:
        meta.extend([shape[d], stride[d]])
    meta_t = torch.tensor(meta, dtype=torch.int64, device=input.device)
    total = input.numel()
    pad = triton.cdiv(total, _COPY_BLOCK) * _COPY_BLOCK
    dst = torch.empty(pad, dtype=input.dtype, device=input.device)
    grid = (triton.cdiv(total, _COPY_BLOCK), 1, 1)
    with torch_device_fn.device(input.device.index):
        _perm_copy_kernel[grid](
            input,
            dst,
            meta_t,
            total // shape[dim],
            stride[dim],
            len(others),
            total,
            BLOCK_SIZE=_COPY_BLOCK,
            num_warps=4,
        )
    return dst[:total].view(list(view.shape))


def _first_mask(flat: torch.Tensor, num_rows: int, row_len: int) -> torch.Tensor:
    """Padded bool mask: True where a row starts a new consecutive group."""
    device = flat.device
    is_first = torch.empty(_row_pad_len(num_rows), dtype=torch.bool, device=device)
    block = min(_COMPARE_BLOCK, max(_MIN_TILE, triton.next_power_of_2(row_len)))
    n_chunks = triton.cdiv(row_len, block)
    with torch_device_fn.device(device.index):
        if n_chunks == 1:
            _first_single_kernel[(num_rows, 1, 1)](
                flat,
                is_first,
                row_len,
                BLOCK_SIZE=block,
                num_warps=_num_warps(block),
            )
        else:
            chunk_diff = torch.empty(
                (num_rows, n_chunks), dtype=torch.int32, device=device
            )
            _first_chunk_kernel[(num_rows, n_chunks, 1)](
                flat,
                chunk_diff,
                row_len,
                n_chunks,
                BLOCK_SIZE=block,
                num_warps=_num_warps(block),
            )
            bc = max(_MIN_TILE, triton.next_power_of_2(n_chunks))
            _row_diff_reduce_kernel[(num_rows, 1, 1)](
                chunk_diff,
                is_first,
                n_chunks,
                BLOCK_CHUNKS=bc,
                num_warps=_num_warps(bc),
            )
    return is_first


def _gather_output(
    moved: torch.Tensor,
    unique_indices_pad: torch.Tensor,
    num_unique: int,
    dim: int,
    input_shape: torch.Size,
):
    device = moved.device
    out_shape = (num_unique,) + tuple(moved.shape[1:])
    if num_unique == 0:
        return torch.empty(out_shape, dtype=moved.dtype, device=device).movedim(
            0, dim
        )
    row_len = moved[0].numel()
    flat = moved.reshape(moved.shape[0], row_len)
    if row_len == 1:
        pad = _pad_len(num_unique)
        out = torch.empty(pad, dtype=moved.dtype, device=device)
        grid = (triton.cdiv(pad, _COPY_BLOCK), 1, 1)
        with torch_device_fn.device(device.index):
            _gather_1d_kernel[grid](
                flat.reshape(-1),
                unique_indices_pad,
                out,
                num_unique,
                BLOCK_SIZE=_COPY_BLOCK,
                num_warps=4,
            )
        return out[:num_unique].reshape(out_shape).movedim(0, dim)

    block = min(_COMPARE_BLOCK, max(_MIN_TILE, triton.next_power_of_2(row_len)))
    n_elems = num_unique * row_len
    pad = triton.cdiv(n_elems, block) * block
    dest = torch.empty(pad, dtype=moved.dtype, device=device)
    with torch_device_fn.device(device.index):
        _gather_flat_kernel[(triton.cdiv(n_elems, block), 1, 1)](
            flat,
            unique_indices_pad,
            dest,
            n_elems,
            row_len=row_len,
            BLOCK_SIZE=block,
            num_warps=4,
        )
    return dest[:n_elems].reshape(out_shape).movedim(0, dim)


# ---------------------------------------------------------------------------
# operator
# ---------------------------------------------------------------------------


def unique_dim_consecutive(
    input: torch.Tensor,
    dim: int,
    return_inverse: bool = False,
    return_counts: bool = False,
):
    """``aten::unique_dim_consecutive`` with the consecutive-only contract."""
    logger.debug("GEMS_KUNLUNXIN UNIQUE_DIM_CONSECUTIVE")

    ndim = input.ndim
    if ndim == 0:
        raise IndexError(
            "Dimension specified as {} but tensor has no dimensions".format(dim)
        )
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-ndim}, {ndim - 1}], but got {dim})"
        )

    device = input.device
    size_dim = input.size(dim)

    # Matches aten: dims with size 0 that are not the selected dim cannot be
    # uniqued ("There are 0 sized dimensions, and they aren't selected").
    if any(s == 0 for i, s in enumerate(input.shape) if i != dim):
        raise RuntimeError(
            "There are 0 sized dimensions, and they aren't selected, "
            "so unique cannot be applied"
        )

    empty_i64 = torch.empty(0, dtype=torch.int64, device=device)
    if size_dim == 0:
        output_shape = tuple(input.shape[:dim]) + (0,) + tuple(input.shape[dim + 1 :])
        return (
            torch.empty(output_shape, dtype=input.dtype, device=device),
            empty_i64,
            torch.empty(0, dtype=torch.int64, device=device),
        )

    moved = _moved_contiguous(input, dim)
    row_len = moved.numel() // size_dim
    flat = moved.reshape(size_dim, row_len)

    is_first = _first_mask(flat, size_dim, row_len)
    first_positions = torch.nonzero(is_first[:size_dim], as_tuple=False).flatten()
    num_unique = first_positions.numel()

    inverse_indices = empty_i64
    counts = torch.empty(0, dtype=torch.int64, device=device)
    if return_inverse:
        # Identity permutation: consecutive mode never sorts.
        identity = _to_row_padded(
            torch.arange(size_dim, dtype=torch.int64, device=device), size_dim
        )
        inverse_indices = _inverse(identity, is_first, size_dim)
    if return_counts:
        pos = _copy_pad(first_positions, extra=1)
        counts_buf = torch.empty(
            _pad_len(num_unique), dtype=torch.int64, device=device
        )
        if num_unique:
            with torch_device_fn.device(device.index):
                _counts_kernel[(counts_buf.numel() // _ROW_BLOCK, 1, 1)](
                    pos,
                    counts_buf,
                    size_dim,
                    num_unique,
                    BLOCK_SIZE=_ROW_BLOCK,
                    num_warps=4,
                )
        counts = counts_buf[:num_unique]

    output = _gather_output(
        moved, _copy_pad(first_positions), num_unique, dim, input.shape
    )
    return output, inverse_indices, counts

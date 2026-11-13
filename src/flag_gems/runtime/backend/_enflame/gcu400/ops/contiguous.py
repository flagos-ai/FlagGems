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

import logging

import torch
import triton
import triton.language as tl

from .copy import copy_

logger = logging.getLogger(__name__)

# gcu400 codegen caps the pointwise grid at 48 CTAs; keep the same budget here.
_GRID_SIZE = 48
# Row segments move at most 8 Ki elements per iteration (offsets are int32
# lanes, so a tile costs ~3 x 32 KiB of on-chip memory plus the payload).
_ROW_BLOCK_CAP = 8192
_STRIDED_BLOCK = 8192
_GATHER_BLOCK = 4096
_MAX_RANK = 8
# Offsets are int32 lanes; keep headroom for the tile arithmetic.
_OFFSET_LIMIT = 2**31 - 1 - _ROW_BLOCK_CAP

_LOADABLE_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.int8,
    torch.uint8,
)


@triton.jit
def _rows_copy_kernel(
    src_ptr,
    dst_ptr,
    n_rows,
    row_len,
    row_stride,
    chunks_per_row,
    BLOCK: tl.constexpr,
    GRID: tl.constexpr,
):
    """Copy equally strided contiguous rows into a flat destination.

    Each iteration moves one contiguous 1-D segment of one row.  Two other
    layouts were measured and rejected on gcu400: the generic pointwise path
    moves every tile through a block-pointer (DTE) transfer, which faults the
    device on large sliced 4-D sources (a ``contiguous`` on a sliced
    (16, 128, 64, 1280) fp16 tensor takes the card down with a SIC halt,
    observed 2026-11-12), and a (rows x cols) affine 2-D tile silently reads
    garbage for the second row when the row stride is large (measured on a
    (2, 128, 64, 1280) fp16 source, 2026-11-12).  Plain 1-D offset
    loads/stores stay on the synchronous port and are masked exactly, so they
    cannot overrun; see also the flat-copy rationale in ops/copy.py.
    """
    pid = tl.program_id(0)
    num_chunks = n_rows * chunks_per_row
    for chunk_id in tl.range(pid, num_chunks, GRID):
        row = chunk_id // chunks_per_row
        col0 = (chunk_id % chunks_per_row) * BLOCK
        offs = col0 + tl.arange(0, BLOCK)
        mask = offs < row_len
        src_off = row * row_stride + offs
        val = tl.load(src_ptr + src_off, mask=mask, other=0)
        dst_off = row * row_len + offs
        tl.store(dst_ptr + dst_off, val, mask=mask)


@triton.jit
def _strided_copy_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    row_stride,
    BLOCK: tl.constexpr,
    GRID: tl.constexpr,
):
    """Flat copy for constant-stride sources (row_len == 1): element p of the
    contiguous destination comes from p * row_stride of the source."""
    pid = tl.program_id(0)
    num_tiles = tl.cdiv(n_elements, BLOCK)
    for tile_id in tl.range(pid, num_tiles, GRID):
        offs = tile_id * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        val = tl.load(src_ptr + offs * row_stride, mask=mask, other=0)
        tl.store(dst_ptr + offs, val, mask=mask)


@triton.jit
def _gather_copy_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    st0,
    st1,
    st2,
    st3,
    st4,
    st5,
    st6,
    st7,
    BLOCK: tl.constexpr,
    GRID: tl.constexpr,
):
    """Flat copy with a per-element multi-index decomposition.

    Handles stride patterns the row plan cannot express (transposed, padded
    or broadcast views).  The modulo is kept in the rem form, which lowers
    to the correct masked-gather path on this backend.
    """
    pid = tl.program_id(0)
    num_tiles = tl.cdiv(n_elements, BLOCK)
    for tile_id in tl.range(pid, num_tiles, GRID):
        offs = tile_id * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        rem = offs
        i7 = rem % s7
        rem = rem // s7
        i6 = rem % s6
        rem = rem // s6
        i5 = rem % s5
        rem = rem // s5
        i4 = rem % s4
        rem = rem // s4
        i3 = rem % s3
        rem = rem // s3
        i2 = rem % s2
        rem = rem // s2
        i1 = rem % s1
        rem = rem // s1
        i0 = rem
        src_off = (
            i0 * st0
            + i1 * st1
            + i2 * st2
            + i3 * st3
            + i4 * st4
            + i5 * st5
            + i6 * st6
            + i7 * st7
        )
        val = tl.load(src_ptr + src_off, mask=mask, other=0)
        tl.store(dst_ptr + offs, val, mask=mask)


def _tail_run(shape, strides):
    """Length of the trailing span laid out contiguously, and the first row dim."""
    ndim = len(shape)
    dim = ndim - 1
    expected = 1
    row_len = 1
    while dim >= 0:
        if shape[dim] == 1:
            dim -= 1
            continue
        if strides[dim] != expected:
            break
        row_len *= shape[dim]
        expected *= shape[dim]
        dim -= 1
    return row_len, dim


def _row_plan(inp):
    """Return (n_rows, row_len, row_stride) when the tensor is equally strided
    contiguous rows, i.e. element (r, c) lives at r * row_stride + c."""
    shape = tuple(inp.shape)
    strides = tuple(inp.stride())
    row_len, split = _tail_run(shape, strides)
    n_rows = 1
    for size in shape[: split + 1]:
        n_rows *= size
    row_dims = [
        (shape[dim], strides[dim]) for dim in range(split + 1) if shape[dim] > 1
    ]
    if row_dims:
        for (outer_size, outer_stride), (inner_size, inner_stride) in zip(
            row_dims, row_dims[1:]
        ):
            if outer_stride != inner_stride * inner_size:
                return None
        row_stride = row_dims[-1][1]
    else:
        row_stride = 0
    return n_rows, row_len, row_stride


def _offsets_fit(inp):
    """Whether every in-kernel element offset stays inside the int32 lane."""
    if inp.numel() >= _OFFSET_LIMIT:
        return False
    highest = 0
    lowest = 0
    for size, stride in zip(inp.shape, inp.stride()):
        extent = (size - 1) * stride
        highest += max(extent, 0)
        lowest += min(extent, 0)
    return highest < _OFFSET_LIMIT and -lowest < _OFFSET_LIMIT


def _launch_rows_copy(src, dst, n_rows, row_len, row_stride):
    block = min(triton.next_power_of_2(row_len), _ROW_BLOCK_CAP)
    chunks_per_row = triton.cdiv(row_len, block)
    num_chunks = n_rows * chunks_per_row
    grid = (min(_GRID_SIZE, num_chunks),)
    _rows_copy_kernel[grid](
        src,
        dst,
        n_rows,
        row_len,
        row_stride,
        chunks_per_row,
        BLOCK=block,
        GRID=_GRID_SIZE,
        num_warps=4,
    )


def _launch_strided_copy(src, dst, n_elements, row_stride):
    num_tiles = triton.cdiv(n_elements, _STRIDED_BLOCK)
    grid = (min(_GRID_SIZE, num_tiles),)
    _strided_copy_kernel[grid](
        src,
        dst,
        n_elements,
        row_stride,
        BLOCK=_STRIDED_BLOCK,
        GRID=_GRID_SIZE,
        num_warps=4,
    )


def _launch_gather_copy(src, dst, inp):
    numel = inp.numel()
    shape = list(inp.shape) + [1] * (_MAX_RANK - inp.dim())
    strides = list(inp.stride()) + [0] * (_MAX_RANK - inp.dim())
    num_tiles = triton.cdiv(numel, _GATHER_BLOCK)
    grid = (min(_GRID_SIZE, num_tiles),)
    _gather_copy_kernel[grid](
        src,
        dst,
        numel,
        *shape,
        *strides,
        BLOCK=_GATHER_BLOCK,
        GRID=_GRID_SIZE,
        num_warps=4,
    )


def _kernel_contiguous(out, inp):
    """Fill the fresh contiguous tensor with an exact-mask copy; False if the
    case has to keep the copy_ path (exotic dtype/layout, int32 overflow)."""
    if inp.layout != torch.strided or out.layout != torch.strided:
        return False
    if inp.dtype != out.dtype or inp.dtype not in _LOADABLE_DTYPES:
        return False
    if inp.dim() > _MAX_RANK or not _offsets_fit(inp):
        return False
    numel = inp.numel()
    if numel == 0:
        return True
    if inp.dtype == torch.bool:
        src = inp.view(torch.uint8)
        dst = out.view(torch.uint8)
    else:
        src = inp
        dst = out
    plan = _row_plan(inp)
    if plan is not None:
        n_rows, row_len, row_stride = plan
        if row_len == 1:
            _launch_strided_copy(src, dst, numel, row_stride)
        else:
            _launch_rows_copy(src, dst, n_rows, row_len, row_stride)
    else:
        _launch_gather_copy(src, dst, inp)
    return True


def contiguous(inp, memory_format=torch.contiguous_format):
    assert memory_format == torch.contiguous_format
    logger.debug("GEMS_ENFLAME CONTIGUOUS")
    if inp.is_contiguous(memory_format=memory_format):
        return inp
    out = torch.empty_like(inp, memory_format=memory_format)
    if _kernel_contiguous(out, inp):
        return out
    return copy_(out, inp)

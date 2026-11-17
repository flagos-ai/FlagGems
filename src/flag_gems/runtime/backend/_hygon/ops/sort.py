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

from flag_gems.ops.argsort import _argsort_pack_key, _argsort_row_offset
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from .argsort import argsort as _stable_argsort

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _stable_sort_tiles(
    inp,
    values,
    indices,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    AXIS_STRIDE: tl.constexpr,
    OUT_STRIDES: tl.constexpr,
    OUT_AXIS_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    INDEX_BITS: tl.constexpr,
    DESCENDING: tl.constexpr,
):
    rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = tl.arange(0, BLOCK)
    base = _argsort_row_offset(rows, SHAPE, STRIDES)
    valid = (rows[:, None] < ROWS) & (cols[None, :] < N)
    x = tl.load(inp + base[:, None] + cols[None, :] * AXIS_STRIDE, valid, other=0)
    col_indices = tl.broadcast_to(cols[None, :], (ROW_BLOCK, BLOCK))
    if x.dtype == tl.int1:
        packed = _argsort_pack_key(x.to(tl.uint8), col_indices, INDEX_BITS, DESCENDING)
    else:
        packed = _argsort_pack_key(x, col_indices, INDEX_BITS, DESCENDING)
    padding = ~tl.full((), 0, packed.dtype)
    packed = tl.where(valid, packed, padding)
    if BLOCK > 1:
        packed = tl.sort(packed, dim=1, descending=False)
    sorted_cols = (packed & ((1 << INDEX_BITS) - 1)).to(tl.int32)
    # Reload the original value to preserve zero signs and NaN payloads.
    sorted_values = tl.load(
        inp + base[:, None] + sorted_cols * AXIS_STRIDE, valid, other=0
    )
    out_base = _argsort_row_offset(rows, SHAPE, OUT_STRIDES)
    offset = out_base[:, None] + cols[None, :] * OUT_AXIS_STRIDE
    tl.store(values + offset, sorted_values, valid)
    tl.store(indices + offset, sorted_cols, valid)


@libentry()
@triton.jit
def _stable_sort_gather(
    inp,
    indices,
    values,
    TOTAL: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    index = tl.load(indices + offsets, mask, other=0)
    remaining = offsets
    source = tl.full((BLOCK,), 0, tl.int64)
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        coord = remaining % SHAPE[axis]
        remaining = remaining // SHAPE[axis]
        if axis == DIM:
            source += index * STRIDES[axis]
        else:
            source += coord * STRIDES[axis]
    x = tl.load(inp + source, mask, other=0)
    tl.store(values + offsets, x, mask)


@libentry()
@triton.jit
def _stable_bool_count(
    inp,
    counts,
    N: tl.constexpr,
    TILES: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    AXIS_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    task = tl.program_id(0)
    row = task // TILES
    tile = task % TILES
    cols = tile * BLOCK + tl.arange(0, BLOCK)
    base = _argsort_row_offset(row, SHAPE, STRIDES)
    x = tl.load(inp + base + cols * AXIS_STRIDE, cols < N, other=False)
    tl.store(counts + task, tl.sum(x.to(tl.int32), 0))


@libentry()
@triton.jit
def _stable_bool_scatter(
    inp,
    counts,
    values,
    indices,
    N: tl.constexpr,
    TILES: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    AXIS_STRIDE: tl.constexpr,
    OUT_STRIDES: tl.constexpr,
    OUT_AXIS_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    TILE_COUNTS: tl.constexpr,
    DESCENDING: tl.constexpr,
):
    task = tl.program_id(0)
    row = task // TILES
    tile = task % TILES
    t = tl.arange(0, TILE_COUNTS)
    count = tl.load(counts + row * TILES + t, t < TILES, other=0)
    before = tl.sum(tl.where(t < tile, count, 0), 0)
    total = tl.sum(count, 0)
    cols = tile * BLOCK + tl.arange(0, BLOCK)
    base = _argsort_row_offset(row, SHAPE, STRIDES)
    x = tl.load(inp + base + cols * AXIS_STRIDE, cols < N, other=False)
    prefix = tl.cumsum(x.to(tl.int32), 0) + before
    if DESCENDING:
        dest = tl.where(x, prefix - 1, total + cols - prefix)
    else:
        dest = tl.where(x, N - total + prefix - 1, cols - prefix)
    out_base = _argsort_row_offset(row, SHAPE, OUT_STRIDES)
    offsets = out_base + dest.to(tl.int64) * OUT_AXIS_STRIDE
    tl.store(values + offsets, x, cols < N)
    tl.store(indices + offsets, cols, cols < N)


def sort(inp, dim=-1, descending=False):
    logger.debug("GEMS_HYGON SORT")
    return sort_stable(inp, stable=False, dim=dim, descending=descending)


def sort_stable(inp, *, stable, dim=-1, descending=False):
    logger.debug("GEMS_HYGON SORT_STABLE")
    rank = inp.ndim
    if dim < -max(rank, 1) or dim >= max(rank, 1):
        raise IndexError("Dimension out of range")
    dim %= max(rank, 1)
    values = torch.empty(inp.shape, dtype=inp.dtype, device=inp.device)
    if inp.numel() == 0:
        return values, torch.empty(inp.shape, dtype=torch.int64, device=inp.device)
    n = inp.shape[dim] if rank else 1
    rows = inp.numel() // n
    if inp.dtype == torch.bool and n > 4096:
        indices = torch.empty(inp.shape, dtype=torch.int64, device=inp.device)
        shape = tuple(s for i, s in enumerate(inp.shape) if i != dim)
        strides = tuple(s for i, s in enumerate(inp.stride()) if i != dim)
        out_strides = tuple(s for i, s in enumerate(values.stride()) if i != dim)
        axis_stride = inp.stride(dim)
        out_axis_stride = values.stride(dim)
        block = 1024
        tiles = triton.cdiv(n, block)
        counts = torch.empty((rows, tiles), dtype=torch.int32, device=inp.device)
        with torch_device_fn.device(inp.device):
            _stable_bool_count[(rows * tiles,)](
                inp, counts, n, tiles, shape, strides, axis_stride, block, num_warps=4
            )
            _stable_bool_scatter[(rows * tiles,)](
                inp,
                counts,
                values,
                indices,
                n,
                tiles,
                shape,
                strides,
                axis_stride,
                out_strides,
                out_axis_stride,
                block,
                triton.next_power_of_2(tiles),
                descending,
                num_warps=4,
            )
    elif n <= 4096 and inp.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.bool,
    ):
        indices = torch.empty(inp.shape, dtype=torch.int64, device=inp.device)
        shape = tuple(s for i, s in enumerate(inp.shape) if i != dim)
        strides = tuple(s for i, s in enumerate(inp.stride()) if i != dim)
        out_strides = tuple(s for i, s in enumerate(values.stride()) if i != dim)
        axis_stride = inp.stride(dim) if rank else 1
        out_axis_stride = values.stride(dim) if rank else 1
        block = triton.next_power_of_2(n)
        row_block = max(1, 128 // block)
        if block <= 256:
            warps = 1
        elif block <= 1024:
            warps = 4
        else:
            warps = 16 if inp.element_size() == 4 else 8
        with torch_device_fn.device(inp.device):
            _stable_sort_tiles[(triton.cdiv(rows, row_block),)](
                inp,
                values,
                indices,
                n,
                rows,
                shape,
                strides,
                axis_stride,
                out_strides,
                out_axis_stride,
                block,
                row_block,
                (n - 1).bit_length(),
                descending,
                num_warps=warps,
            )
    else:
        indices = _stable_argsort(inp, dim=dim, descending=descending)
        with torch_device_fn.device(inp.device):
            _stable_sort_gather[(triton.cdiv(inp.numel(), 512),)](
                inp,
                indices,
                values,
                inp.numel(),
                tuple(inp.shape),
                tuple(inp.stride()),
                dim,
                512,
                num_warps=4,
            )
    return values, indices

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
import math

import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

# The Ascend driver flattens the launch grid before rtKernelLaunch
# (coreDim = grid.x * grid.y * grid.z), so it is the *product* of the axes that
# must stay <= 65535 - capping each axis separately is not enough.
_ASCEND_MAX_CORE_DIM = 65535


def _ascend_launch_grid(grid_x, grid_y=None):
    """Cap a launch grid so Ascend's flattened coreDim stays within its limit.

    A grid whose product already fits is returned unchanged. Otherwise the
    larger axis absorbs the reduction. Every kernel in this module walks its
    axes with ``num_programs(axis=N)`` strides, so a reduced axis only makes
    each program cover more of its dimension.
    """
    grid_x = min(grid_x, _ASCEND_MAX_CORE_DIM)
    if grid_y is None:
        return (grid_x,)
    grid_y = min(grid_y, _ASCEND_MAX_CORE_DIM)
    if grid_x * grid_y > _ASCEND_MAX_CORE_DIM:
        if grid_x >= grid_y:
            grid_x = max(1, _ASCEND_MAX_CORE_DIM // grid_y)
        else:
            grid_y = max(1, _ASCEND_MAX_CORE_DIM // grid_x)
    return (grid_x, grid_y)


@libentry()
@triton.jit(
    # runtime dims; avoid per-shape recompilation
    do_not_specialize=[
        "input_dim",
        "index_len",
        "inner_size",
        "numel",
    ]
)
def index_copy_ascend_flat_kernel(
    out_ptr,
    src_ptr,
    index_ptr,
    input_dim,
    index_len,
    inner_size,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(axis=0)
    worker_count = tle.num_programs(axis=0)

    # Ascend caps the flattened launch grid at _ASCEND_MAX_CORE_DIM, so each
    # program strides over multiple blocks when the grid was capped.
    for block_start in range(pid * BLOCK_SIZE, numel, worker_count * BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        offsets_i64 = offsets.to(tl.int64)

        # [outer, index, inner] decomposition
        inner_offset = offsets_i64 % inner_size
        src_row = offsets_i64 // inner_size
        index_offset = src_row % index_len
        outer_offset = src_row // index_len

        dst_index = tl.load(index_ptr + index_offset, mask=mask, other=0)
        valid_index = (dst_index >= 0) & (dst_index < input_dim)

        value = tl.load(src_ptr + offsets_i64, mask=mask, other=0.0)
        dst_offset = (outer_offset * input_dim + dst_index) * inner_size + inner_offset

        # Triton Ascend does not support the mask argument of tl.device_assert.
        tl.store(out_ptr + dst_offset, value, mask=mask & valid_index)


@libentry()
@triton.jit(
    do_not_specialize=[
        "input_dim",
        "index_len",
        "inner_size",
        "row_count",
    ]
)
def index_copy_ascend_row_kernel(
    out_ptr,
    src_ptr,
    index_ptr,
    input_dim,
    index_len,
    inner_size,
    row_count,
    BLOCK_SIZE: tl.constexpr,
):
    # one program per [outer, index] row and inner tile
    pid_row = tle.program_id(axis=0)
    worker_row = tle.num_programs(axis=0)
    pid_inner = tle.program_id(axis=1)
    worker_inner = tle.num_programs(axis=1)

    # Ascend caps the flattened launch grid (grid.x * grid.y) at
    # _ASCEND_MAX_CORE_DIM, so each program strides over multiple rows and
    # inner tiles when the grid was capped.
    for row_id in range(pid_row, row_count, worker_row):
        row_id_i64 = row_id.to(tl.int64)

        index_offset = row_id_i64 % index_len
        outer_offset = row_id_i64 // index_len

        dst_index = tl.load(index_ptr + index_offset)
        valid_index = (dst_index >= 0) & (dst_index < input_dim)

        for inner_start in range(
            pid_inner * BLOCK_SIZE, inner_size, worker_inner * BLOCK_SIZE
        ):
            inner_offsets = inner_start + tl.arange(0, BLOCK_SIZE)
            inner_offsets_i64 = inner_offsets.to(tl.int64)

            mask = (inner_offsets < inner_size) & valid_index

            src_offsets = row_id_i64 * inner_size + inner_offsets_i64
            dst_offsets = (
                outer_offset * input_dim + dst_index
            ) * inner_size + inner_offsets_i64

            value = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)
            tl.store(out_ptr + dst_offsets, value, mask=mask)


def _select_flat_block_size(numel, inner_size):
    # narrow block keeps more programs in flight for small workloads
    if inner_size == 1 and numel <= 4096:
        return 256
    return 1024


@libentry()
@triton.jit
def index_copy_ascend_strided_kernel(
    out_ptr,
    src_ptr,
    index_ptr,
    numel,
    input_dim,
    SHAPE: tl.constexpr,
    OUT_STRIDES: tl.constexpr,
    SRC_STRIDES: tl.constexpr,
    DIM: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0).to(tl.int64)
    step = tle.num_programs(0).to(tl.int64) * BLOCK_SIZE
    for start in range(pid * BLOCK_SIZE, numel, step):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        remaining = offsets
        src_offset = tl.full((BLOCK_SIZE,), 0, tl.int64)
        dst_offset = tl.full((BLOCK_SIZE,), 0, tl.int64)
        valid_index = tl.full((BLOCK_SIZE,), True, tl.int1)
        for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
            coordinate = remaining % SHAPE[axis]
            remaining = remaining // SHAPE[axis]
            src_offset += coordinate * SRC_STRIDES[axis]
            if axis == DIM:
                if SHAPE[axis] == 1:
                    # A scalar load avoids a masked, repeated-address gather.
                    index = tl.load(index_ptr).to(tl.int64)
                else:
                    index = tl.load(
                        index_ptr + coordinate * INDEX_STRIDE, mask, other=0
                    ).to(tl.int64)
                valid_index = (index >= 0) & (index < input_dim)
                dst_offset += index * OUT_STRIDES[axis]
            else:
                dst_offset += coordinate * OUT_STRIDES[axis]
        value = tl.load(src_ptr + src_offset, mask & valid_index, other=0)
        tl.store(out_ptr + dst_offset, value, mask & valid_index)


def _launch_index_copy(out, dim, index, src):
    index_len = index.numel()
    inner_size = math.prod(out.shape[dim + 1 :])
    outer_size = math.prod(out.shape[:dim])

    if index_len == 0 or inner_size == 0 or outer_size == 0:
        return

    input_dim = out.size(dim)
    numel = outer_size * index_len * inner_size

    with torch_device_fn.device(out.device):
        if not (out.is_contiguous() and src.is_contiguous() and index.is_contiguous()):
            # Iterate in destination storage order for coalesced writes. Tensor
            # pointers already include storage_offset; strides describe the view.
            order = sorted(range(out.ndim), key=lambda axis: -out.stride(axis))
            block_size = 256
            grid = _ascend_launch_grid(triton.cdiv(numel, block_size))
            index_copy_ascend_strided_kernel[grid](
                out,
                src,
                index,
                numel,
                input_dim,
                SHAPE=tuple(src.shape[axis] for axis in order),
                OUT_STRIDES=tuple(out.stride(axis) for axis in order),
                SRC_STRIDES=tuple(src.stride(axis) for axis in order),
                DIM=order.index(dim),
                INDEX_STRIDE=index.stride(0) if index.ndim else 1,
                BLOCK_SIZE=block_size,
            )
            return
        if inner_size <= 4:
            # flat kernel for tiny suffixes
            block_size = _select_flat_block_size(numel, inner_size)
            grid = _ascend_launch_grid(triton.cdiv(numel, block_size))
            index_copy_ascend_flat_kernel[grid](
                out,
                src,
                index,
                input_dim,
                index_len,
                inner_size,
                numel,
                BLOCK_SIZE=block_size,
            )
            return

        # row kernel for wider suffixes
        row_count = outer_size * index_len
        block_size = 256
        grid = _ascend_launch_grid(row_count, triton.cdiv(inner_size, block_size))
        index_copy_ascend_row_kernel[grid](
            out,
            src,
            index,
            input_dim,
            index_len,
            inner_size,
            row_count,
            BLOCK_SIZE=block_size,
        )


def index_copy(inp, dim, index, src):
    logger.debug("GEMS_ASCEND INDEX_COPY")
    dim %= inp.ndim
    out = inp.clone()
    _launch_index_copy(out, dim, index, src)
    return out


def index_copy_(inp, dim, index, src):
    logger.debug("GEMS_ASCEND INDEX_COPY_")
    dim %= inp.ndim
    _launch_index_copy(inp, dim, index, src)
    return inp

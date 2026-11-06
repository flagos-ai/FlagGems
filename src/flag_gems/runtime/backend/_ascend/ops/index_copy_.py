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
from flag_gems.runtime.backend._ascend.tensorview import (
    HAS_TENSORVIEW,
    tensorview_call,
    tile_rows_for_ub,
)
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

if HAS_TENSORVIEW:
    import triton.experimental.tle as tle_exp
    from triton.experimental.tle.language.dsa import tile_load, tile_store

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


def _launch_index_copy(out, dim, index, src):
    index_len = index.numel()
    inner_size = math.prod(out.shape[dim + 1 :])
    outer_size = math.prod(out.shape[:dim])

    if index_len == 0 or inner_size == 0:
        return

    input_dim = out.size(dim)
    numel = outer_size * index_len * inner_size

    with torch_device_fn.device(out.device):
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


def _index_copy_generic(inp, dim, index, src):
    logger.debug("GEMS_ASCEND INDEX_COPY")
    dim %= inp.ndim
    out = inp.clone()
    _launch_index_copy(out, dim, index, src)
    return out


def _index_copy__generic(inp, dim, index, src):
    logger.debug("GEMS_ASCEND INDEX_COPY_")
    dim %= inp.ndim
    _launch_index_copy(inp, dim, index, src)
    return inp


# ---------------------------------------------------------------------------
# TensorView fast path
#
# Along dim 0 the index names whole contiguous rows, so the source rows arrive
# through a PartitionView and go straight out through a GatherScatterView whose
# sparse dimension takes the index. One block DMA per tile instead of the
# generic kernel's strided element-wise copy.
#
# Unlike index_add this is a pure write, so repeated indices are safe: whichever
# tile writes last wins, which is the semantics torch.index_copy_ specifies for
# duplicates anyway.
# ---------------------------------------------------------------------------
TV_BLOCK_T = 32

if HAS_TENSORVIEW:

    @triton.jit
    def index_copy_tv_kernel(
        dst,
        src,
        index,
        num_indices,
        num_rows,
        WIDTH: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_WIDTH: tl.constexpr,
    ):
        tile = tl.program_id(0)
        offsets = tile * BLOCK_T + tl.arange(0, BLOCK_T)
        rows = tl.load(index + offsets, mask=offsets < num_indices, other=-1).to(
            tl.int32
        )

        src_view = tle_exp.make_partition_view(
            src,
            shape=[num_indices, WIDTH],
            strides=[WIDTH, 1],
            tile=[BLOCK_T, BLOCK_WIDTH],
        )
        values = tile_load(
            src_view, index=(tile, 0), dst_space=tle_exp.language.dsa.ascend.UB
        )

        dst_view = tle_exp.make_gather_scatter_view(
            dst,
            shape=[num_rows, WIDTH],
            strides=[WIDTH, 1],
            tile=[BLOCK_T, BLOCK_WIDTH],
            sparse_dim=[0],
        )
        tile_store(
            dst_view,
            value=values,
            index=(rows, 0),
            src_space=tle_exp.language.dsa.ascend.UB,
        )

    def _launch_index_copy_tv(out, index, src, block_t=None):
        src = src.contiguous()
        num_rows = out.shape[0]
        width = math.prod(out.shape[1:]) if out.dim() > 1 else 1
        if block_t is None:
            # One tile of values is live between the load and the store.
            block_t = tile_rows_for_ub(
                triton.next_power_of_2(width) * out.element_size(), 1, TV_BLOCK_T
            )
        num_indices = index.numel()
        grid = (triton.cdiv(num_indices, block_t),)
        with torch_device_fn.device(out.device):
            index_copy_tv_kernel[grid](
                out,
                src,
                index,
                num_indices,
                num_rows,
                WIDTH=width,
                BLOCK_T=block_t,
                BLOCK_WIDTH=triton.next_power_of_2(width),
            )
        return out

    def _index_copy_tensorview(inp, dim, index, src):
        logger.debug("GEMS_ASCEND INDEX_COPY(TENSORVIEW)")
        out = inp.contiguous().clone()
        return _launch_index_copy_tv(out, index.contiguous(), src)

    def _index_copy__tensorview(inp, dim, index, src):
        logger.debug("GEMS_ASCEND INDEX_COPY_(TENSORVIEW)")
        if inp.is_contiguous():
            return _launch_index_copy_tv(inp, index.contiguous(), src)
        work = inp.contiguous()
        _launch_index_copy_tv(work, index.contiguous(), src)
        inp.copy_(work)
        return inp

    def _tv_eligible(inp, dim, index, src):
        if inp.numel() == 0 or index.numel() == 0:
            return False
        if dim % max(inp.ndim, 1) != 0:
            return False
        if src.dtype != inp.dtype:
            return False
        return index.numel() == src.shape[0]

    def index_copy(inp, dim, index, src):
        if not _tv_eligible(inp, dim, index, src):
            return _index_copy_generic(inp, dim, index, src)
        return tensorview_call(
            "INDEX_COPY",
            _index_copy_tensorview,
            _index_copy_generic,
            inp,
            dim,
            index,
            src,
        )

    def index_copy_(inp, dim, index, src):
        if not _tv_eligible(inp, dim, index, src):
            return _index_copy__generic(inp, dim, index, src)
        return tensorview_call(
            "INDEX_COPY_",
            _index_copy__tensorview,
            _index_copy__generic,
            inp,
            dim,
            index,
            src,
        )

else:
    index_copy = _index_copy_generic
    index_copy_ = _index_copy__generic

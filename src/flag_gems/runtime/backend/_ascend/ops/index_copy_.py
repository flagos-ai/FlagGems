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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._ascend.tensorview import (
    HAS_TENSORVIEW,
    tensorview_call,
    tile_fits_ub,
    tile_rows_for_ub,
)
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

if HAS_TENSORVIEW:
    import triton.experimental.tle as tle_exp
    from triton.experimental.tle import tile_load, tile_store

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


@libentry()
@triton.jit
def _index_copy_strided_row_kernel(
    out,
    src,
    indices,
    rows,
    columns,
    input_dim,
    ROW_SHAPE: tl.constexpr,
    OUT_STRIDES: tl.constexpr,
    SRC_STRIDES: tl.constexpr,
    DIM: tl.constexpr,
    SRC_INNER_STRIDE: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    column_tiles = tl.cdiv(columns, BLOCK)
    for tile in range(tle.program_id(0), rows * column_tiles, tl.num_programs(0)):
        row = tile // column_tiles
        src_base = tl.full((), 0, tl.int64)
        dst_base = tl.full((), 0, tl.int64)
        valid = tl.full((), True, tl.int1)
        for axis in tl.static_range(len(ROW_SHAPE) - 1, -1, -1):
            coordinate = row % ROW_SHAPE[axis]
            row = row // ROW_SHAPE[axis]
            src_base += coordinate * SRC_STRIDES[axis]
            if axis == DIM:
                index = tl.load(indices + coordinate * INDEX_STRIDE).to(tl.int64)
                valid = (index >= 0) & (index < input_dim)
                dst_base += tl.where(valid, index, 0) * OUT_STRIDES[axis]
            else:
                dst_base += coordinate * OUT_STRIDES[axis]
        start = ((tile % column_tiles) * BLOCK).to(tl.int32)
        cols = start + tl.arange(0, BLOCK)
        mask = (cols < columns) & valid
        # Decode row/index scalars once. Preserve 64-bit row bases while keeping
        # the contiguous column traversal separate from multidimensional offsets.
        values = tl.load((src + src_base) + cols * SRC_INNER_STRIDE, mask, other=0)
        tl.store((out + dst_base) + cols, values, mask)


def _strided_row_layout(out, dim, index, src):
    if not (
        out.dtype == src.dtype
        and out.numel() > 0
        and out.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and out.device == src.device == index.device
        and index.ndim == 1
        and index.dtype in (torch.int32, torch.int64)
        and out.ndim == src.ndim
        and src.size(dim) == index.numel()
        and all(out.size(i) == src.size(i) for i in range(out.ndim) if i != dim)
        and all(stride >= 0 for stride in src.stride())
        and out.untyped_storage().data_ptr() != src.untyped_storage().data_ptr()
    ):
        return None
    # A dense permutation has no overlapping destination lanes. Other views keep
    # the general strided implementation, including slices with storage holes.
    expected = 1
    inner = None
    for axis in sorted(range(out.ndim), key=lambda i: out.stride(i)):
        if out.size(axis) > 1:
            if out.stride(axis) != expected:
                return None
            if expected == 1:
                inner = axis
            expected *= out.size(axis)
    if inner is None or inner == dim or src.size(inner) < 128:
        return None
    if src.stride(inner) <= 0:
        return None
    # Include padded column lanes in the address-width guard.
    for tensor in (src, out):
        span = sum(
            (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
        )
        if span + 256 * tensor.stride(inner) >= 2**31:
            return None
    order = sorted(
        (axis for axis in range(out.ndim) if axis != inner),
        key=lambda axis: -out.stride(axis),
    )
    return inner, order


@triton.jit
def _index_copy_check_bounds_kernel(
    index, valid, N, SIZE: tl.constexpr, DEVICE_CHECK: tl.constexpr, BLOCK: tl.constexpr
):
    good = tl.full((), True, tl.int1)
    for tile in range(tl.program_id(0), tl.cdiv(N, BLOCK), tl.num_programs(0)):
        offsets = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        raw = tl.load(index + offsets, offsets < N, other=0).to(tl.int64)
        good &= (
            tl.min(((offsets >= N) | ((raw >= 0) & (raw < SIZE))).to(tl.int32), 0) != 0
        )
    if DEVICE_CHECK:
        if not good:
            tl.device_assert(False, "index out of range")
    else:
        tl.store(valid + tl.program_id(0), good)


def _check_index_copy_bounds(index, size):
    if index.numel() == 0:
        return
    index = index.contiguous()
    block = min(1024, triton.next_power_of_2(index.numel()))
    programs = min(48, triton.cdiv(index.numel(), block))
    capturing = torch_device_fn.is_current_stream_capturing()
    valid = (
        None
        if capturing
        else torch.empty(programs, device=index.device, dtype=torch.int32)
    )
    _index_copy_check_bounds_kernel[(programs,)](
        index, valid, index.numel(), size, capturing, block, debug=capturing
    )
    # Capture cannot read a host scalar. Enable the device assertion explicitly
    # on that path; eager calls keep the normal Python IndexError contract.
    if not capturing and not all(valid.cpu().tolist()):
        raise IndexError(f"index out of range for dimension with size {size}")


def _launch_index_copy(out, dim, index, src):
    index_len = index.numel()
    inner_size = math.prod(out.shape[dim + 1 :])
    outer_size = math.prod(out.shape[:dim])

    if index_len == 0:
        return

    input_dim = out.size(dim)
    numel = outer_size * index_len * inner_size

    with torch_device_fn.device(out.device):
        _check_index_copy_bounds(index, input_dim)
        if inner_size == 0 or outer_size == 0:
            return
        if not (out.is_contiguous() and src.is_contiguous() and index.is_contiguous()):
            layout = _strided_row_layout(out, dim, index, src)
            if layout is not None:
                inner, row_order = layout
                rows = math.prod(src.size(axis) for axis in row_order)
                columns = src.size(inner)
                grid = _ascend_launch_grid(rows * triton.cdiv(columns, 256))
                _index_copy_strided_row_kernel[grid](
                    out,
                    src,
                    index,
                    rows,
                    columns,
                    input_dim,
                    ROW_SHAPE=tuple(src.size(axis) for axis in row_order),
                    OUT_STRIDES=tuple(out.stride(axis) for axis in row_order),
                    SRC_STRIDES=tuple(src.stride(axis) for axis in row_order),
                    DIM=row_order.index(dim),
                    SRC_INNER_STRIDE=src.stride(inner),
                    INDEX_STRIDE=index.stride(0),
                    BLOCK=256,
                )
                return
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
            bounds_check=True,
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
        if index.numel() != src.shape[0]:
            return False
        # The trailing dimensions collapse into one row, so a wide tail makes a
        # row too wide to stage and UB overflow is a compile error. Same
        # row_bytes and live_tiles the launch uses.
        width = math.prod(inp.shape[1:]) if inp.dim() > 1 else 1
        return tile_fits_ub(triton.next_power_of_2(width) * inp.element_size(), 1)

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

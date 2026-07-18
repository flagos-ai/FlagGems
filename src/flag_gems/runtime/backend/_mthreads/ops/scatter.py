import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.scatter import scatter as _generic_scatter
from flag_gems.ops.scatter import scatter_ as _generic_scatter_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.shape_utils import MemOverlap, has_internal_overlapping

logger = logging.getLogger(__name__)

_ATOMIC_BLOCK = 2048
_ATOMIC_LOOP = 1
_SMALL_OUTPUT_WIDTH = 64
_SMALL_INDEX_WIDTH = 128
_ATOMIC_MIN_OUTPUT_WIDTH = 1024
_ATOMIC_MAX_OUTPUT_WIDTH = 8192


@libentry()
@triton.jit
def _scatter_add_2d_kernel(
    src,
    index,
    out,
    rows,
    index_width,
    out_stride_0,
    out_stride_1,
    index_stride_0,
    index_stride_1,
    src_stride_0,
    src_stride_1,
    USE_INT64: tl.constexpr,
    BLOCK: tl.constexpr,
    LOOP: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK * LOOP + tl.arange(0, BLOCK)
    for loop_idx in tl.static_range(0, LOOP):
        current_columns = columns + loop_idx * BLOCK
        current_row = row
        if USE_INT64:
            current_row = current_row.to(tl.int64)
            current_columns = current_columns.to(tl.int64)
        mask = (current_row < rows) & (current_columns < index_width)
        src_values = tl.load(
            src + current_row * src_stride_0 + current_columns * src_stride_1,
            mask=mask,
            other=0.0,
        )
        target_columns = tl.load(
            index + current_row * index_stride_0 + current_columns * index_stride_1,
            mask=mask,
            other=0,
        )
        if not USE_INT64:
            target_columns = target_columns.to(tl.int32)
        out_offsets = current_row * out_stride_0 + target_columns * out_stride_1
        tl.atomic_add(out + out_offsets, src_values, mask=mask, sem="relaxed")


@libentry()
@triton.jit
def _scatter_add_2d_small_kernel(
    inp,
    src,
    index,
    out,
    out_rows,
    index_rows,
    inp_stride_0,
    inp_stride_1,
    out_stride_0,
    out_stride_1,
    index_stride_0,
    index_stride_1,
    src_stride_0,
    src_stride_1,
    OUT_WIDTH: tl.constexpr,
    INDEX_WIDTH: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    BLOCK_INDEX: tl.constexpr,
):
    row = tl.program_id(0)
    out_columns = tl.arange(0, BLOCK_OUT)
    source_columns = tl.arange(0, BLOCK_INDEX)
    source_mask = source_columns < INDEX_WIDTH
    target_columns = tl.load(
        index + row * index_stride_0 + source_columns * index_stride_1,
        mask=(row < index_rows) & source_mask,
        other=-1,
    )
    source_values = tl.load(
        src + row * src_stride_0 + source_columns * src_stride_1,
        mask=(row < index_rows) & source_mask,
        other=0.0,
    ).to(tl.float32)
    matches = out_columns[:, None] == target_columns[None, :]
    updates = tl.sum(
        tl.where(matches & source_mask[None, :], source_values[None, :], 0.0),
        axis=1,
    )
    out_mask = (row < out_rows) & (out_columns < OUT_WIDTH)
    inp_offsets = row * inp_stride_0 + out_columns * inp_stride_1
    out_offsets = row * out_stride_0 + out_columns * out_stride_1
    initial = tl.load(inp + inp_offsets, mask=out_mask, other=0.0).to(tl.float32)
    tl.store(out + out_offsets, initial + updates, mask=out_mask)


def _can_use_small_path(inp, index):
    return inp.shape[1] <= _SMALL_OUTPUT_WIDTH and index.shape[1] <= _SMALL_INDEX_WIDTH


def _can_use_fast_path(inp, dim, index, src, reduce):
    if reduce != "add" or inp.dtype is not torch.float32:
        return False
    if inp.ndim != 2 or index.ndim != 2 or src.ndim != 2:
        return False
    if dim % inp.ndim != 1:
        return False
    if index.shape[0] > inp.shape[0]:
        return False
    if src.shape[0] < index.shape[0] or src.shape[1] < index.shape[1]:
        return False
    if has_internal_overlapping(inp) == MemOverlap.Yes:
        return False
    small_path = _can_use_small_path(inp, index)
    atomic_path = _ATOMIC_MIN_OUTPUT_WIDTH <= inp.shape[1] <= _ATOMIC_MAX_OUTPUT_WIDTH
    return small_path or atomic_path


def _scatter_add_2d(inp, index, src, out):
    index_rows, index_width = index.shape
    if index.numel() == 0:
        return out

    src_view = src.as_strided(index.shape, src.stride())
    with torch_device_fn.device(inp.device):
        if inp.shape[1] <= _SMALL_OUTPUT_WIDTH and index_width <= _SMALL_INDEX_WIDTH:
            block_out = triton.next_power_of_2(inp.shape[1])
            block_index = triton.next_power_of_2(index_width)
            _scatter_add_2d_small_kernel[(inp.shape[0],)](
                inp,
                src_view,
                index,
                out,
                inp.shape[0],
                index_rows,
                *inp.stride(),
                *out.stride(),
                *index.stride(),
                *src_view.stride(),
                inp.shape[1],
                index_width,
                block_out,
                block_index,
                num_warps=4,
                num_stages=1,
            )
        else:
            use_int64 = max(out.numel(), index.numel(), src.numel()) >= 2**31
            grid = (
                index_rows,
                triton.cdiv(index_width, _ATOMIC_BLOCK * _ATOMIC_LOOP),
            )
            _scatter_add_2d_kernel[grid](
                src_view,
                index,
                out,
                index_rows,
                index_width,
                *out.stride(),
                *index.stride(),
                *src_view.stride(),
                use_int64,
                _ATOMIC_BLOCK,
                _ATOMIC_LOOP,
                num_warps=4,
                num_stages=1,
            )
    return out


def scatter(inp, dim, index, src, reduce=None):
    logger.debug("GEMS_MTHREADS SCATTER")
    if not _can_use_fast_path(inp, dim, index, src, reduce):
        return _generic_scatter(inp, dim, index, src, reduce)
    if index.numel() > 0 and _can_use_small_path(inp, index):
        out = torch.empty_like(inp)
    else:
        out = inp.clone()
    return _scatter_add_2d(inp, index, src, out)


def scatter_(inp, dim, index, src, reduce=None):
    logger.debug("GEMS_MTHREADS SCATTER_")
    if not _can_use_fast_path(inp, dim, index, src, reduce):
        return _generic_scatter_(inp, dim, index, src, reduce)
    _scatter_add_2d(inp, index, src, inp)
    return inp

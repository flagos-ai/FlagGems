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

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)
_SMALL_OUTPUT_WIDTH = 64
_SMALL_INDEX_WIDTH = 128


@triton.jit
def _reduce_mul(left, right):
    return left * right


@libentry()
@triton.jit
def _scatter_reduce_2d_small_kernel(
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
    IS_MULTIPLY: tl.constexpr,
):
    row = tl.program_id(0)
    out_columns = tl.arange(0, BLOCK_OUT)
    source_columns = tl.arange(0, BLOCK_INDEX)
    source_mask = source_columns < INDEX_WIDTH
    row_mask = row < index_rows
    target_columns = tl.load(
        index + row * index_stride_0 + source_columns * index_stride_1,
        mask=row_mask & source_mask,
        other=-1,
    )
    source_values = tl.load(
        src + row * src_stride_0 + source_columns * src_stride_1,
        mask=row_mask & source_mask,
        other=0.0,
    ).to(tl.float32)
    matches = (out_columns[:, None] == target_columns[None, :]) & source_mask[None, :]
    if IS_MULTIPLY:
        updates = tl.reduce(
            tl.where(matches, source_values[None, :], 1.0),
            axis=1,
            combine_fn=_reduce_mul,
        )
    else:
        updates = tl.sum(tl.where(matches, source_values[None, :], 0.0), axis=1)

    out_mask = (row < out_rows) & (out_columns < OUT_WIDTH)
    initial = tl.load(
        inp + row * inp_stride_0 + out_columns * inp_stride_1,
        mask=out_mask,
        other=0.0,
    ).to(tl.float32)
    result = initial * updates if IS_MULTIPLY else initial + updates
    tl.store(
        out + row * out_stride_0 + out_columns * out_stride_1,
        result,
        mask=out_mask,
    )


def _can_use_small_fp16_path(inp, dim, index, src, reduce):
    if inp.dtype is not torch.float16 or reduce not in ("add", "multiply"):
        return False
    if inp.ndim != 2 or index.ndim != 2 or src.ndim != 2:
        return False
    if dim % inp.ndim != 1 or index.shape[0] > inp.shape[0]:
        return False
    if src.shape[0] < index.shape[0] or src.shape[1] < index.shape[1]:
        return False
    if not (0 < inp.shape[1] <= _SMALL_OUTPUT_WIDTH):
        return False
    if not (0 < index.shape[1] <= _SMALL_INDEX_WIDTH):
        return False
    return has_internal_overlapping(inp) != MemOverlap.Yes


def _scatter_reduce_2d_small(inp, index, src, reduce, out):
    src_view = src.as_strided(index.shape, src.stride())
    block_out = triton.next_power_of_2(inp.shape[1])
    block_index = triton.next_power_of_2(index.shape[1])
    with torch_device_fn.device(inp.device):
        _scatter_reduce_2d_small_kernel[(inp.shape[0],)](
            inp,
            src_view,
            index,
            out,
            inp.shape[0],
            index.shape[0],
            *inp.stride(),
            *out.stride(),
            *index.stride(),
            *src_view.stride(),
            inp.shape[1],
            index.shape[1],
            block_out,
            block_index,
            reduce == "multiply",
            num_warps=4,
            num_stages=1,
        )
    return out


def _should_use_musa_kernel(inp, reduce):
    if reduce == "multiply" or (reduce == "add" and inp.dtype is torch.float32):
        return True
    return (
        reduce == "add"
        and inp.dtype is torch.float16
        and inp.ndim == 2
        and inp.shape[1] > 256
    )


def scatter(inp, dim, index, src, reduce=None):
    logger.debug("GEMS_MTHREADS SCATTER")
    if _can_use_small_fp16_path(inp, dim, index, src, reduce):
        return _scatter_reduce_2d_small(inp, index, src, reduce, torch.empty_like(inp))
    if _should_use_musa_kernel(inp, reduce):
        return torch.ops.aten.scatter.reduce.redispatch(
            _FALLBACK_KEYSET, inp, dim, index, src, reduce=reduce
        )
    return _generic_scatter(inp, dim, index, src, reduce)


def scatter_(inp, dim, index, src, reduce=None):
    logger.debug("GEMS_MTHREADS SCATTER_")
    if _can_use_small_fp16_path(inp, dim, index, src, reduce):
        return _scatter_reduce_2d_small(inp, index, src, reduce, inp)
    if _should_use_musa_kernel(inp, reduce):
        return torch.ops.aten.scatter_.reduce.redispatch(
            _FALLBACK_KEYSET, inp, dim, index, src, reduce=reduce
        )
    return _generic_scatter_(inp, dim, index, src, reduce)

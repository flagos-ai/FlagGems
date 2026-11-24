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
"""GCU400 ``torch.diagonal`` backward.

``grad_input`` is an all-zero tensor whose diagonal slice along
``(dim1, dim2)`` carries ``grad_output``.  Producing it needs only two
streaming Triton kernels, no native aten compute:

1. ``_zero_fill_kernel``      writes zeros over the whole ``grad_input``;
2. ``_diag_scatter_kernel``   copies the (usually tiny) ``grad_output`` block
   into the strided diagonal positions.

Doing the zero fill and the diagonal scatter ourselves removes the previous
``torch.zeros`` + aten strided ``copy_`` pair, which cost ~5x the traffic of
the single write pass below and added a fixed host-side overhead per call.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# 48 CTAs is the scheduling granularity used by the other GCU400 ops in this
# backend; the grid-stride loop below lets each CTA walk several tiles.
_NUM_CTAS = 48
# Element counts per program for the zero fill. 64KB per program is where the
# streaming store saturates with 48 CTAs (128KB regresses for fp32): 32768
# elements for 2-byte dtypes, 16384 for 4-byte ones; below 64K elements the
# whole tensor fits in one tile and a 32KB tile keeps the launch cheap.
_ZERO_BLOCK_SMALL = 4096
_ZERO_BLOCK_4B = 8192
_ZERO_BLOCK_4B_WIDE = 16384
_ZERO_BLOCK_2B = 32768
_SMALL_NUMEL = 1 << 16
_WIDE_NUMEL = 1 << 22
# The diagonal block is tiny compared with grad_input; one 1024-element tile
# per program is enough to fill the device with a single wave.
_SCATTER_BLOCK = 1024


@triton.jit
def _zero_fill_kernel(out_ptr, numel, BLOCK: tl.constexpr, WIDE: tl.constexpr):
    step = tl.num_programs(0) * BLOCK
    for base in range(tl.program_id(0) * BLOCK, numel, step):
        offsets = base + tl.arange(0, BLOCK)
        if WIDE:
            offsets = offsets.to(tl.int64)
        mask = offsets < numel
        tl.store(
            out_ptr + offsets, tl.zeros((BLOCK,), dtype=out_ptr.dtype.element_ty), mask
        )


@triton.jit
def _diag_scatter_tiled_kernel(
    src_ptr,
    dst_ptr,
    base_offset,
    num_rows,
    num_cols,
    SRC_ROW: tl.constexpr,
    SRC_COL: tl.constexpr,
    DST_ROW: tl.constexpr,
    DST_COL: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    WIDE: tl.constexpr,
):
    """2D tiled copy of the diagonal block into ``grad_input``.

    A rank-2 diagonal view has one axis that is contiguous in the incoming
    gradient and (usually) the other axis contiguous in the destination, so a
    linear enumeration always pays an uncoalesced access on one side.  Walking
    a ``BLOCK_R x BLOCK_C`` tile keeps the loads coalesced along the
    source-contiguous axis and the stores coalesced along the
    destination-contiguous axis, which measured ~6x faster than the linear
    scatter at the same element count.
    """
    pid = tl.program_id(0)
    num_col_tiles = tl.cdiv(num_cols, BLOCK_C)
    rows = (pid // num_col_tiles) * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = (pid % num_col_tiles) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (rows[:, None] < num_rows) & (cols[None, :] < num_cols)
    src_offsets = rows[:, None] * SRC_ROW + cols[None, :] * SRC_COL
    dst_offsets = base_offset + rows[:, None] * DST_ROW + cols[None, :] * DST_COL
    if WIDE:
        src_offsets = src_offsets.to(tl.int64)
        dst_offsets = dst_offsets.to(tl.int64)
    values = tl.load(src_ptr + src_offsets, mask=mask)
    tl.store(dst_ptr + dst_offsets, values, mask=mask)


@triton.jit
def _diag_scatter_kernel(
    src_ptr,
    dst_ptr,
    base_offset,
    num_elem,
    DECODE_STRIDES: tl.constexpr,
    SRC_STRIDES: tl.constexpr,
    DST_STRIDES: tl.constexpr,
    BLOCK: tl.constexpr,
    WIDE: tl.constexpr,
):
    """Scatter a row-major linear block into a strided diagonal view.

    ``DECODE_STRIDES`` are the row-major strides of the diagonal shape and are
    only used to turn the linear id back into a multi-index; ``SRC_STRIDES`` and
    ``DST_STRIDES`` are the real (possibly strided) layout of the incoming
    gradient and of the destination diagonal view.  Keeping the two apart is
    required because ``grad_output`` inherits the strides of the forward
    diagonal view (its last axis is strided by 1 *before* the diagonal axis).
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < num_elem
    if WIDE:
        offsets = offsets.to(tl.int64)
    src_offsets = offsets * 0
    dst_offsets = offsets * 0 + base_offset
    remainder = offsets
    for dim in tl.static_range(0, len(DECODE_STRIDES)):
        index = remainder // DECODE_STRIDES[dim]
        remainder = remainder - index * DECODE_STRIDES[dim]
        src_offsets += index * SRC_STRIDES[dim]
        dst_offsets += index * DST_STRIDES[dim]
    values = tl.load(src_ptr + src_offsets, mask=mask)
    tl.store(dst_ptr + dst_offsets, values, mask=mask)


def _contiguous_strides(shape):
    strides = []
    accumulator = 1
    for size in reversed(shape):
        strides.append(accumulator)
        accumulator *= size
    return tuple(reversed(strides))


def _diagonal_view_meta(input_sizes, offset, dim1, dim2):
    """Metadata of ``grad_input.diagonal(offset, dim1, dim2)`` for a contiguous
    ``grad_input``: (shape, strides, storage offset) of that view.

    torch.diagonal keeps the untouched dims in their original order and appends
    the new diagonal axis at the very end, which is what we reproduce here
    instead of materialising the view through aten.
    """
    rank = len(input_sizes)
    dim1 = dim1 if dim1 >= 0 else dim1 + rank
    dim2 = dim2 if dim2 >= 0 else dim2 + rank
    input_strides = _contiguous_strides(input_sizes)
    size1, size2 = input_sizes[dim1], input_sizes[dim2]
    if offset >= 0:
        length = max(0, min(size1, size2 - offset))
    else:
        length = max(0, min(size1 + offset, size2))
    base_offset = (
        max(0, offset) * input_strides[dim2] + max(0, -offset) * input_strides[dim1]
    )
    kept_dims = [dim for dim in range(rank) if dim not in (dim1, dim2)]
    view_shape = tuple(input_sizes[dim] for dim in kept_dims) + (length,)
    view_strides = tuple(input_strides[dim] for dim in kept_dims) + (
        input_strides[dim1] + input_strides[dim2],
    )
    return view_shape, view_strides, base_offset


def _zero_fill_block(numel, element_size):
    if numel <= _SMALL_NUMEL:
        return _ZERO_BLOCK_SMALL
    if element_size <= 2:
        return _ZERO_BLOCK_2B
    # 4-byte dtypes need the wider tile only once the grid-stride loop has
    # enough iterations to amortise it (>= 4M elements).
    return _ZERO_BLOCK_4B_WIDE if numel >= _WIDE_NUMEL else _ZERO_BLOCK_4B


def _tiled_scatter_shape(num_rows, num_cols, element_size):
    """Pick a ``BLOCK_R x BLOCK_C`` tile whose register budget stays near 32KB.

    ``BLOCK_C`` follows the source-contiguous axis so it takes the largest
    power of two that still fits the column count; the row extent fills the
    rest of the budget.  Falls back to ``(1, BLOCK_C)`` for very small views.
    """
    elements_per_tile = max(1, (32 * 1024) // element_size)
    block_c = min(1024, triton.next_power_of_2(max(1, num_cols)))
    block_r = max(1, elements_per_tile // block_c)
    if block_r > triton.next_power_of_2(max(1, num_rows)):
        block_r = triton.next_power_of_2(max(1, num_rows))
    return block_r, block_c


def _launch_tiled_scatter(
    grad_output, grad_input, diag_shape, diag_strides, base_offset, wide
):
    """Use the tiled transpose scatter for rank-2 diagonal views.

    Returns ``True`` when the tiled kernel was launched.  Only the rank-2
    layout whose unit-stride destination axis is the outer one takes this
    path: the generic linear scatter walks the inner axis fastest, so that is
    exactly the layout where every store is strided.  Rank 1/3+ views and
    already store-coalesced layouts keep the linear kernel.
    """
    if len(diag_shape) != 2:
        return False
    if diag_strides[0] != 1 or diag_strides[1] == 1:
        return False
    src_strides = tuple(grad_output.stride())
    row_axis = 0
    col_axis = 1
    num_rows = diag_shape[row_axis]
    num_cols = diag_shape[col_axis]
    if num_rows == 0 or num_cols == 0:
        return False
    block_r, block_c = _tiled_scatter_shape(
        num_rows, num_cols, grad_output.element_size()
    )
    grid = (triton.cdiv(num_rows, block_r) * triton.cdiv(num_cols, block_c),)
    _diag_scatter_tiled_kernel[grid](
        grad_output,
        grad_input,
        base_offset,
        num_rows,
        num_cols,
        SRC_ROW=src_strides[row_axis],
        SRC_COL=src_strides[col_axis],
        DST_ROW=diag_strides[row_axis],
        DST_COL=diag_strides[col_axis],
        BLOCK_R=block_r,
        BLOCK_C=block_c,
        WIDE=wide,
        num_warps=1,
    )
    return True


def diagonal_backward(grad_output, input_sizes, offset, dim1, dim2):
    logger.debug("GEMS_ENFLAME DIAGONAL_BACKWARD")
    input_sizes = tuple(input_sizes)
    grad_input = torch.empty(
        input_sizes, dtype=grad_output.dtype, device=grad_output.device
    )
    numel = grad_input.numel()
    if numel == 0:
        return grad_input

    block = _zero_fill_block(numel, grad_output.element_size())
    wide = numel >= 2**31
    zero_grid = (min(_NUM_CTAS, triton.cdiv(numel, block)),)
    _zero_fill_kernel[zero_grid](grad_input, numel, BLOCK=block, WIDE=wide, num_warps=1)

    diag_shape, diag_strides, base_offset = _diagonal_view_meta(
        input_sizes, offset, dim1, dim2
    )
    diag_numel = 1
    for size in diag_shape:
        diag_numel *= size
    if diag_numel == 0 or grad_output.numel() == 0:
        return grad_input

    if _launch_tiled_scatter(
        grad_output, grad_input, diag_shape, diag_strides, base_offset, wide
    ):
        return grad_input

    _diag_scatter_kernel[(triton.cdiv(diag_numel, _SCATTER_BLOCK),)](
        grad_output,
        grad_input,
        base_offset,
        diag_numel,
        DECODE_STRIDES=_contiguous_strides(diag_shape),
        SRC_STRIDES=tuple(grad_output.stride()),
        DST_STRIDES=diag_strides,
        BLOCK=_SCATTER_BLOCK,
        WIDE=wide,
        num_warps=1,
    )
    return grad_input

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

logger = logging.getLogger(__name__)


@triton.jit
def slice_backward_kernel(
    grad_output_ptr,
    grad_input_ptr,
    numel,
    inner,
    slice_len,
    dim_size,
    start,
    step,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offsets < numel

    grad = tl.load(grad_output_ptr + offsets, mask=mask)

    outer_idx = offsets // (slice_len * inner)

    slice_idx = (offsets // inner) % slice_len

    inner_idx = offsets % inner

    dim_index = start + slice_idx * step

    input_offset = outer_idx * dim_size * inner + dim_index * inner + inner_idx

    tl.store(grad_input_ptr + input_offset, grad, mask=mask)


@triton.jit
def slice_backward_rows_kernel(
    grad_output_ptr,
    grad_input_ptr,
    inner,
    slice_len,
    dim_size,
    start,
    step,
    num_rows,
    ncol,
    total_blocks,
    ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    """Row-tiled scatter for slices with a contiguous inner tail.

    One program covers ``ROWS`` (outer, slice) rows x ``BLOCK_COLS`` inner
    columns, so the three per-element integer divisions of the flat kernel
    collapse into two per-program divisions. Each row is a contiguous run of
    ``inner`` elements on both sides, which keeps the load/store coalesced.

    The host guarantees ``ncol * BLOCK_COLS == inner``, so no column mask is
    required (predicated 2-D tiles miscompute on this backend). Tail rows are
    clamped onto the last valid row instead of masked: the scatter is
    idempotent per destination element, so the repeated write is harmless.

    The tile space is flattened over a 2D grid so neither grid dimension
    exceeds the hardware limit of 65535 (huge ``outer * slice_len`` inputs).
    """
    grid_x = tl.minimum(65535, total_blocks)
    linear_pid = tl.program_id(0) + tl.program_id(1) * grid_x
    if linear_pid >= total_blocks:
        return

    col_block = linear_pid % ncol
    row_block = linear_pid // ncol

    rows = row_block * ROWS + tl.arange(0, ROWS)
    rows = tl.minimum(rows, num_rows - 1)
    # rows enumerates (outer, slice_idx) pairs, so a single division per row
    # recovers the two logical coordinates.
    outer_idx = rows // slice_len
    slice_idx = rows - outer_idx * slice_len

    cols = col_block * BLOCK_COLS + tl.arange(0, BLOCK_COLS)

    src = rows[:, None] * inner + cols[None, :]
    dst_row = (outer_idx * dim_size + (start + slice_idx * step)) * inner
    dst = dst_row[:, None] + cols[None, :]

    tl.store(grad_input_ptr + dst, tl.load(grad_output_ptr + src))


def slice_backward(
    grad_output,
    input_sizes,
    dim,
    start,
    end,
    step,
):
    logger.debug("GEMS_ENFLAME SLICE_BACKWARD")
    grad_input = torch.zeros(
        input_sizes,
        device=grad_output.device,
        dtype=grad_output.dtype,
    )

    shape = list(input_sizes)

    if dim < 0:
        dim += len(shape)

    outer = 1
    for i in range(dim):
        outer *= shape[i]

    inner = 1
    for i in range(dim + 1, len(shape)):
        inner *= shape[i]

    dim_size = shape[dim]

    slice_len = grad_output.shape[dim]
    if start < 0:
        start += dim_size
    start = max(0, min(start, dim_size))

    numel = grad_output.numel()

    # The row-tiled path needs a moderate tensor: on multi-GB grad_input buffers
    # (>= 2**27 elements) the wide unmasked scatter trips a device fault, and the
    # flat kernel is already bandwidth bound there (it wins ~3.8x over aten for
    # the 512M-element stride-2 case).
    if inner >= 64 and 0 < numel <= (1 << 27):
        # Contiguous inner tail: use the row-tiled scatter, where every row is
        # one contiguous run of ``inner`` elements on both sides.
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        num_rows = outer * slice_len
        # Largest power-of-two column block that divides the row exactly: this is
        # what lets the kernel run fully unmasked (see the kernel docstring).
        block_cols = 1
        while block_cols * 2 <= min(1024, inner) and inner % (block_cols * 2) == 0:
            block_cols *= 2

        if block_cols >= 64:
            # One 64K-element tile per program is the bandwidth sweet spot on
            # GCU400 (measured 2.1 TB/s vs 0.85 TB/s at 16K). The 2D grid
            # fallback keeps both dimensions under the 65535 hardware limit.
            rows_per_prog = max(1, 65536 // block_cols)
            ncol = inner // block_cols
            total_blocks = triton.cdiv(num_rows, rows_per_prog) * ncol
            grid = (min(65535, total_blocks), triton.cdiv(total_blocks, 65535))
            slice_backward_rows_kernel[grid](
                grad_output,
                grad_input,
                inner,
                slice_len,
                dim_size,
                start,
                step,
                num_rows,
                ncol,
                total_blocks,
                ROWS=rows_per_prog,
                BLOCK_COLS=block_cols,
                num_warps=8,
            )
            return grad_input

    BLOCK = 1024
    if numel // BLOCK > 65535:
        BLOCK = 16384

    grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)

    slice_backward_kernel[grid](
        grad_output,
        grad_input,
        numel,
        inner,
        slice_len,
        dim_size,
        start,
        step,
        BLOCK_SIZE=BLOCK,
    )

    return grad_input

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

from flag_gems.ops.padded_dense_to_jagged_forward import _check_inputs
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

_MAX_JAGGED_DIMS = 5
_MAX_GRID = 65535
_BLOCK = 4096


@triton.jit
def _upper_bound(
    offsets,
    value,
    OFFSET_SIZE: tl.constexpr,
    LOG_OFFSET_SIZE: tl.constexpr,
    USE_INT64_INDEX: tl.constexpr,
):
    if USE_INT64_INDEX:
        low = value.to(tl.int64) * 0
        high = low + OFFSET_SIZE
    else:
        low = value.to(tl.int32) * 0
        high = low + OFFSET_SIZE
    for _ in range(LOG_OFFSET_SIZE):
        mid = low + (high - low) // 2
        mid = tl.minimum(mid, OFFSET_SIZE - 1)
        mid_value = tl.load(offsets + mid)
        go_left = value < mid_value
        high = tl.where(go_left, mid, high)
        low = tl.where(go_left, low, mid + 1)
    return low


@triton.jit
def _walk_up_offset_tree(
    offsets,
    value,
    max_length,
    OFFSET_SIZE: tl.constexpr,
    LOG_OFFSET_SIZE: tl.constexpr,
    USE_INT64_INDEX: tl.constexpr,
):
    parent = (
        _upper_bound(
            offsets,
            value,
            OFFSET_SIZE,
            LOG_OFFSET_SIZE,
            USE_INT64_INDEX,
        )
        - 1
    )
    parent_in_bounds = (parent >= 0) & (parent + 1 < OFFSET_SIZE)
    safe_parent = tl.where(parent_in_bounds, parent, 0)
    sequence_start = tl.load(offsets + safe_parent)
    sequence_end = tl.load(offsets + safe_parent + 1)
    coordinate = value - sequence_start
    level_valid = (
        parent_in_bounds
        & (sequence_start >= 0)
        & (sequence_end >= sequence_start)
        & (value >= sequence_start)
        & (value < sequence_end)
        & (coordinate < max_length)
    )
    return safe_parent, tl.where(level_valid, coordinate, 0), level_valid


@triton.jit
def _padded_dense_to_jagged_kernel(
    dense,
    offsets_0,
    offsets_1,
    offsets_2,
    offsets_3,
    offsets_4,
    output,
    total_L,
    sequence_count,
    inner_size,
    prefix_volume,
    max_length_0,
    max_length_1,
    max_length_2,
    max_length_3,
    max_length_4,
    NUM_JAGGED_DIMS: tl.constexpr,
    OFFSET_SIZE_0: tl.constexpr,
    OFFSET_SIZE_1: tl.constexpr,
    OFFSET_SIZE_2: tl.constexpr,
    OFFSET_SIZE_3: tl.constexpr,
    OFFSET_SIZE_4: tl.constexpr,
    LOG_OFFSET_SIZE_0: tl.constexpr,
    LOG_OFFSET_SIZE_1: tl.constexpr,
    LOG_OFFSET_SIZE_2: tl.constexpr,
    LOG_OFFSET_SIZE_3: tl.constexpr,
    LOG_OFFSET_SIZE_4: tl.constexpr,
    USE_INT64_INDEX: tl.constexpr,
    BLOCK: tl.constexpr,
    MAX_GRID: tl.constexpr,
):
    # One program copies one BLOCK-sized chunk of one deepest-level sequence.
    # Work is mapped through the three grid dimensions; the kernel body has no
    # loop of any kind.
    sequence_idx = tl.program_id(0) + tl.program_id(2) * MAX_GRID
    local = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    in_range = sequence_idx < sequence_count

    if USE_INT64_INDEX:
        tree = sequence_idx.to(tl.int64)
    else:
        tree = sequence_idx.to(tl.int32)
    jagged_index = tree * 0
    jagged_stride = tree * 0 + 1
    valid = in_range

    if NUM_JAGGED_DIMS == 5:
        sequence_start = tl.load(offsets_4 + sequence_idx, mask=in_range, other=0)
        sequence_end = tl.load(offsets_4 + sequence_idx + 1, mask=in_range, other=0)
    elif NUM_JAGGED_DIMS == 4:
        sequence_start = tl.load(offsets_3 + sequence_idx, mask=in_range, other=0)
        sequence_end = tl.load(offsets_3 + sequence_idx + 1, mask=in_range, other=0)
    elif NUM_JAGGED_DIMS == 3:
        sequence_start = tl.load(offsets_2 + sequence_idx, mask=in_range, other=0)
        sequence_end = tl.load(offsets_2 + sequence_idx + 1, mask=in_range, other=0)
    elif NUM_JAGGED_DIMS == 2:
        sequence_start = tl.load(offsets_1 + sequence_idx, mask=in_range, other=0)
        sequence_end = tl.load(offsets_1 + sequence_idx + 1, mask=in_range, other=0)
    else:
        sequence_start = tl.load(offsets_0 + sequence_idx, mask=in_range, other=0)
        sequence_end = tl.load(offsets_0 + sequence_idx + 1, mask=in_range, other=0)

    sequence_length = sequence_end - sequence_start
    if NUM_JAGGED_DIMS == 5:
        final_max_length = max_length_4
    elif NUM_JAGGED_DIMS == 4:
        final_max_length = max_length_3
    elif NUM_JAGGED_DIMS == 3:
        final_max_length = max_length_2
    elif NUM_JAGGED_DIMS == 2:
        final_max_length = max_length_1
    else:
        final_max_length = max_length_0

    sequence_valid = (
        (sequence_start >= 0)
        & (sequence_end >= sequence_start)
        & (sequence_end <= total_L)
        & (sequence_length <= final_max_length)
    )

    # Walk the deepest level first: its coordinate carries unit stride inside
    # the sequence's dense row, and every parent level adds with the stride of
    # the level below it.
    if NUM_JAGGED_DIMS == 5:
        parent, coordinate, level_valid = _walk_up_offset_tree(
            offsets_4,
            sequence_start,
            max_length_4,
            OFFSET_SIZE_4,
            LOG_OFFSET_SIZE_4,
            USE_INT64_INDEX,
        )
        valid = valid & level_valid
        jagged_index += coordinate * jagged_stride
        jagged_stride *= max_length_4
        tree = parent
    elif NUM_JAGGED_DIMS == 4:
        parent, coordinate, level_valid = _walk_up_offset_tree(
            offsets_3,
            sequence_start,
            max_length_3,
            OFFSET_SIZE_3,
            LOG_OFFSET_SIZE_3,
            USE_INT64_INDEX,
        )
        valid = valid & level_valid
        jagged_index += coordinate * jagged_stride
        jagged_stride *= max_length_3
        tree = parent
    elif NUM_JAGGED_DIMS == 3:
        parent, coordinate, level_valid = _walk_up_offset_tree(
            offsets_2,
            sequence_start,
            max_length_2,
            OFFSET_SIZE_2,
            LOG_OFFSET_SIZE_2,
            USE_INT64_INDEX,
        )
        valid = valid & level_valid
        jagged_index += coordinate * jagged_stride
        jagged_stride *= max_length_2
        tree = parent
    elif NUM_JAGGED_DIMS == 2:
        parent, coordinate, level_valid = _walk_up_offset_tree(
            offsets_1,
            sequence_start,
            max_length_1,
            OFFSET_SIZE_1,
            LOG_OFFSET_SIZE_1,
            USE_INT64_INDEX,
        )
        valid = valid & level_valid
        jagged_index += coordinate * jagged_stride
        jagged_stride *= max_length_1
        tree = parent

    if NUM_JAGGED_DIMS >= 5:
        parent, coordinate, level_valid = _walk_up_offset_tree(
            offsets_3,
            tree,
            max_length_3,
            OFFSET_SIZE_3,
            LOG_OFFSET_SIZE_3,
            USE_INT64_INDEX,
        )
        valid = valid & level_valid
        jagged_index += coordinate * jagged_stride
        jagged_stride *= max_length_3
        tree = parent

    if NUM_JAGGED_DIMS >= 4:
        parent, coordinate, level_valid = _walk_up_offset_tree(
            offsets_2,
            tree,
            max_length_2,
            OFFSET_SIZE_2,
            LOG_OFFSET_SIZE_2,
            USE_INT64_INDEX,
        )
        valid = valid & level_valid
        jagged_index += coordinate * jagged_stride
        jagged_stride *= max_length_2
        tree = parent

    if NUM_JAGGED_DIMS >= 3:
        parent, coordinate, level_valid = _walk_up_offset_tree(
            offsets_1,
            tree,
            max_length_1,
            OFFSET_SIZE_1,
            LOG_OFFSET_SIZE_1,
            USE_INT64_INDEX,
        )
        valid = valid & level_valid
        jagged_index += coordinate * jagged_stride
        jagged_stride *= max_length_1
        tree = parent

    # For a single jagged level the sequence *is* a batch row: the dense row is
    # `batch_idx * max_length` and the in-batch coordinate is zero. Two or more
    # levels walk the remaining offset levels up to the batch.
    if NUM_JAGGED_DIMS >= 2:
        parent, coordinate, level_valid = _walk_up_offset_tree(
            offsets_0,
            tree,
            max_length_0,
            OFFSET_SIZE_0,
            LOG_OFFSET_SIZE_0,
            USE_INT64_INDEX,
        )
        valid = valid & level_valid
        jagged_index += coordinate * jagged_stride
        batch_idx = parent
    else:
        batch_idx = tree

    sequence_elements = sequence_length * inner_size
    # Fold the scalar validity into the lane limit so the copy mask stays a
    # single lane comparison (a vector comparison ANDed with a scalar does not
    # lower correctly on this backend).
    limit = tl.where(sequence_valid & valid, sequence_elements, 0)
    mask = local < limit
    # ``jagged_index`` already indexes the whole padded block inside one batch
    # row, so the batch term is the full block volume
    # (``prefix_volume * final_max_length``) and ``inner_size`` turns the row
    # index into element offsets.
    batch_rows = prefix_volume * final_max_length
    dense_start = (batch_idx * batch_rows + jagged_index) * inner_size
    values = tl.load(dense + dense_start + local, mask=mask, other=0)
    tl.store(output + sequence_start * inner_size + local, values, mask=mask)


def _packed_views(dense, output, inner_size):
    """Copy every bit, including NaN payloads and signed zeros.

    FP64 storage is reinterpreted as INT64 so no 64-bit float load is emitted,
    and FP16 is widened so a single lane moves four adjacent elements.
    """
    kernel_dense = dense
    kernel_output = output
    kernel_inner_size = inner_size
    if dense.dtype == torch.float64:
        kernel_dense = dense.view(torch.int64)
        kernel_output = output.view(torch.int64)
    elif dense.dtype == torch.float16 and inner_size % 4 == 0:
        kernel_dense = dense.view(torch.int64)
        kernel_output = output.view(torch.int64)
        kernel_inner_size = inner_size // 4
    elif dense.dtype == torch.float16 and inner_size % 2 == 0:
        kernel_dense = dense.view(torch.int32)
        kernel_output = output.view(torch.int32)
        kernel_inner_size = inner_size // 2
    return kernel_dense, kernel_output, kernel_inner_size


def _padded_dense_to_jagged_forward(dense, offsets, total_L=None):
    """Convert padded dense storage into a one- to five-level jagged tensor."""
    logger.debug("GEMS_KUNLUNXIN PADDED_DENSE_TO_JAGGED_FORWARD")

    num_jagged_dims, max_lengths, total_L = _check_inputs(dense, offsets, total_L)
    dense_contiguous = dense.contiguous()
    offsets_contiguous = [offset.contiguous() for offset in offsets]
    inner_size = dense.size(-1)
    output = torch.empty((total_L, inner_size), dtype=dense.dtype, device=dense.device)

    if output.numel() == 0 or dense.size(0) == 0:
        return output

    kernel_dense, kernel_output, kernel_inner_size = _packed_views(
        dense_contiguous, output, inner_size
    )

    prefix_volume = 1
    for length in max_lengths[:-1]:
        prefix_volume *= length

    padded_offsets = list(offsets_contiguous) + [offsets_contiguous[-1]] * (
        _MAX_JAGGED_DIMS - num_jagged_dims
    )
    padded_lengths = list(max_lengths) + [1] * (_MAX_JAGGED_DIMS - num_jagged_dims)
    offset_sizes = [offset.numel() for offset in padded_offsets]
    offset_logs = [size.bit_length() for size in offset_sizes]

    use_int64_index = max(dense.numel(), output.numel(), *offset_sizes) >= 2**31

    sequence_count = offsets_contiguous[-1].numel() - 1
    block = _BLOCK
    chunks = triton.cdiv(max_lengths[-1] * kernel_inner_size, block)
    # The hardware limits every grid dimension to MAX_GRID; widen the chunk
    # instead of looping in the kernel.
    while chunks > _MAX_GRID:
        block *= 2
        chunks = triton.cdiv(max_lengths[-1] * kernel_inner_size, block)
    grid = (
        min(sequence_count, _MAX_GRID),
        chunks,
        triton.cdiv(sequence_count, _MAX_GRID),
    )

    with torch_device_fn.device(dense.device):
        _padded_dense_to_jagged_kernel[grid](
            kernel_dense,
            *padded_offsets,
            kernel_output,
            total_L,
            sequence_count,
            kernel_inner_size,
            prefix_volume,
            *padded_lengths,
            NUM_JAGGED_DIMS=num_jagged_dims,
            OFFSET_SIZE_0=offset_sizes[0],
            OFFSET_SIZE_1=offset_sizes[1],
            OFFSET_SIZE_2=offset_sizes[2],
            OFFSET_SIZE_3=offset_sizes[3],
            OFFSET_SIZE_4=offset_sizes[4],
            LOG_OFFSET_SIZE_0=offset_logs[0],
            LOG_OFFSET_SIZE_1=offset_logs[1],
            LOG_OFFSET_SIZE_2=offset_logs[2],
            LOG_OFFSET_SIZE_3=offset_logs[3],
            LOG_OFFSET_SIZE_4=offset_logs[4],
            USE_INT64_INDEX=use_int64_index,
            BLOCK=block,
            MAX_GRID=_MAX_GRID,
        )
    return output

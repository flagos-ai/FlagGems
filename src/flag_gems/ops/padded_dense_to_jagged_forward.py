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
import weakref
from collections import OrderedDict

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_SUPPORTED_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int64,
}
_SUPPORTED_OFFSET_DTYPES = {torch.int32, torch.int64}
_MAX_JAGGED_DIMS = 5
_MAX_GRID_SIZE = 65535
_VALIDATION_CACHE_CAPACITY = 128

# Offsets are commonly reused for many invocations. Content validation must read
# device data to report malformed trees synchronously, so cache successful checks
# by object identity and Tensor version. A weak reference prevents this bounded
# cache from keeping user tensors alive. Kernels still mask every derived address;
# the cache is a synchronization optimization, not the memory-safety boundary.
_validation_cache = OrderedDict()


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
    else:
        low = value.to(tl.int32) * 0
    high = low + OFFSET_SIZE

    for _ in range(LOG_OFFSET_SIZE):
        active = low < high
        mid = low + (high - low) // 2
        mid_value = tl.load(offsets + mid, mask=active, other=0)
        go_left = value < mid_value
        high = tl.where(active & go_left, mid, high)
        low = tl.where(active & ~go_left, mid + 1, low)
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
    safe_coordinate = tl.where(level_valid, coordinate, 0)
    return safe_parent, safe_coordinate, level_valid


@triton.jit
def _padded_dense_to_jagged_single_kernel(
    dense,
    offsets,
    output,
    total_tasks,
    chunks_per_batch,
    max_length,
    inner_size,
    total_L,
    USE_INT64_INDEX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    worker = tle.program_id(0)
    worker_count = tle.num_programs(0)

    for raw_task in tl.range(worker, total_tasks, worker_count):
        if USE_INT64_INDEX:
            task = raw_task.to(tl.int64)
        else:
            task = raw_task.to(tl.int32)
        batch_idx = task // chunks_per_batch
        chunk_idx = task - batch_idx * chunks_per_batch
        sequence_start = tl.load(offsets + batch_idx)
        sequence_end = tl.load(offsets + batch_idx + 1)
        sequence_elements = (sequence_end - sequence_start) * inner_size
        sequence_valid = (
            (sequence_start >= 0)
            & (sequence_end >= sequence_start)
            & (sequence_end <= total_L)
            & (sequence_end - sequence_start <= max_length)
        )

        local_offsets = chunk_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = (local_offsets < sequence_elements) & sequence_valid
        dense_offsets = batch_idx * max_length * inner_size + local_offsets
        output_offsets = sequence_start * inner_size + local_offsets
        values = tl.load(dense + dense_offsets, mask=mask)
        tl.store(output + output_offsets, values, mask=mask)


@triton.jit
def _padded_dense_to_jagged_multi_kernel(
    dense,
    offsets_0,
    offsets_1,
    offsets_2,
    offsets_3,
    offsets_4,
    output,
    deepest_parent_count,
    chunks_per_sequence,
    inner_size,
    prefix_padded_volume,
    final_max_length,
    total_L,
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
    BLOCK_SIZE: tl.constexpr,
):
    worker = tle.program_id(0)
    worker_count = tle.num_programs(0)

    for raw_sequence_idx in tl.range(worker, deepest_parent_count, worker_count):
        if USE_INT64_INDEX:
            sequence_idx = raw_sequence_idx.to(tl.int64)
            tree_offset = sequence_idx
            jagged_index = tree_offset * 0
            jagged_stride = tree_offset * 0 + 1
        else:
            sequence_idx = raw_sequence_idx.to(tl.int32)
            tree_offset = sequence_idx
            jagged_index = tree_offset * 0
            jagged_stride = tree_offset * 0 + 1
        tree_valid = tree_offset >= 0

        if NUM_JAGGED_DIMS == 5:
            sequence_start = tl.load(offsets_4 + sequence_idx)
            sequence_end = tl.load(offsets_4 + sequence_idx + 1)
        elif NUM_JAGGED_DIMS == 4:
            sequence_start = tl.load(offsets_3 + sequence_idx)
            sequence_end = tl.load(offsets_3 + sequence_idx + 1)
        elif NUM_JAGGED_DIMS == 3:
            sequence_start = tl.load(offsets_2 + sequence_idx)
            sequence_end = tl.load(offsets_2 + sequence_idx + 1)
        else:
            sequence_start = tl.load(offsets_1 + sequence_idx)
            sequence_end = tl.load(offsets_1 + sequence_idx + 1)

        sequence_length = sequence_end - sequence_start
        sequence_valid = (
            (sequence_start >= 0)
            & (sequence_end >= sequence_start)
            & (sequence_end <= total_L)
            & (sequence_length <= final_max_length)
        )

        if NUM_JAGGED_DIMS >= 5:
            parent, coordinate, level_valid = _walk_up_offset_tree(
                offsets_3,
                tree_offset,
                max_length_3,
                OFFSET_SIZE_3,
                LOG_OFFSET_SIZE_3,
                USE_INT64_INDEX,
            )
            tree_valid = tree_valid & level_valid
            jagged_index += coordinate * jagged_stride
            jagged_stride *= max_length_3
            tree_offset = parent

        if NUM_JAGGED_DIMS >= 4:
            parent, coordinate, level_valid = _walk_up_offset_tree(
                offsets_2,
                tree_offset,
                max_length_2,
                OFFSET_SIZE_2,
                LOG_OFFSET_SIZE_2,
                USE_INT64_INDEX,
            )
            tree_valid = tree_valid & level_valid
            jagged_index += coordinate * jagged_stride
            jagged_stride *= max_length_2
            tree_offset = parent

        if NUM_JAGGED_DIMS >= 3:
            parent, coordinate, level_valid = _walk_up_offset_tree(
                offsets_1,
                tree_offset,
                max_length_1,
                OFFSET_SIZE_1,
                LOG_OFFSET_SIZE_1,
                USE_INT64_INDEX,
            )
            tree_valid = tree_valid & level_valid
            jagged_index += coordinate * jagged_stride
            jagged_stride *= max_length_1
            tree_offset = parent

        parent, coordinate, level_valid = _walk_up_offset_tree(
            offsets_0,
            tree_offset,
            max_length_0,
            OFFSET_SIZE_0,
            LOG_OFFSET_SIZE_0,
            USE_INT64_INDEX,
        )
        tree_valid = tree_valid & level_valid
        jagged_index += coordinate * jagged_stride
        batch_idx = parent

        sequence_elements = sequence_length * inner_size
        prefix_dense_row = batch_idx * prefix_padded_volume + jagged_index
        dense_sequence_start = prefix_dense_row * final_max_length * inner_size
        output_sequence_start = sequence_start * inner_size
        for raw_chunk_idx in tl.range(0, chunks_per_sequence):
            if USE_INT64_INDEX:
                chunk_idx = raw_chunk_idx.to(tl.int64)
            else:
                chunk_idx = raw_chunk_idx.to(tl.int32)
            local_offsets = chunk_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = (local_offsets < sequence_elements) & sequence_valid & tree_valid
            dense_offsets = dense_sequence_start + local_offsets
            output_offsets = output_sequence_start + local_offsets
            values = tl.load(dense + dense_offsets, mask=mask, other=0)
            tl.store(output + output_offsets, values, mask=mask)


def _tensor_version(tensor):
    try:
        return tensor._version
    except RuntimeError:
        # Inference tensors do not expose a version counter. They are deliberately
        # not cached because their contents cannot be invalidated reliably.
        return None


def _validation_cache_key(offsets, max_lengths):
    versions = tuple(_tensor_version(offset) for offset in offsets)
    if any(version is None for version in versions):
        return None
    return (
        tuple(id(offset) for offset in offsets),
        versions,
        tuple(offset.numel() for offset in offsets),
        tuple(max_lengths),
    )


def _lookup_validated_terminal(cache_key, offsets):
    if cache_key is None:
        return None
    cached = _validation_cache.get(cache_key)
    if cached is None:
        return None
    references, terminal = cached
    if not all(reference() is offset for reference, offset in zip(references, offsets)):
        _validation_cache.pop(cache_key, None)
        return None
    _validation_cache.move_to_end(cache_key)
    return terminal


def _cache_validated_terminal(cache_key, offsets, terminal):
    if cache_key is None:
        return
    try:
        references = tuple(weakref.ref(offset) for offset in offsets)
    except TypeError:
        return
    _validation_cache[cache_key] = (references, terminal)
    _validation_cache.move_to_end(cache_key)
    while len(_validation_cache) > _VALIDATION_CACHE_CAPACITY:
        _validation_cache.popitem(last=False)


def _validate_offset_contents(offsets, max_lengths):
    cache_key = _validation_cache_key(offsets, max_lengths)
    terminal = _lookup_validated_terminal(cache_key, offsets)
    if terminal is not None:
        return terminal

    host_offsets = [offset.detach().cpu().tolist() for offset in offsets]
    for level, (values, max_length) in enumerate(zip(host_offsets, max_lengths)):
        if values[0] != 0:
            raise RuntimeError(f"offsets[{level}] must start at 0")
        previous = values[0]
        for current in values[1:]:
            if current < previous:
                raise RuntimeError(f"offsets[{level}] must be nondecreasing")
            if current - previous > max_length:
                raise RuntimeError(
                    f"offsets[{level}] contains a segment longer than "
                    f"dense.size({level + 1})"
                )
            previous = current

        if level + 1 < len(host_offsets):
            child_count = len(host_offsets[level + 1]) - 1
            if values[-1] != child_count:
                raise RuntimeError(
                    f"offsets[{level}] terminal value must equal the number of "
                    f"segments in offsets[{level + 1}]"
                )

    terminal = host_offsets[-1][-1]
    _cache_validated_terminal(cache_key, offsets, terminal)
    return terminal


def _check_inputs(dense, offsets, total_L):
    if not isinstance(offsets, (list, tuple)) or len(offsets) == 0:
        raise RuntimeError("offsets must be a non-empty list of tensors")

    num_jagged_dims = dense.dim() - 2
    if not 1 <= num_jagged_dims <= _MAX_JAGGED_DIMS:
        raise RuntimeError(
            "dense must have between 1 and 5 jagged dimensions "
            f"(rank 3 to 7), but got rank {dense.dim()}"
        )
    if len(offsets) != num_jagged_dims:
        raise RuntimeError(
            f"expected {num_jagged_dims} offsets tensors for dense rank "
            f"{dense.dim()}, but got {len(offsets)}"
        )
    if dense.dtype not in _SUPPORTED_DTYPES:
        raise NotImplementedError(
            f"_padded_dense_to_jagged_forward is not implemented for {dense.dtype}"
        )

    if not isinstance(offsets[0], torch.Tensor):
        raise TypeError("offsets[0] must be a Tensor")
    first_offset_dtype = offsets[0].dtype
    for level, offset in enumerate(offsets):
        if not isinstance(offset, torch.Tensor):
            raise TypeError(f"offsets[{level}] must be a Tensor")
        if offset.dim() != 1:
            raise RuntimeError(f"offsets[{level}] must be one-dimensional")
        if offset.numel() == 0:
            raise RuntimeError(f"offsets[{level}] must be non-empty")
        if offset.dtype not in _SUPPORTED_OFFSET_DTYPES:
            raise RuntimeError(
                f"offsets[{level}] must have dtype torch.int32 or torch.int64"
            )
        if offset.dtype != first_offset_dtype:
            raise RuntimeError("all offsets tensors must have the same dtype")
        if offset.device != dense.device:
            raise RuntimeError(f"offsets[{level}] must be on the same device as dense")

    batch_size = dense.size(0)
    if offsets[0].numel() - 1 != batch_size:
        raise RuntimeError(
            "offsets[0].numel() - 1 must equal dense.size(0), but got "
            f"{offsets[0].numel() - 1} and {batch_size}"
        )

    max_lengths = tuple(dense.size(level + 1) for level in range(num_jagged_dims))
    terminal = _validate_offset_contents(offsets, max_lengths)

    if total_L is None:
        resolved_total_L = terminal
    else:
        resolved_total_L = int(total_L)
        if resolved_total_L < 0:
            raise RuntimeError("total_L must be non-negative")
        if resolved_total_L != terminal:
            raise RuntimeError(
                "total_L must equal the terminal value of the final offsets "
                f"tensor, but got {resolved_total_L} and {terminal}"
            )
    return num_jagged_dims, max_lengths, resolved_total_L


def _launch_single(dense, offset, output, max_length, inner_size):
    if output.numel() == 0 or dense.size(0) == 0:
        return
    block_size = 256
    chunks_per_batch = triton.cdiv(max_length * inner_size, block_size)
    total_tasks = dense.size(0) * chunks_per_batch
    grid = (min(total_tasks, _MAX_GRID_SIZE),)
    use_int64_index = max(dense.numel(), output.numel(), offset.numel()) >= 2**31
    _padded_dense_to_jagged_single_kernel[grid](
        dense,
        offset,
        output,
        total_tasks,
        chunks_per_batch,
        max_length,
        inner_size,
        output.size(0),
        USE_INT64_INDEX=use_int64_index,
        BLOCK_SIZE=block_size,
    )


def _launch_multi(dense, offsets, output, max_lengths, total_L, inner_size):
    if output.numel() == 0:
        return
    block_size = 256
    deepest_parent_count = offsets[-1].numel() - 1
    chunks_per_sequence = triton.cdiv(max_lengths[-1] * inner_size, block_size)
    grid = (min(deepest_parent_count, _MAX_GRID_SIZE),)

    padded_offsets = list(offsets) + [offsets[-1]] * (_MAX_JAGGED_DIMS - len(offsets))
    padded_lengths = list(max_lengths) + [1] * (_MAX_JAGGED_DIMS - len(max_lengths))
    offset_sizes = [offset.numel() for offset in padded_offsets]
    offset_logs = [size.bit_length() for size in offset_sizes]
    prefix_padded_volume = 1
    for length in max_lengths[:-1]:
        prefix_padded_volume *= length

    use_int64_index = (
        max(
            dense.numel(),
            output.numel(),
            *offset_sizes,
            prefix_padded_volume,
            max_lengths[-1] * inner_size,
        )
        >= 2**31
    )
    _padded_dense_to_jagged_multi_kernel[grid](
        dense,
        *padded_offsets,
        output,
        deepest_parent_count,
        chunks_per_sequence,
        inner_size,
        prefix_padded_volume,
        max_lengths[-1],
        total_L,
        *padded_lengths,
        NUM_JAGGED_DIMS=len(offsets),
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
        BLOCK_SIZE=block_size,
    )


def _padded_dense_to_jagged_forward(dense, offsets, total_L=None):
    """Convert padded dense storage into a one- to five-level jagged tensor."""
    logger.debug("GEMS PADDED DENSE TO JAGGED FORWARD")

    num_jagged_dims, max_lengths, total_L = _check_inputs(dense, offsets, total_L)
    dense_contiguous = dense.contiguous()
    offsets_contiguous = [offset.contiguous() for offset in offsets]
    inner_size = dense.size(-1)
    output = torch.empty((total_L, inner_size), dtype=dense.dtype, device=dense.device)

    # This operator only copies elements. Storage views preserve every output
    # bit, including NaN payloads and signed zero. FP64-as-int64 also avoids
    # backends that cannot lower FP64 loads, while packing FP16 widens copies.
    kernel_dense = dense_contiguous
    kernel_output = output
    kernel_inner_size = inner_size
    if dense.dtype == torch.float64:
        kernel_dense = dense_contiguous.view(torch.int64)
        kernel_output = output.view(torch.int64)
    elif dense.dtype == torch.float16 and inner_size % 4 == 0:
        kernel_dense = dense_contiguous.view(torch.int64)
        kernel_output = output.view(torch.int64)
        kernel_inner_size = inner_size // 4
    elif dense.dtype == torch.float16 and inner_size % 2 == 0:
        kernel_dense = dense_contiguous.view(torch.int32)
        kernel_output = output.view(torch.int32)
        kernel_inner_size = inner_size // 2

    with torch_device_fn.device(dense.device):
        if num_jagged_dims == 1:
            _launch_single(
                kernel_dense,
                offsets_contiguous[0],
                kernel_output,
                max_lengths[0],
                kernel_inner_size,
            )
        else:
            _launch_multi(
                kernel_dense,
                offsets_contiguous,
                kernel_output,
                max_lengths,
                total_L,
                kernel_inner_size,
            )
    return output

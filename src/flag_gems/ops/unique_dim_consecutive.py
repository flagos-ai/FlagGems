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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.libentry import libentry

logger = logging.getLogger(__name__)

_UNIQUE_DIM_CONSECUTIVE_COMPARE_BLOCK_SIZE = 1024
_UNIQUE_DIM_CONSECUTIVE_GATHER_BLOCK_SIZE = 1024


@libentry()
@triton.jit
def _unique_dim_consecutive_compare_kernel(
    flat_ptr: tl.tensor,
    is_first_ptr: tl.tensor,
    num_rows: int,
    row_len: int,
    BLOCK_SIZE: tl.constexpr,
):
    """Compare each row with the previous row to mark first occurrences in consecutive groups."""
    row = ext.program_id(0)
    chunk = ext.program_id(1)
    offsets = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_len

    if row == 0:
        # First row is always the start of a new group
        out = tl.where(chunk == 0, True, False)
    else:
        # Compare current row with previous row
        cur = tl.load(flat_ptr + row * row_len + offsets, mask=mask)
        prev = tl.load(flat_ptr + (row - 1) * row_len + offsets, mask=mask)
        neq = (cur != prev) & mask
        # If any element differs, this row starts a new group
        has_diff = tl.sum(neq.to(tl.int32), axis=0) != 0
        out = has_diff.to(tl.int1)

    # Only the first chunk of each row writes the result
    if chunk == 0:
        tl.store(is_first_ptr + row, out)


@libentry()
@triton.jit
def _unique_dim_consecutive_compare_multichunk_kernel(
    flat_ptr: tl.tensor,
    chunk_diff_ptr: tl.tensor,
    num_rows: int,
    row_len: int,
    num_chunks: int,
    BLOCK_SIZE: tl.constexpr,
):
    """Compare rows chunk by chunk for large row sizes."""
    row = ext.program_id(0)
    chunk = ext.program_id(1)
    offsets = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_len

    if row == 0:
        out = tl.where(chunk == 0, 1, 0)
    else:
        cur = tl.load(flat_ptr + row * row_len + offsets, mask=mask)
        prev = tl.load(flat_ptr + (row - 1) * row_len + offsets, mask=mask)
        neq = (cur != prev) & mask
        has_diff = tl.sum(neq.to(tl.int32), axis=0) != 0
        out = has_diff.to(tl.int32)

    tl.store(chunk_diff_ptr + row * num_chunks + chunk, out)


@libentry()
@triton.jit
def _unique_dim_consecutive_reduce_chunks_kernel(
    chunk_diff_ptr: tl.tensor,
    is_first_ptr: tl.tensor,
    num_chunks: int,
    BLOCK_CHUNKS: tl.constexpr,
):
    """Reduce chunk differences to determine if row differs from previous."""
    row = ext.program_id(0)
    chunks = tl.arange(0, BLOCK_CHUNKS)
    mask = chunks < num_chunks
    vals = tl.load(chunk_diff_ptr + row * num_chunks + chunks, mask=mask, other=0)
    # If any chunk differs, the row is first in its group
    tl.store(is_first_ptr + row, tl.sum(vals, axis=0) != 0)


@libentry()
@triton.jit
def _unique_dim_consecutive_cumsum_kernel(
    is_first_ptr: tl.tensor,
    group_id_ptr: tl.tensor,
    num_unique_ptr: tl.tensor,
    num_rows: int,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute cumulative sum to assign group IDs (for small num_rows)."""
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_rows
    is_first = tl.load(is_first_ptr + offsets, mask=mask, other=False)
    group_id = tl.cumsum(is_first.to(tl.int64), axis=0) - 1
    tl.store(group_id_ptr + offsets, group_id, mask=mask)

    # Store the total number of unique groups
    last_id = tl.sum(tl.where(offsets == num_rows - 1, group_id + 1, 0), axis=0)
    tl.store(num_unique_ptr, last_id)


@libentry()
@triton.jit
def _unique_dim_consecutive_gather_indices_kernel(
    is_first_ptr: tl.tensor,
    group_id_ptr: tl.tensor,
    unique_indices_ptr: tl.tensor,
    num_rows: int,
    BLOCK_SIZE: tl.constexpr,
):
    """Gather indices of first rows in each consecutive group."""
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_rows

    is_first = tl.load(is_first_ptr + offsets, mask=mask, other=False)
    group_id = tl.load(group_id_ptr + offsets, mask=mask, other=0)

    # Scatter the row index to its group position
    write_mask = is_first & mask
    tl.store(unique_indices_ptr + group_id, offsets.to(tl.int64), mask=write_mask)


@libentry()
@triton.jit
def _unique_dim_consecutive_gather_output_kernel(
    flat_ptr: tl.tensor,
    unique_indices_ptr: tl.tensor,
    output_ptr: tl.tensor,
    num_unique: int,
    row_len: int,
    BLOCK_SIZE: tl.constexpr,
):
    """Gather unique rows into output tensor."""
    row = ext.program_id(0)
    chunk = ext.program_id(1)
    col = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col < row_len

    src_row = tl.load(unique_indices_ptr + row)
    values = tl.load(flat_ptr + src_row * row_len + col, mask=mask)
    tl.store(output_ptr + row * row_len + col, values, mask=mask)


def _triton_num_warps(block_size: int) -> int:
    if block_size >= 8192:
        return 8
    if block_size >= 2048:
        return 4
    return 1


def _compare_consecutive_rows(flat: torch.Tensor) -> torch.Tensor:
    """Compare consecutive rows and return a boolean mask of first occurrences."""
    num_rows, row_len = flat.shape
    device = flat.device

    if num_rows <= 1:
        is_first = torch.ones(num_rows, dtype=torch.bool, device=device)
        return is_first

    if row_len == 0:
        # Empty rows are all considered identical
        is_first = torch.zeros(num_rows, dtype=torch.bool, device=device)
        is_first[0] = True
        return is_first

    block_size = min(
        _UNIQUE_DIM_CONSECUTIVE_COMPARE_BLOCK_SIZE, triton.next_power_of_2(row_len)
    )
    num_chunks = triton.cdiv(row_len, block_size)

    if num_chunks == 1:
        # Single chunk case - direct comparison
        is_first = torch.empty(num_rows, dtype=torch.bool, device=device)
        with torch_device_fn.device(device.index):
            _unique_dim_consecutive_compare_kernel[(num_rows, 1, 1)](
                flat,
                is_first,
                num_rows,
                row_len,
                BLOCK_SIZE=block_size,
                num_warps=_triton_num_warps(block_size),
            )
        return is_first

    # Multi-chunk case - need to reduce across chunks
    chunk_diff = torch.empty((num_rows, num_chunks), dtype=torch.int32, device=device)
    is_first = torch.empty(num_rows, dtype=torch.bool, device=device)

    with torch_device_fn.device(device.index):
        _unique_dim_consecutive_compare_multichunk_kernel[(num_rows, num_chunks, 1)](
            flat,
            chunk_diff,
            num_rows,
            row_len,
            num_chunks,
            BLOCK_SIZE=block_size,
            num_warps=_triton_num_warps(block_size),
        )
        _unique_dim_consecutive_reduce_chunks_kernel[(num_rows, 1, 1)](
            chunk_diff,
            is_first,
            num_chunks,
            BLOCK_CHUNKS=triton.next_power_of_2(num_chunks),
            num_warps=_triton_num_warps(triton.next_power_of_2(num_chunks)),
        )

    return is_first


def _gather_unique_rows(
    flat: torch.Tensor, unique_indices: torch.Tensor
) -> torch.Tensor:
    """Gather unique rows from flat tensor."""
    num_unique = unique_indices.numel()
    row_len = flat.shape[1]
    device = flat.device

    if num_unique == 0:
        return torch.empty((0, row_len), dtype=flat.dtype, device=device)

    output = torch.empty((num_unique, row_len), dtype=flat.dtype, device=device)
    num_chunks = triton.cdiv(row_len, _UNIQUE_DIM_CONSECUTIVE_GATHER_BLOCK_SIZE)

    with torch_device_fn.device(device.index):
        _unique_dim_consecutive_gather_output_kernel[(num_unique, num_chunks, 1)](
            flat,
            unique_indices,
            output,
            num_unique,
            row_len,
            BLOCK_SIZE=_UNIQUE_DIM_CONSECUTIVE_GATHER_BLOCK_SIZE,
            num_warps=4,
        )

    return output


def unique_dim_consecutive(
    input: torch.Tensor,
    dim: int,
    return_inverse: bool = False,
    return_counts: bool = False,
):
    """
    Eliminates all but the first element from every consecutive group of equivalent elements along a dimension.

    Args:
        input: the input tensor
        dim: the dimension to apply unique
        return_inverse: Whether to return inverse indices
        return_counts: Whether to return counts for each unique element

    Returns:
        (Tensor, Tensor, Tensor): output, inverse_indices, counts
    """
    logger.debug("GEMS UNIQUE_DIM_CONSECUTIVE")

    ndim = input.ndim if input.ndim > 0 else 1
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= max(input.ndim, 1):
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-input.ndim}, {input.ndim - 1}], but got {dim})"
        )

    device = input.device
    size_dim = input.size(dim) if input.ndim > 0 else input.numel()

    if size_dim == 0:
        output = input.clone()
        inverse_indices = torch.empty(0, dtype=torch.int64, device=device)
        counts = torch.empty(0, dtype=torch.int64, device=device)
        return output, inverse_indices, counts

    # Move target dimension to front and flatten other dimensions
    moved = input.movedim(dim, 0).contiguous()
    flat = moved.reshape(size_dim, -1)

    # Compare consecutive rows
    is_first = _compare_consecutive_rows(flat)

    # Handle small tensors with single-kernel cumsum
    if size_dim <= 8192:
        group_id = torch.empty(size_dim, dtype=torch.int64, device=device)
        num_unique = torch.empty((), dtype=torch.int64, device=device)

        block_size = triton.next_power_of_2(size_dim)
        with torch_device_fn.device(device.index):
            _unique_dim_consecutive_cumsum_kernel[(1, 1, 1)](
                is_first,
                group_id,
                num_unique,
                size_dim,
                BLOCK_SIZE=block_size,
                num_warps=_triton_num_warps(block_size),
            )

        num_unique_val = int(num_unique.item())
    else:
        # For large tensors, use PyTorch cumsum
        group_id = torch.cumsum(is_first.to(torch.int64), dim=0) - 1
        num_unique_val = int(group_id[-1].item()) + 1

    # Gather unique indices
    unique_indices = torch.empty(num_unique_val, dtype=torch.int64, device=device)
    if num_unique_val > 0:
        grid = (triton.cdiv(size_dim, _UNIQUE_DIM_CONSECUTIVE_GATHER_BLOCK_SIZE), 1, 1)
        with torch_device_fn.device(device.index):
            _unique_dim_consecutive_gather_indices_kernel[grid](
                is_first,
                group_id,
                unique_indices,
                size_dim,
                BLOCK_SIZE=_UNIQUE_DIM_CONSECUTIVE_GATHER_BLOCK_SIZE,
                num_warps=4,
            )

    # Gather unique rows
    unique_flat = _gather_unique_rows(flat, unique_indices)

    # Reshape back to original structure
    output_shape = list(input.shape)
    output_shape[dim] = num_unique_val
    output = unique_flat.reshape((num_unique_val,) + moved.shape[1:]).movedim(0, dim)

    # Compute inverse indices if requested
    inverse_indices = torch.empty(0, dtype=torch.int64, device=device)
    if return_inverse:
        inverse_indices = group_id

    # Compute counts if requested
    counts = torch.empty(0, dtype=torch.int64, device=device)
    if return_counts:
        # Count occurrences of each group
        first_positions = torch.nonzero(is_first, as_tuple=False).flatten()
        counts = torch.empty(num_unique_val, dtype=torch.int64, device=device)
        if num_unique_val > 0:
            # Calculate counts as differences between consecutive first positions
            next_positions = torch.cat(
                [
                    first_positions[1:],
                    torch.tensor([size_dim], dtype=torch.int64, device=device),
                ]
            )
            counts = next_positions - first_positions

    return output, inverse_indices, counts


def unique_dim_consecutive_out(
    input: torch.Tensor,
    dim: int,
    return_inverse: bool = False,
    return_counts: bool = False,
    *,
    out0: torch.Tensor,
    out1: torch.Tensor,
    out2: torch.Tensor,
):
    """A variant of ``unique_dim_consecutive`` that writes results into the provided out tensors."""
    logger.debug("GEMS UNIQUE_DIM_CONSECUTIVE_OUT")

    output, inverse_indices, counts = unique_dim_consecutive(
        input, dim, return_inverse=return_inverse, return_counts=return_counts
    )

    out0.resize_(output.shape).copy_(output)
    out1.resize_(inverse_indices.shape).copy_(inverse_indices)
    out2.resize_(counts.shape).copy_(counts)

    return out0, out1, out2

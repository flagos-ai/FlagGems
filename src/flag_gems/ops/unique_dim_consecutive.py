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

from flag_gems.ops.unique_dim import (
    _UNIQUE_DIM_COMPARE_BLOCK_SIZE,
    _UNIQUE_DIM_GATHER_BLOCK_SIZE,
    _UNIQUE_DIM_HASH_MIN_ROW_LEN,
    _triton_num_warps,
    _unique_dim_counts_kernel,
    _unique_dim_gather_output,
    _unique_dim_row_chunk_diff_hash_kernel,
    _unique_dim_row_chunk_diff_kernel,
    _unique_dim_row_diff_reduce_kernel,
    _unique_dim_row_hash,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.libentry import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _unique_dim_consecutive_first_kernel(
    flat_ptr: tl.tensor,
    is_first_ptr: tl.tensor,
    num_rows: int,
    row_len: int,
    BLOCK_SIZE: tl.constexpr,
):
    """Mark rows that differ from the immediately preceding row (in memory
    order, no permutation involved). Row 0 always starts a group."""
    row = ext.program_id(0)
    out = tl.full((), 1, dtype=tl.int32)
    if row != 0:
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < row_len
        cur = tl.load(flat_ptr + row * row_len + offsets, mask=mask)
        prev = tl.load(flat_ptr + (row - 1) * row_len + offsets, mask=mask)
        neq = (cur != prev) & mask
        diff = tl.sum(neq.to(tl.int32), axis=0)
        out = tl.where(diff != 0, 1, 0)
    tl.store(is_first_ptr + row, out)


def _unique_dim_consecutive_first_mask(flat: torch.Tensor) -> torch.Tensor:
    """Bool mask: True where a row starts a new consecutive group."""
    num_rows, row_len = flat.shape
    is_first = torch.ones(num_rows, dtype=torch.bool, device=flat.device)
    if num_rows <= 1 or row_len == 0:
        return is_first

    block_size = min(_UNIQUE_DIM_COMPARE_BLOCK_SIZE, triton.next_power_of_2(row_len))
    num_chunks = triton.cdiv(row_len, block_size)
    if num_chunks == 1:
        with torch_device_fn.device(flat.device.index):
            _unique_dim_consecutive_first_kernel[(num_rows, 1, 1)](
                flat,
                is_first,
                num_rows,
                row_len,
                BLOCK_SIZE=block_size,
                num_warps=_triton_num_warps(block_size),
            )
        return is_first

    # Wide rows: compare in column chunks (reusing the sorted-diff kernels with
    # the identity permutation, which degenerates them to adjacent-row diffs).
    sorted_indices = torch.arange(num_rows, dtype=torch.int64, device=flat.device)
    row_chunk_diff = torch.empty(
        (num_rows, num_chunks), dtype=torch.int32, device=flat.device
    )
    grid = (num_rows, num_chunks, 1)
    row_hash = (
        _unique_dim_row_hash(flat) if row_len >= _UNIQUE_DIM_HASH_MIN_ROW_LEN else None
    )
    with torch_device_fn.device(flat.device.index):
        if row_hash is None:
            _unique_dim_row_chunk_diff_kernel[grid](
                flat,
                sorted_indices,
                row_chunk_diff,
                num_rows,
                row_len,
                num_chunks,
                BLOCK_SIZE=block_size,
                num_warps=_triton_num_warps(block_size),
            )
        else:
            _unique_dim_row_chunk_diff_hash_kernel[grid](
                flat,
                sorted_indices,
                row_hash,
                row_chunk_diff,
                num_rows,
                row_len,
                num_chunks,
                BLOCK_SIZE=block_size,
                num_warps=_triton_num_warps(block_size),
            )
        _unique_dim_row_diff_reduce_kernel[(num_rows, 1, 1)](
            row_chunk_diff,
            is_first,
            num_chunks,
            BLOCK_CHUNKS=triton.next_power_of_2(num_chunks),
            num_warps=_triton_num_warps(triton.next_power_of_2(num_chunks)),
        )
    return is_first


def unique_dim_consecutive(
    input: torch.Tensor,
    dim: int,
    return_inverse: bool = False,
    return_counts: bool = False,
):
    """Consecutive-only ``torch.unique`` along a dimension
    (a.k.a. ``aten::unique_dim_consecutive``).

    Treats each slice along ``dim`` as a single element and collapses every
    *consecutive* run of equivalent slices into the first one. Unlike
    ``unique_dim`` no sorting is performed; unlike ``unique_consecutive`` the
    ``dim`` argument is mandatory (mirroring the aten signature).
    """
    logger.debug("GEMS UNIQUE_DIM_CONSECUTIVE")

    ndim = input.ndim
    if ndim == 0:
        raise IndexError(
            "Dimension specified as {} but tensor has no dimensions".format(dim)
        )
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-ndim}, {ndim - 1}], but got {dim})"
        )

    device = input.device
    size_dim = input.size(dim)

    # Matches aten: dims with size 0 that are not the selected dim cannot be
    # uniqued ("There are 0 sized dimensions, and they aren't selected").
    if any(s == 0 for i, s in enumerate(input.shape) if i != dim):
        raise RuntimeError(
            "There are 0 sized dimensions, and they aren't selected, "
            "so unique cannot be applied"
        )

    if size_dim == 0:
        output_shape = tuple(input.shape[:dim]) + (0,) + tuple(input.shape[dim + 1 :])
        return (
            torch.empty(output_shape, dtype=input.dtype, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
        )

    moved = input.movedim(dim, 0).contiguous()
    flat = moved.reshape(size_dim, -1)

    is_first = _unique_dim_consecutive_first_mask(flat)
    group_id = torch.cumsum(is_first.to(torch.int64), dim=0) - 1
    num_unique = int(group_id[-1].item()) + 1

    counts = torch.empty(0, dtype=torch.int64, device=device)
    inverse_indices = torch.empty(0, dtype=torch.int64, device=device)

    if return_inverse:
        inverse_indices = group_id
    if return_counts:
        # Sizes of each consecutive group: positions where is_first is True,
        # diffed against the next group start (or the total for the last one).
        first_positions = torch.nonzero(is_first, as_tuple=False).flatten()
        counts = torch.empty(num_unique, dtype=torch.int64, device=device)
        if num_unique > 0:
            grid = (triton.cdiv(num_unique, _UNIQUE_DIM_GATHER_BLOCK_SIZE), 1, 1)
            with torch_device_fn.device(device.index):
                _unique_dim_counts_kernel[grid](
                    first_positions,
                    counts,
                    size_dim,
                    num_unique,
                    BLOCK_SIZE=_UNIQUE_DIM_GATHER_BLOCK_SIZE,
                    num_warps=4,
                )

    # Keep only the first row of each consecutive group.
    unique_indices = (
        first_positions
        if return_counts
        else torch.nonzero(is_first, as_tuple=False).flatten()
    )
    output = _unique_dim_gather_output(moved, unique_indices, dim, input.shape)

    return output, inverse_indices, counts

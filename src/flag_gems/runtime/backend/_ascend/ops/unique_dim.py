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
    _UNIQUE_DIM_GROUP_SCAN_BLOCK_SIZE,
    _argsort_keys,
    _build_composite_key,
)
from flag_gems.ops.unique_dim import (
    _group_id_from_sorted as _generic_group_id_from_sorted,
)
from flag_gems.ops.unique_dim import (
    _lex_argsort_rows_cascade,
    _monotonic_key_bits,
    _remap_info,
    _triton_gather_1d,
    _triton_num_warps,
    _unique_dim_counts,
    _unique_dim_first_mask,
    _unique_dim_gather_output,
    _unique_dim_inverse_from_permutation,
    _unique_dim_unique_indices,
    _unique_dim_unique_indices_and_inverse,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.libentry import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _unique_dim_group_id_kernel(
    composite_ptr: tl.tensor,
    group_id_ptr: tl.tensor,
    last_group_id_ptr: tl.tensor,
    num_rows: int,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_rows
    cur = tl.load(composite_ptr + offsets, mask=mask)
    prev_offsets = tl.where(offsets > 0, offsets - 1, 0)
    prev = tl.load(composite_ptr + prev_offsets, mask=mask)
    is_new_group = tl.where((offsets > 0) & mask, cur != prev, 0)
    group_id = tl.cumsum(is_new_group)
    tl.store(group_id_ptr + offsets, group_id, mask=mask)
    last_group_id = tl.sum(group_id * (offsets == num_rows - 1).to(tl.int32))
    tl.store(last_group_id_ptr, last_group_id)


def _empty_int64(device: torch.device) -> torch.Tensor:
    return torch.empty_strided((0,), (1,), dtype=torch.int64, device=device)


def _group_id_from_sorted(sorted_keys: torch.Tensor):
    num_rows = sorted_keys.numel()
    device = sorted_keys.device
    if num_rows == 0:
        return _empty_int64(device), -1
    if num_rows > _UNIQUE_DIM_GROUP_SCAN_BLOCK_SIZE:
        return _generic_group_id_from_sorted(sorted_keys)

    group_id = torch.empty(num_rows, dtype=torch.int64, device=device)
    last_group_id = torch.empty((), dtype=torch.int64, device=device)
    block_size = triton.next_power_of_2(num_rows)
    with torch_device_fn.device(device.index):
        _unique_dim_group_id_kernel[(1, 1, 1)](
            sorted_keys,
            group_id,
            last_group_id,
            num_rows,
            BLOCK_SIZE=block_size,
            num_warps=_triton_num_warps(block_size),
        )
    return group_id, int(last_group_id.item())


def _lex_argsort_rows_composite(flat: torch.Tensor):
    key_bits = _monotonic_key_bits(flat.dtype)
    if key_bits is None:
        return None

    num_rows, num_cols = flat.shape
    device = flat.device
    if num_cols == 0:
        indices = torch.arange(num_rows, dtype=torch.int64, device=device)
        return indices, False
    if num_rows <= 1:
        indices = torch.arange(num_rows, dtype=torch.int64, device=device)
        return indices, True

    key_scale = 1 << key_bits
    flat_view, remap_kind, key_offset = _remap_info(flat)
    indices = None
    group_id = None
    all_unique = False
    for col in range(num_cols):
        keys = _build_composite_key(
            flat_view,
            col,
            indices,
            group_id,
            num_rows,
            num_cols,
            key_offset,
            key_scale,
            remap_kind,
        )
        perm, sorted_keys = _argsort_keys(keys)
        indices = perm if col == 0 else _triton_gather_1d(indices, perm)
        group_id, last_group_id = _group_id_from_sorted(sorted_keys)
        if last_group_id == num_rows - 1:
            all_unique = True
            break
    return indices, all_unique


def _lex_argsort_rows(flat: torch.Tensor) -> tuple[torch.Tensor, bool]:
    composite = _lex_argsort_rows_composite(flat)
    if composite is not None:
        return composite
    return _lex_argsort_rows_cascade(flat), False


def unique_dim(
    input: torch.Tensor,
    dim: int,
    sorted: bool = True,
    return_inverse: bool = False,
    return_counts: bool = False,
):
    logger.debug("GEMS_ASCEND UNIQUE_DIM")

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
        return input.clone(), _empty_int64(device), _empty_int64(device)

    moved = input.movedim(dim, 0).contiguous()
    flat = moved.reshape(size_dim, -1)
    sorted_indices, all_unique = _lex_argsort_rows(flat)

    inverse_indices = _empty_int64(device)
    counts = _empty_int64(device)
    if all_unique:
        if return_counts:
            counts = torch.ones(size_dim, dtype=torch.int64, device=device)
        if return_inverse:
            inverse_indices = _unique_dim_inverse_from_permutation(sorted_indices)
        output = _unique_dim_gather_output(moved, sorted_indices, dim, input.shape)
        return output, inverse_indices, counts

    is_first = _unique_dim_first_mask(flat, sorted_indices)
    if return_inverse:
        unique_in_orig, inverse_indices = _unique_dim_unique_indices_and_inverse(
            sorted_indices,
            is_first,
        )
    else:
        unique_in_orig = _unique_dim_unique_indices(sorted_indices, is_first)

    if return_counts:
        counts = _unique_dim_counts(is_first, size_dim)

    output = _unique_dim_gather_output(moved, unique_in_orig, dim, input.shape)
    return output, inverse_indices, counts

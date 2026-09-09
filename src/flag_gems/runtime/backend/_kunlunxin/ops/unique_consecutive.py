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

logger = logging.getLogger(__name__)

_BLOCK_SIZE = 1024

# The generic implementation drives everything through masked loads with
# `other=` fallbacks and data-dependent masked stores. On this backend masked
# loads leak whatever really sits at the masked address instead of `other`,
# which turned the scatter index `out_idx` into garbage; the subsequent
# out-of-bounds stores wedged the NOC (the large path hung the device, the
# small path produced mismatched outputs). This override recomposes
# unique_consecutive from primitives already proven on this backend:
#
#   1. a clamped elementwise kernel producing "is first of its run" flags,
#   2. torch.cumsum for the inverse mapping,
#   3. torch.nonzero (vendored, clamp-based) for the run-start positions,
#   4. a clamped gather kernel for the output values,
#   5. a clamped neighbour-difference kernel for the counts.
#
# All kernels clamp indices instead of masking: tail lanes redundantly rewrite
# the last valid element, because masked stores past the end of a small buffer
# are also unreliable here.


@triton.jit
def _run_start_flags_kernel(
    data,
    flags,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets_c = tl.minimum(offsets, numel - 1)
    cur = tl.load(data + offsets_c)
    prev = tl.load(data + tl.maximum(offsets_c - 1, 0))
    is_start = (offsets_c == 0) | (cur != prev)
    tl.store(flags + offsets_c, is_start.to(tl.int64))


@triton.jit
def _gather_1d_kernel(
    src,
    indices,
    out,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets_c = tl.minimum(offsets, numel - 1)
    idx = tl.load(indices + offsets_c)
    values = tl.load(src + idx)
    tl.store(out + offsets_c, values)


@triton.jit
def _sub_one_kernel(
    src,
    dst,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    # `cumsum_result - 1` cannot go through the intercepted torch.sub here:
    # the gems sub computes int64 through float32 on this backend, so values
    # past 2**24 come back rounded (first bad element is exactly 16777217).
    # Out-of-place so the clamped tail lanes stay idempotent.
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets_c = tl.minimum(offsets, numel - 1)
    values = tl.load(src + offsets_c)
    tl.store(dst + offsets_c, values - 1)


@triton.jit
def _counts_clamped_kernel(
    first_positions,
    counts,
    num_rows,
    num_unique,
    BLOCK_SIZE: tl.constexpr,
):
    # counts[i] = first_positions[i + 1] - first_positions[i], with the last
    # neighbour replaced by num_rows via tl.where instead of a masked load
    # with `other=num_rows` (which leaks past-the-end memory on this backend).
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets_c = tl.minimum(offsets, num_unique - 1)
    positions = tl.load(first_positions + offsets_c)
    next_loaded = tl.load(first_positions + tl.minimum(offsets_c + 1, num_unique - 1))
    is_last = (offsets_c + 1) >= num_unique
    end = tl.where(is_last, positions * 0 + num_rows, next_loaded)
    tl.store(counts + offsets_c, end - positions)


def unique_consecutive(
    input: torch.Tensor,
    return_inverse: bool = False,
    return_counts: bool = False,
    dim: int = None,
):
    logger.debug("GEMS_KUNLUNXIN UNIQUE_CONSECUTIVE")
    if dim is not None:
        # Same fallback as the generic implementation.
        return torch.unique_consecutive(
            input,
            return_inverse=return_inverse,
            return_counts=return_counts,
            dim=dim,
        )

    flat = input.ravel()
    num_tasks = flat.numel()
    device = input.device

    if num_tasks == 0:
        output = torch.empty(0, dtype=input.dtype, device=device)
        inverse_indices = (
            torch.empty(0, dtype=torch.int64, device=device) if return_inverse else None
        )
        counts = (
            torch.empty(0, dtype=torch.int64, device=device) if return_counts else None
        )
        return output, inverse_indices, counts

    flags = torch.empty(num_tasks, dtype=torch.int64, device=device)
    grid_in = (triton.cdiv(num_tasks, _BLOCK_SIZE),)
    with torch_device_fn.device(device.index):
        _run_start_flags_kernel[grid_in](
            flat,
            flags,
            num_tasks,
            BLOCK_SIZE=_BLOCK_SIZE,
        )

    inverse_indices = None
    if return_inverse:
        group_numbers = torch.cumsum(flags, dim=0)
        inverse_flat = torch.empty(num_tasks, dtype=torch.int64, device=device)
        with torch_device_fn.device(device.index):
            _sub_one_kernel[grid_in](
                group_numbers,
                inverse_flat,
                num_tasks,
                BLOCK_SIZE=_BLOCK_SIZE,
            )
        inverse_indices = inverse_flat.view_as(input)

    first_positions = torch.nonzero(flags, as_tuple=False).flatten()
    out_size = first_positions.numel()

    output = torch.empty(out_size, dtype=input.dtype, device=device)
    grid_out = (triton.cdiv(out_size, _BLOCK_SIZE),)
    with torch_device_fn.device(device.index):
        _gather_1d_kernel[grid_out](
            flat,
            first_positions,
            output,
            out_size,
            BLOCK_SIZE=_BLOCK_SIZE,
        )

    counts = None
    if return_counts:
        counts = torch.empty(out_size, dtype=torch.int64, device=device)
        with torch_device_fn.device(device.index):
            _counts_clamped_kernel[grid_out](
                first_positions,
                counts,
                num_tasks,
                out_size,
                BLOCK_SIZE=_BLOCK_SIZE,
            )

    return output, inverse_indices, counts

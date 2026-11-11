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

"""MUSA-specialized ``unsafe_index_put`` (aten._unsafe_index_put).

The generic implementation normalizes a full-shape boolean mask with
``nonzero`` and then scatters the resulting integer coordinates: K is only
known after the scan, so the call synchronizes the host, materialises int64
coordinate tensors and launches the scatter kernel on a data-dependent grid.

This variant short-circuits that shape with a fused write pass:
``out[i] = mask[i] ? values[rank(i)] : inp[i]`` (or ``inp[i] + values[rank(i)]``
when accumulating), where ``rank(i)`` is the row-major rank of ``i`` among the
True elements. The only cross-thread dependency is a prefix sum of per-block
True counts, which two small kernels produce before the write pass. There is
no host sync, no int64 coordinate tensor and no atomics.

Everything else (integer indices, partial masks, non-contiguous inputs,
broadcast values, dtype mismatches) keeps the generic implementation.
"""

import logging
import operator

import torch
import triton
import triton.language as tl

from flag_gems.ops.unsafe_index_put import unsafe_index_put as _generic_unsafe_index_put
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.codegen_config_utils import (
    get_codegen_config,
    get_heuristics_for_num_warps,
)

logger = logging.getLogger(__name__)

_INT32_MAX = (1 << 31) - 1
_PORTABLE_GRID_LIMIT = 65535

_MASK_BLOCK_SIZE = 2048
_MASK_SCAN_BLOCK = 2048
_MASK_SINGLE_BLOCK_LIMIT = 4096


@triton.jit(do_not_specialize=["n_elements", "n_values"])
def _mask_put_single_block_kernel(
    inp_ptr,
    mask_ptr,
    values_ptr,
    out_ptr,
    n_elements,
    n_values,
    ACCUMULATE: tl.constexpr,
    VALUES_SCALAR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    valid = offsets < n_elements
    selected = tl.load(mask_ptr + offsets, mask=valid, other=0) != 0
    original = tl.load(inp_ptr + offsets, mask=valid, other=0)
    if VALUES_SCALAR:
        value = tl.load(values_ptr)
        updated = original + value if ACCUMULATE else value
    else:
        rank = tl.cumsum(selected.to(tl.int32), axis=0) - 1
        # Clamp keeps out-of-range reads in bounds if values is shorter than K.
        index = tl.minimum(rank, n_values - 1)
        value = tl.load(values_ptr + index, mask=valid & selected, other=0)
        updated = original + value if ACCUMULATE else value
    tl.store(out_ptr + offsets, tl.where(selected, updated, original), mask=valid)


@triton.jit(do_not_specialize=["n_elements", "n_blocks"])
def _mask_count_kernel(
    mask_ptr,
    counts_ptr,
    n_elements,
    n_blocks,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    blocks_per_program = n_blocks // num_programs
    extra = n_blocks - blocks_per_program * num_programs
    block_start = pid * blocks_per_program + tl.minimum(pid, extra)
    block_end = block_start + blocks_per_program + tl.where(pid < extra, 1, 0)
    for block_id in range(block_start, block_end):
        offsets = block_id * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < n_elements
        selected = tl.load(mask_ptr + offsets, mask=valid, other=0) != 0
        tl.store(counts_ptr + block_id, tl.sum(selected.to(tl.int32), axis=0))


@triton.jit(do_not_specialize=["n_blocks"])
def _mask_scan_kernel(
    counts_ptr,
    offsets_ptr,
    n_blocks,
    BLOCK: tl.constexpr,
):
    carry = tl.zeros((1,), dtype=tl.int32)
    for start in range(0, n_blocks, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        valid = offsets < n_blocks
        counts = tl.load(counts_ptr + offsets, mask=valid, other=0)
        tl.store(
            offsets_ptr + offsets,
            tl.cumsum(counts, axis=0) - counts + carry,
            mask=valid,
        )
        carry = carry + tl.sum(counts, axis=0)


@triton.jit(do_not_specialize=["n_elements", "n_blocks", "n_values"])
def _mask_put_kernel(
    inp_ptr,
    mask_ptr,
    values_ptr,
    offsets_ptr,
    out_ptr,
    n_elements,
    n_blocks,
    n_values,
    ACCUMULATE: tl.constexpr,
    VALUES_SCALAR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    blocks_per_program = n_blocks // num_programs
    extra = n_blocks - blocks_per_program * num_programs
    block_start = pid * blocks_per_program + tl.minimum(pid, extra)
    block_end = block_start + blocks_per_program + tl.where(pid < extra, 1, 0)
    for block_id in range(block_start, block_end):
        offsets = block_id * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < n_elements
        selected = tl.load(mask_ptr + offsets, mask=valid, other=0) != 0
        original = tl.load(inp_ptr + offsets, mask=valid, other=0)
        if VALUES_SCALAR:
            value = tl.load(values_ptr)
            updated = original + value if ACCUMULATE else value
        else:
            rank = tl.cumsum(selected.to(tl.int32), axis=0) - 1
            base = tl.load(offsets_ptr + block_id)
            # Clamp keeps out-of-range reads in bounds if values is shorter than K.
            index = tl.minimum(base + rank, n_values - 1)
            value = tl.load(values_ptr + index, mask=valid & selected, other=0)
            updated = original + value if ACCUMULATE else value
        tl.store(out_ptr + offsets, tl.where(selected, updated, original), mask=valid)


def _mask_grid_limit():
    config = get_codegen_config()
    if config is None or not config.max_grid_size:
        return _PORTABLE_GRID_LIMIT
    return min(operator.index(config.max_grid_size[0]), _PORTABLE_GRID_LIMIT)


def _mask_num_warps(block):
    # A MUSA CTA is capped at 4 warps (128 threads); the shared heuristic
    # already encodes the MCU policy, so no extra clamp is needed here.
    return max(1, get_heuristics_for_num_warps(block))


def _can_use_mask_kernel(inp, indices, values):
    """Full-shape bool mask, contiguous operands, vector values.

    Everything else (integer indices, partial masks, non-contiguous inputs,
    broadcast values) keeps the generic path, which already handles it.
    """
    if not isinstance(inp, torch.Tensor) or not isinstance(values, torch.Tensor):
        return False
    if len(indices) != 1:
        return False
    mask = indices[0]
    if not isinstance(mask, torch.Tensor):
        return False
    if mask.dtype not in (torch.bool, torch.int8):
        return False
    if inp.ndim == 0 or tuple(mask.shape) != tuple(inp.shape):
        return False
    if mask.device != inp.device or values.device != inp.device:
        return False
    if not inp.is_contiguous() or not mask.is_contiguous():
        return False
    if values.dtype != inp.dtype or values.numel() == 0:
        return False
    if values.numel() != 1 and (values.dim() != 1 or not values.is_contiguous()):
        return False
    numel = inp.numel()
    if numel == 0 or numel > _INT32_MAX or values.numel() > _INT32_MAX:
        return False
    return True


def _masked_index_put(inp, mask, values, accumulate):
    n_elements = inp.numel()
    values_scalar = values.numel() == 1

    with torch_device_fn.device(inp.device):
        out = torch.empty_like(inp)

        if n_elements <= _MASK_SINGLE_BLOCK_LIMIT:
            block = triton.next_power_of_2(n_elements)
            _mask_put_single_block_kernel[(1,)](
                inp,
                mask,
                values,
                out,
                n_elements,
                values.numel(),
                ACCUMULATE=accumulate,
                VALUES_SCALAR=values_scalar,
                BLOCK=block,
                num_warps=_mask_num_warps(block),
            )
            return out

        n_blocks = triton.cdiv(n_elements, _MASK_BLOCK_SIZE)
        grid = min(n_blocks, _mask_grid_limit())
        counts = torch.empty(n_blocks, dtype=torch.int32, device=inp.device)
        offsets = torch.empty(n_blocks, dtype=torch.int32, device=inp.device)

        _mask_count_kernel[(grid,)](
            mask,
            counts,
            n_elements,
            n_blocks,
            BLOCK=_MASK_BLOCK_SIZE,
            num_warps=_mask_num_warps(_MASK_BLOCK_SIZE),
        )
        _mask_scan_kernel[(1,)](
            counts,
            offsets,
            n_blocks,
            BLOCK=_MASK_SCAN_BLOCK,
            num_warps=_mask_num_warps(_MASK_SCAN_BLOCK),
        )
        _mask_put_kernel[(grid,)](
            inp,
            mask,
            values,
            offsets,
            out,
            n_elements,
            n_blocks,
            values.numel(),
            ACCUMULATE=accumulate,
            VALUES_SCALAR=values_scalar,
            BLOCK=_MASK_BLOCK_SIZE,
            num_warps=_mask_num_warps(_MASK_BLOCK_SIZE),
        )

    return out


def unsafe_index_put(inp, indices, values, accumulate=False):
    logger.debug("GEMS UNSAFE_INDEX_PUT")

    indices = list(indices)
    if _can_use_mask_kernel(inp, indices, values):
        return _masked_index_put(inp, indices[0], values, accumulate)

    return _generic_unsafe_index_put(inp, indices, values, accumulate)

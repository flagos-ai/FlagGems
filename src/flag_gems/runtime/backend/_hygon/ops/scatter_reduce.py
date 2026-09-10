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

"""Hygon specializations for scatter_reduce."""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.scatter_reduce import (
    _scatter_reduce_rowwise,
    _select_rowwise_strategy,
)
from flag_gems.ops.scatter_reduce import scatter_reduce as _generic_scatter_reduce
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.triton_version_utils import _triton_version_at_least

logger = logging.getLogger(__name__)

_DIRECT_ADD_BLOCK = 256
_DIRECT_ADD_LOOP = 4
_MAX_DIRECT_ADD_ELEMENTS_PER_LAUNCH = 65535 * _DIRECT_ADD_BLOCK * _DIRECT_ADD_LOOP
_PACKED16_BLOCK = 64
_PACKED16_MAX_ROW_EXTENT = 256
_PACKED16_PRODUCT_MAX_ROW_EXTENT = 1024
_TRITON_SUPPORTS_BF16_ATOMIC_ADD = _triton_version_at_least(3, 4)
_LINK_BUILD_BLOCK = 128
_LINK_BUILD_LOOP = 4
_LINK_FINAL_BLOCK = 512
_MAX_GRID_X = 65535


def _same_tensor_mapping(lhs, rhs):
    return (
        lhs.data_ptr() == rhs.data_ptr()
        and lhs.shape == rhs.shape
        and lhs.stride() == rhs.stride()
    )


def _direct_result_is_safe(inp, src, result):
    if result is None:
        return True
    if _same_tensor_mapping(result, inp):
        return not torch._C._overlaps(result, src)
    return not torch._C._overlaps(result, inp) and not torch._C._overlaps(result, src)


@triton.jit
def hygon_scatter_reduce_prod_build_lists_kernel(
    index_ptr,
    src_ptr,
    heads_ptr,
    next_ptr,
    N,
    index_ncols,
    src_ncols,
    out_ncols,
    INCLUDE_SELF: tl.constexpr,
    BLOCK: tl.constexpr,
    LOOP: tl.constexpr,
):
    """Build one lock-free source list per output using integer exchange."""
    pid = tl.program_id(0).to(tl.int64) + tl.program_id(1).to(
        tl.int64
    ) * tl.num_programs(0)
    lanes = tl.arange(0, BLOCK).to(tl.int64)
    base = pid * (BLOCK * LOOP) + lanes

    for loop_idx in range(LOOP):
        offsets = base + loop_idx * BLOCK
        mask = offsets < N
        row = offsets // index_ncols
        col = offsets % index_ncols
        source = tl.load(
            src_ptr + row * src_ncols + col,
            mask=mask,
            other=1.0,
        ).to(tl.float32)
        if INCLUDE_SELF:
            changes_value = source != 1.0
        else:
            changes_value = mask
        active = mask & changes_value
        index = tl.load(index_ptr + offsets, mask=active, other=0).to(tl.int64)
        out_offsets = row * out_ncols + index
        previous = tl.atomic_xchg(
            heads_ptr + out_offsets,
            offsets.to(tl.int32),
            mask=active,
            sem="relaxed",
        )
        tl.store(next_ptr + offsets, previous, mask=active)


@triton.jit
def hygon_scatter_reduce_prod_finalize_lists_kernel(
    inp_ptr,
    src_ptr,
    heads_ptr,
    next_ptr,
    result_ptr,
    out_numel,
    index_ncols,
    src_ncols,
    INCLUDE_SELF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Traverse independent source lists and multiply each source exactly once."""
    pid = tl.program_id(0).to(tl.int64) + tl.program_id(1).to(
        tl.int64
    ) * tl.num_programs(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offsets < out_numel
    node = tl.load(heads_ptr + offsets, mask=mask, other=-1)
    touched = node >= 0
    if INCLUDE_SELF:
        value = tl.load(inp_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    else:
        value = tl.full((BLOCK,), 1.0, tl.float32)

    active = mask & touched
    done = ~active
    all_done = False
    while not all_done:
        safe_node = tl.where(active, node, 0).to(tl.int64)
        row = safe_node // index_ncols
        col = safe_node % index_ncols
        source = tl.load(
            src_ptr + row * src_ncols + col,
            mask=active,
            other=1.0,
        ).to(tl.float32)
        value = tl.where(active, value * source, value)
        node = tl.load(next_ptr + safe_node, mask=active, other=-1)
        active &= node >= 0
        done |= ~active
        all_done = tl.sum(done.to(tl.int32)) == BLOCK

    if not INCLUDE_SELF:
        inp = tl.load(inp_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        value = tl.where(touched, value, inp)
    tl.store(result_ptr + offsets, value, mask=mask)


@triton.jit
def hygon_scatter_reduce_packed16_kernel(
    inp_ptr,
    index_ptr,
    src_ptr,
    result_ptr,
    out_nrows,
    index_nrows,
    index_ncols: tl.constexpr,
    src_ncols: tl.constexpr,
    out_ncols: tl.constexpr,
    REDUCE: tl.constexpr,
    IS_BFLOAT16: tl.constexpr,
    INITIALIZE_RESULT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Update two adjacent 16-bit outputs through one race-safe int32 CAS."""
    row = tl.program_id(0).to(tl.int64) + tl.program_id(1).to(
        tl.int64
    ) * tl.num_programs(0)
    lanes = tl.arange(0, BLOCK)
    if INITIALIZE_RESULT:
        for start in tl.range(0, out_ncols, BLOCK):
            out_cols = start + lanes
            out_mask = (row < out_nrows) & (out_cols < out_ncols)
            inp = tl.load(
                inp_ptr + row * out_ncols + out_cols,
                mask=out_mask,
            )
            tl.store(
                result_ptr + row * out_ncols + out_cols,
                inp,
                mask=out_mask,
            )
        tl.debug_barrier()

    for start in tl.range(0, index_ncols, BLOCK):
        col = start + lanes
        mask = (row < index_nrows) & (col < index_ncols)
        index_offsets = row * index_ncols + col
        index = tl.load(index_ptr + index_offsets, mask=mask, other=0).to(tl.int64)
        source = tl.load(
            src_ptr + row * src_ncols + col,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        out_offsets = row * out_ncols + index
        word_offsets = out_offsets // 2
        upper = (out_offsets & 1) != 0
        word_ptr = result_ptr.to(tl.pointer_type(tl.int32, 1), bitcast=True)
        word_ptr += word_offsets

        done = tl.where(mask, 0, 1).to(tl.int1)
        block_done = False
        while not block_done:
            current_word = tl.load(word_ptr, mask=mask, other=0)
            current_bits = tl.where(
                upper,
                (current_word >> 16) & 0xFFFF,
                current_word & 0xFFFF,
            ).to(tl.int16)
            if IS_BFLOAT16:
                current = current_bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
            else:
                current = current_bits.to(tl.float16, bitcast=True).to(tl.float32)

            if REDUCE == 0:
                updated = current + source
            elif REDUCE == 1:
                updated = current * source
            elif REDUCE == 3:
                updated = tl.maximum(current, source)
            else:
                updated = tl.minimum(current, source)
            updated = tl.where(done, current, updated)
            if IS_BFLOAT16:
                updated_bits = updated.to(tl.bfloat16).to(tl.int16, bitcast=True)
            else:
                updated_bits = updated.to(tl.float16).to(tl.int16, bitcast=True)
            updated_bits = updated_bits.to(tl.int32) & 0xFFFF
            updated_word = tl.where(
                upper,
                (current_word & 0xFFFF) | (updated_bits << 16),
                (current_word & -65536) | updated_bits,
            )
            previous_word = tl.atomic_cas(
                word_ptr,
                current_word,
                updated_word,
                sem="acq_rel",
            )
            done |= current_word == previous_word
            block_done = tl.sum(done.to(tl.int32)) == BLOCK


@triton.jit
def hygon_scatter_reduce_fp16_add_kernel(
    inp_ptr,
    index_ptr,
    src_ptr,
    result_ptr,
    out_nrows,
    index_nrows,
    index_ncols: tl.constexpr,
    src_ncols: tl.constexpr,
    out_ncols: tl.constexpr,
    INITIALIZE_RESULT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Use HCU's native fp16 add for the short-row include-self fast path."""
    row = tl.program_id(0).to(tl.int64) + tl.program_id(1).to(
        tl.int64
    ) * tl.num_programs(0)
    lanes = tl.arange(0, BLOCK)
    if INITIALIZE_RESULT:
        for start in tl.range(0, out_ncols, BLOCK):
            out_cols = start + lanes
            out_mask = (row < out_nrows) & (out_cols < out_ncols)
            inp = tl.load(
                inp_ptr + row * out_ncols + out_cols,
                mask=out_mask,
            )
            tl.store(
                result_ptr + row * out_ncols + out_cols,
                inp,
                mask=out_mask,
            )
        tl.debug_barrier()

    for start in tl.range(0, index_ncols, BLOCK):
        col = start + lanes
        mask = (row < index_nrows) & (col < index_ncols)
        index_offsets = row * index_ncols + col
        index = tl.load(index_ptr + index_offsets, mask=mask, other=0).to(tl.int64)
        source = tl.load(
            src_ptr + row * src_ncols + col,
            mask=mask,
            other=0.0,
        )
        tl.atomic_add(
            result_ptr + row * out_ncols + index,
            source,
            mask=mask,
            sem="relaxed",
        )


@triton.jit
def hygon_scatter_reduce_direct_add_kernel(
    index_ptr,
    src_ptr,
    result_ptr,
    N,
    row_start,
    index_ncols,
    src_ncols,
    out_ncols,
    BLOCK: tl.constexpr,
    LOOP: tl.constexpr,
):
    """Accumulate directly into an alias-safe in-place half-precision result."""
    pid = tl.program_id(0)
    lanes = tl.arange(0, BLOCK)
    base_offsets = pid * BLOCK * LOOP + lanes

    for loop_idx in range(LOOP):
        offsets = (base_offsets + loop_idx * BLOCK).to(tl.int64)
        mask = offsets < N
        row = offsets // index_ncols
        col = offsets % index_ncols
        index = tl.load(index_ptr + offsets, mask=mask, other=0).to(tl.int64)
        source = tl.load(
            src_ptr + row * src_ncols + col,
            mask=mask,
            other=0.0,
        )
        out_offsets = (row + row_start) * out_ncols + index
        tl.atomic_add(
            result_ptr + out_offsets,
            source,
            mask=mask,
            sem="relaxed",
        )


@triton.jit
def hygon_scatter_reduce_packed16_false_kernel(
    inp_ptr,
    index_ptr,
    src_ptr,
    accumulator_ptr,
    touched_ptr,
    result_ptr,
    out_nrows,
    index_nrows,
    index_ncols: tl.constexpr,
    src_ncols: tl.constexpr,
    out_ncols: tl.constexpr,
    REDUCE: tl.constexpr,
    IS_BFLOAT16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Reduce short rows without retaining a full-size FP32 accumulator."""
    row = tl.program_id(0).to(tl.int64) + tl.program_id(1).to(
        tl.int64
    ) * tl.num_programs(0)
    lanes = tl.arange(0, BLOCK)

    for start in tl.range(0, out_ncols, BLOCK):
        out_cols = start + lanes
        out_mask = (row < out_nrows) & (out_cols < out_ncols)
        if REDUCE == 0:
            initial = 0.0
        elif REDUCE == 1:
            initial = 1.0
        elif REDUCE == 3:
            initial = float("-inf")
        else:
            initial = float("inf")
        out_offsets = row * out_ncols + out_cols
        tl.store(accumulator_ptr + out_offsets, initial, mask=out_mask)
        tl.store(touched_ptr + out_offsets, 0, mask=out_mask)

    tl.debug_barrier()

    for start in tl.range(0, index_ncols, BLOCK):
        col = start + lanes
        mask = (row < index_nrows) & (col < index_ncols)
        index_offsets = row * index_ncols + col
        index = tl.load(index_ptr + index_offsets, mask=mask, other=0).to(tl.int64)
        source_value = tl.load(
            src_ptr + row * src_ncols + col,
            mask=mask,
            other=0.0,
        )
        source = source_value.to(tl.float32)
        out_offsets = row * out_ncols + index

        if REDUCE == 0 and not IS_BFLOAT16:
            tl.atomic_add(
                accumulator_ptr + out_offsets,
                source_value,
                mask=mask,
                sem="relaxed",
            )
        else:
            word_offsets = out_offsets // 2
            upper = (out_offsets & 1) != 0
            word_ptr = accumulator_ptr.to(tl.pointer_type(tl.int32, 1), bitcast=True)
            word_ptr += word_offsets
            done = tl.where(mask, 0, 1).to(tl.int1)
            block_done = False
            while not block_done:
                current_word = tl.load(word_ptr, mask=mask, other=0)
                current_bits = tl.where(
                    upper,
                    (current_word >> 16) & 0xFFFF,
                    current_word & 0xFFFF,
                ).to(tl.int16)
                if IS_BFLOAT16:
                    current = current_bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
                else:
                    current = current_bits.to(tl.float16, bitcast=True).to(tl.float32)
                if REDUCE == 0:
                    updated = current + source
                elif REDUCE == 1:
                    updated = current * source
                elif REDUCE == 3:
                    updated = tl.maximum(current, source)
                else:
                    updated = tl.minimum(current, source)
                updated = tl.where(done, current, updated)
                if IS_BFLOAT16:
                    updated_bits = updated.to(tl.bfloat16).to(tl.int16, bitcast=True)
                else:
                    updated_bits = updated.to(tl.float16).to(tl.int16, bitcast=True)
                updated_bits = updated_bits.to(tl.int32) & 0xFFFF
                updated_word = tl.where(
                    upper,
                    (current_word & 0xFFFF) | (updated_bits << 16),
                    (current_word & -65536) | updated_bits,
                )
                previous_word = tl.atomic_cas(
                    word_ptr,
                    current_word,
                    updated_word,
                    sem="acq_rel",
                )
                done |= current_word == previous_word
                block_done = tl.sum(done.to(tl.int32)) == BLOCK
        tl.atomic_or(
            touched_ptr + out_offsets,
            1,
            mask=mask,
            sem="relaxed",
        )

    tl.debug_barrier()

    for start in tl.range(0, out_ncols, BLOCK):
        out_cols = start + lanes
        out_mask = (row < out_nrows) & (out_cols < out_ncols)
        out_offsets = row * out_ncols + out_cols
        reduced = tl.load(accumulator_ptr + out_offsets, mask=out_mask, other=0.0)
        original = tl.load(inp_ptr + out_offsets, mask=out_mask, other=0.0)
        touched = tl.load(touched_ptr + out_offsets, mask=out_mask, other=0)
        result = tl.where(touched != 0, reduced, original)
        tl.store(result_ptr + out_offsets, result, mask=out_mask)


def _can_use_packed16(inp, dim, index, src, reduce, include_self, result=None):
    row_extent = max(inp.shape[1], index.shape[1]) if inp.ndim == 2 else 0
    max_row_extent = (
        _PACKED16_PRODUCT_MAX_ROW_EXTENT
        if reduce == "prod"
        else _PACKED16_MAX_ROW_EXTENT
    )
    if (
        reduce not in ("sum", "prod", "amax", "amin")
        or inp.ndim != 2
        or dim not in (-1, 1)
        or inp.dtype not in (torch.float16, torch.bfloat16)
        or src.dtype != inp.dtype
        or index.dtype != torch.int64
        or not inp.is_contiguous()
        or not index.is_contiguous()
        or not src.is_contiguous()
        or inp.numel() == 0
        or index.numel() == 0
        or index.shape[0] > inp.shape[0]
        or index.shape[0] > src.shape[0]
        or index.shape[1] > src.shape[1]
        or inp.shape[1] % 2 != 0
        or row_extent <= 64
        or row_extent > max_row_extent
    ):
        return False
    if result is None:
        return True
    return (
        result.shape == inp.shape
        and result.dtype == inp.dtype
        and result.device == inp.device
        and result.is_contiguous()
        and _direct_result_is_safe(inp, src, result)
    )


def _scatter_reduce_packed16(inp, index, src, reduce, include_self, result=None):
    """Run a short-row reduction using 16-bit Hygon atomics."""
    if result is None:
        result = torch.empty_like(inp)
    if not include_self:
        scratch = torch.empty(
            (3, *inp.shape),
            dtype=inp.dtype,
            device=inp.device,
        )
        accumulator = scratch[0]
        touched = scratch[1:].view(torch.int32).reshape(inp.shape)
        reduce_id = {"sum": 0, "prod": 1, "amax": 3, "amin": 4}[reduce]
        with torch_device_fn.device(inp.device):
            hygon_scatter_reduce_packed16_false_kernel[_split_grid(inp.shape[0])](
                inp,
                index,
                src,
                accumulator,
                touched,
                result,
                inp.shape[0],
                index.shape[0],
                index.shape[1],
                src.shape[1],
                inp.shape[1],
                reduce_id,
                inp.dtype == torch.bfloat16,
                BLOCK=_PACKED16_BLOCK,
            )
        return result

    initialize_result = result.data_ptr() != inp.data_ptr()

    reduce_id = {"sum": 0, "prod": 1, "amax": 3, "amin": 4}[reduce]
    grid = _split_grid(inp.shape[0])
    with torch_device_fn.device(inp.device):
        if reduce == "sum" and inp.dtype == torch.float16:
            hygon_scatter_reduce_fp16_add_kernel[grid](
                inp,
                index,
                src,
                result,
                inp.shape[0],
                index.shape[0],
                index.shape[1],
                src.shape[1],
                inp.shape[1],
                initialize_result,
                BLOCK=_PACKED16_BLOCK,
            )
        else:
            hygon_scatter_reduce_packed16_kernel[grid](
                inp,
                index,
                src,
                result,
                inp.shape[0],
                index.shape[0],
                index.shape[1],
                src.shape[1],
                inp.shape[1],
                reduce_id,
                inp.dtype == torch.bfloat16,
                initialize_result,
                BLOCK=_PACKED16_BLOCK,
            )
    return result


def _can_use_direct_inplace_add(inp, dim, index, src, reduce, include_self):
    return (
        reduce == "sum"
        and include_self
        and inp.ndim == 2
        and dim in (-1, 1)
        and inp.dtype in (torch.float16, torch.bfloat16)
        and (inp.dtype != torch.bfloat16 or _TRITON_SUPPORTS_BF16_ATOMIC_ADD)
        and src.dtype == inp.dtype
        and index.dtype == torch.int64
        and inp.is_contiguous()
        and index.is_contiguous()
        and src.is_contiguous()
        and index.shape[0] <= inp.shape[0]
        and index.shape[0] <= src.shape[0]
        and index.shape[1] <= src.shape[1]
        and max(inp.shape[1], index.shape[1]) > _PACKED16_MAX_ROW_EXTENT
        and inp.numel() != 0
        and index.numel() != 0
        and _direct_result_is_safe(inp, src, inp)
    )


def _scatter_reduce_direct_inplace_add(inp, index, src):
    """Avoid a full FP32 scratch/result copy for large in-place half sums."""
    rows_per_launch = max(
        1,
        _MAX_DIRECT_ADD_ELEMENTS_PER_LAUNCH // index.shape[1],
    )
    with torch_device_fn.device(inp.device):
        for row_start in range(0, index.shape[0], rows_per_launch):
            row_end = min(row_start + rows_per_launch, index.shape[0])
            index_chunk = index[row_start:row_end]
            src_chunk = src[row_start:row_end]
            chunk_numel = index_chunk.numel()
            grid = (
                triton.cdiv(
                    chunk_numel,
                    _DIRECT_ADD_BLOCK * _DIRECT_ADD_LOOP,
                ),
            )
            hygon_scatter_reduce_direct_add_kernel[grid](
                index_chunk,
                src_chunk,
                inp,
                chunk_numel,
                row_start,
                index_chunk.shape[1],
                src_chunk.shape[1],
                inp.shape[1],
                BLOCK=_DIRECT_ADD_BLOCK,
                LOOP=_DIRECT_ADD_LOOP,
            )
    return inp


def _split_grid(programs):
    return min(programs, _MAX_GRID_X), triton.cdiv(programs, _MAX_GRID_X)


def _can_use_linked_product(inp, dim, index, src, result=None):
    if (
        inp.ndim != 2
        or dim not in (-1, 1)
        or inp.dtype not in (torch.float16, torch.float32, torch.bfloat16)
        or src.dtype != inp.dtype
        or index.dtype != torch.int64
        or not inp.is_contiguous()
        or not index.is_contiguous()
        or not src.is_contiguous()
        or inp.numel() == 0
        or index.numel() == 0
        or index.numel() >= 1 << 31
        or index.shape[0] > inp.shape[0]
        or index.shape[0] > src.shape[0]
        or index.shape[1] > src.shape[1]
    ):
        return False
    if result is None:
        return True
    return (
        result.shape == inp.shape
        and result.dtype == inp.dtype
        and result.device == inp.device
        and result.is_contiguous()
        and _direct_result_is_safe(inp, src, result)
    )


def _select_hygon_rowwise_strategy(
    inp,
    dim,
    index,
    src,
    reduce,
    include_self,
    result,
):
    """Prefer row-owned CAS for medium HCU product rows."""
    if reduce == "prod" and _can_use_linked_product(inp, dim, index, src, result):
        row_extent = max(inp.shape[1], index.shape[1])
        if 64 < row_extent <= 1024 and (
            inp.dtype == torch.float32 or row_extent > _PACKED16_MAX_ROW_EXTENT
        ):
            return "atomic"
    return _select_rowwise_strategy(
        inp,
        dim,
        index,
        src,
        reduce,
        include_self,
        result,
    )


def _scatter_reduce_linked_product(inp, index, src, include_self, result=None):
    if result is None:
        result = torch.empty_like(inp)
    heads = torch.full(inp.shape, -1, dtype=torch.int32, device=inp.device)
    next_nodes = torch.empty(index.numel(), dtype=torch.int32, device=inp.device)
    build_programs = triton.cdiv(index.numel(), _LINK_BUILD_BLOCK * _LINK_BUILD_LOOP)
    finalize_programs = triton.cdiv(inp.numel(), _LINK_FINAL_BLOCK)

    with torch_device_fn.device(inp.device):
        hygon_scatter_reduce_prod_build_lists_kernel[_split_grid(build_programs)](
            index,
            src,
            heads,
            next_nodes,
            index.numel(),
            index.shape[1],
            src.shape[1],
            inp.shape[1],
            include_self,
            BLOCK=_LINK_BUILD_BLOCK,
            LOOP=_LINK_BUILD_LOOP,
        )
        hygon_scatter_reduce_prod_finalize_lists_kernel[_split_grid(finalize_programs)](
            inp,
            src,
            heads,
            next_nodes,
            result,
            inp.numel(),
            index.shape[1],
            src.shape[1],
            include_self,
            BLOCK=_LINK_FINAL_BLOCK,
        )
    return result


def _scatter_reduce(inp, dim, index, src, reduce, include_self):
    if inp.numel() == 0 or index.numel() == 0:
        return _generic_scatter_reduce(
            inp,
            dim,
            index,
            src,
            reduce,
            include_self=include_self,
        )
    if _can_use_packed16(
        inp,
        dim,
        index,
        src,
        reduce,
        include_self,
    ):
        return _scatter_reduce_packed16(inp, index, src, reduce, include_self)
    rowwise_strategy = _select_hygon_rowwise_strategy(
        inp,
        dim,
        index,
        src,
        reduce,
        include_self,
        None,
    )
    if rowwise_strategy is not None:
        return _scatter_reduce_rowwise(
            inp,
            index,
            src,
            reduce,
            include_self,
            rowwise_strategy,
        )
    if reduce == "prod" and _can_use_linked_product(inp, dim, index, src):
        return _scatter_reduce_linked_product(inp, index, src, include_self)
    return _generic_scatter_reduce(
        inp,
        dim,
        index,
        src,
        reduce,
        include_self=include_self,
    )


def scatter_reduce(inp, dim, index, src, reduce, *, include_self=True):
    logger.debug("GEMS_HYGON SCATTER_REDUCE_TWO")
    return _scatter_reduce(inp, dim, index, src, reduce, include_self)


def scatter_reduce_(inp, dim, index, src, reduce, *, include_self=True):
    logger.debug("GEMS_HYGON SCATTER_REDUCE_TWO_")
    if _can_use_direct_inplace_add(
        inp,
        dim,
        index,
        src,
        reduce,
        include_self,
    ):
        return _scatter_reduce_direct_inplace_add(inp, index, src)
    if _can_use_packed16(
        inp,
        dim,
        index,
        src,
        reduce,
        include_self,
        inp,
    ):
        return _scatter_reduce_packed16(
            inp,
            index,
            src,
            reduce,
            include_self,
            result=inp,
        )
    rowwise_strategy = None
    if (
        inp.numel() != 0
        and index.numel() != 0
        and _direct_result_is_safe(inp, src, inp)
    ):
        rowwise_strategy = _select_hygon_rowwise_strategy(
            inp,
            dim,
            index,
            src,
            reduce,
            include_self,
            inp,
        )
    if rowwise_strategy is not None:
        return _scatter_reduce_rowwise(
            inp,
            index,
            src,
            reduce,
            include_self,
            rowwise_strategy,
            result=inp,
        )
    if reduce == "prod" and _can_use_linked_product(inp, dim, index, src, inp):
        return _scatter_reduce_linked_product(
            inp,
            index,
            src,
            include_self,
            result=inp,
        )
    result = _scatter_reduce(inp, dim, index, src, reduce, include_self)
    inp.copy_(result)
    return inp


def scatter_reduce_out(
    inp,
    dim,
    index,
    src,
    reduce,
    *,
    include_self=True,
    out=None,
):
    logger.debug("GEMS_HYGON SCATTER_REDUCE_TWO_OUT")
    if out is not None and out.dtype != inp.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {inp.dtype}, but got {out.dtype} instead"
        )
    if out is not None:
        if _can_use_packed16(
            inp,
            dim,
            index,
            src,
            reduce,
            include_self,
            out,
        ):
            return _scatter_reduce_packed16(
                inp,
                index,
                src,
                reduce,
                include_self,
                result=out,
            )
        rowwise_strategy = None
        if (
            inp.numel() != 0
            and index.numel() != 0
            and _direct_result_is_safe(inp, src, out)
        ):
            rowwise_strategy = _select_hygon_rowwise_strategy(
                inp,
                dim,
                index,
                src,
                reduce,
                include_self,
                out,
            )
        if rowwise_strategy is not None:
            return _scatter_reduce_rowwise(
                inp,
                index,
                src,
                reduce,
                include_self,
                rowwise_strategy,
                result=out,
            )
        if reduce == "prod" and _can_use_linked_product(
            inp,
            dim,
            index,
            src,
            out,
        ):
            return _scatter_reduce_linked_product(
                inp,
                index,
                src,
                include_self,
                result=out,
            )
    result = _scatter_reduce(inp, dim, index, src, reduce, include_self)
    if out is not None:
        out.copy_(result)
        return out
    return result

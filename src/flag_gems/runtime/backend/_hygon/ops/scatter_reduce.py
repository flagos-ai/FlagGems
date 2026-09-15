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
from flag_gems.utils import libentry
from flag_gems.utils.triton_version_utils import _triton_version_at_least

logger = logging.getLogger(__name__)

_DIRECT_ADD_BLOCK = 256
_DIRECT_ADD_LOOP = 4
_MAX_DIRECT_ADD_ELEMENTS_PER_LAUNCH = 65535 * _DIRECT_ADD_BLOCK * _DIRECT_ADD_LOOP
_PACKED16_BLOCK = 64
_PACKED16_MAX_ROW_EXTENT = 256
_TRITON_SUPPORTS_BF16_ATOMIC_ADD = _triton_version_at_least(3, 4)
_LINK_BUILD_BLOCK = 128
_LINK_BUILD_LOOP = 4
_LINK_FINAL_BLOCK = 512
_MAX_GRID_X = 65535
# Match the compiled-launch argument ABI used by LibEntry for Triton 3.6.
_CACHE_LIST_LAUNCHERS = _triton_version_at_least(3, 6) and not _triton_version_at_least(
    3, 7
)
_LIST_LAUNCHERS = {}
_MAX_LIST_LAUNCHERS = 128


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
def _product_combine(lhs, rhs):
    return lhs * rhs


@triton.jit
def _maximum_combine(lhs, rhs):
    return tl.maximum(lhs, rhs, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _minimum_combine(lhs, rhs):
    return tl.minimum(lhs, rhs, propagate_nan=tl.PropagateNan.ALL)


@libentry()
@triton.jit(do_not_specialize=["inp_ptr", "index_ptr", "src_ptr", "result_ptr"])
def hygon_scatter_reduce_small_kernel(
    inp_ptr,
    index_ptr,
    src_ptr,
    result_ptr,
    ROWS: tl.constexpr,
    INDEX_ROWS: tl.constexpr,
    OUT_COLS: tl.constexpr,
    INDEX_COLS: tl.constexpr,
    SRC_COLS: tl.constexpr,
    REDUCE: tl.constexpr,
    INCLUDE_SELF: tl.constexpr,
    OUT_BLOCK: tl.constexpr,
    INDEX_BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) + tl.program_id(1).to(
        tl.int64
    ) * tl.num_programs(0)
    columns = tl.arange(0, OUT_BLOCK)
    sources = tl.arange(0, INDEX_BLOCK)
    out_mask = (row < ROWS) & (columns < OUT_COLS)
    source_mask = (row < INDEX_ROWS) & (sources < INDEX_COLS)
    index = tl.load(index_ptr + row * INDEX_COLS + sources, source_mask, other=-1)
    source = tl.load(src_ptr + row * SRC_COLS + sources, source_mask, other=1).to(
        tl.float32
    )
    matches = (columns[:, None] == index[None, :]) & source_mask[None, :]
    original = tl.load(inp_ptr + row * OUT_COLS + columns, out_mask, other=1).to(
        tl.float32
    )
    if REDUCE == "prod":
        reduced = tl.reduce(
            tl.where(matches, source[None, :], 1.0), 1, _product_combine
        )
        if INCLUDE_SELF:
            reduced *= original
    elif REDUCE == "sum" or REDUCE == "mean":
        reduced = tl.sum(tl.where(matches, source[None, :], 0.0), 1)
        if INCLUDE_SELF:
            reduced += original
    elif REDUCE == "amax":
        reduced = tl.reduce(
            tl.where(matches, source[None, :], float("-inf")), 1, _maximum_combine
        )
        if INCLUDE_SELF:
            reduced = _maximum_combine(reduced, original)
    else:
        reduced = tl.reduce(
            tl.where(matches, source[None, :], float("inf")), 1, _minimum_combine
        )
        if INCLUDE_SELF:
            reduced = _minimum_combine(reduced, original)
    if REDUCE == "mean" or not INCLUDE_SELF:
        count = tl.sum(matches.to(tl.int32), 1)
        if REDUCE == "mean":
            denominator = count + 1 if INCLUDE_SELF else count
            reduced /= tl.maximum(denominator, 1)
        if not INCLUDE_SELF:
            reduced = tl.where(count != 0, reduced, original)
    tl.store(result_ptr + row * OUT_COLS + columns, reduced, out_mask)


@libentry()
@triton.jit(
    do_not_specialize=["inp_ptr", "index_ptr", "src_ptr", "scratch_ptr", "result_ptr"]
)
def hygon_scatter_reduce_row_atomic_kernel(
    inp_ptr,
    index_ptr,
    src_ptr,
    scratch_ptr,
    result_ptr,
    ROWS: tl.constexpr,
    INDEX_ROWS: tl.constexpr,
    OUT_COLS: tl.constexpr,
    INDEX_COLS: tl.constexpr,
    SRC_COLS: tl.constexpr,
    REDUCE: tl.constexpr,
    INCLUDE_SELF: tl.constexpr,
    DIRECT_FP32: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) + tl.program_id(1).to(
        tl.int64
    ) * tl.num_programs(0)
    work_ptr = scratch_ptr.to(tl.pointer_type(tl.int32))
    accumulator_ptr = (
        result_ptr.to(tl.pointer_type(tl.int32)) if DIRECT_FP32 else work_ptr
    )
    count_offset = 0 if DIRECT_FP32 else ROWS * OUT_COLS
    touched_ptr = work_ptr + count_offset
    lanes = tl.arange(0, BLOCK)
    for start in range(0, OUT_COLS, BLOCK):
        columns = start + lanes
        mask = (row < ROWS) & (columns < OUT_COLS)
        if INCLUDE_SELF:
            initial = tl.load(inp_ptr + row * OUT_COLS + columns, mask, other=0).to(
                tl.float32
            )
        else:
            initial = tl.full(
                (BLOCK,),
                (
                    0.0
                    if REDUCE == "sum"
                    else (float("-inf") if REDUCE == "amax" else float("inf"))
                ),
                tl.float32,
            )
        tl.store(
            accumulator_ptr + row * OUT_COLS + columns,
            initial.to(tl.int32, bitcast=True),
            mask,
        )
        if not INCLUDE_SELF:
            tl.store(touched_ptr + row * OUT_COLS + columns, 0, mask)
    tl.debug_barrier()
    for start in range(0, INDEX_COLS, BLOCK):
        columns = start + lanes
        mask = (row < INDEX_ROWS) & (columns < INDEX_COLS)
        index = tl.load(index_ptr + row * INDEX_COLS + columns, mask, other=0).to(
            tl.int64
        )
        source = tl.load(src_ptr + row * SRC_COLS + columns, mask, other=0).to(
            tl.float32
        )
        # CAS has no mask. Inactive lanes perform a no-op on their own row,
        # without concentrating all tail lanes on a shared scratch location.
        ptr = accumulator_ptr + row * OUT_COLS + tl.where(mask, index, lanes % OUT_COLS)
        expected = tl.load(ptr, mask, other=0)
        pending = mask
        while tl.sum(pending.to(tl.int32), 0) > 0:
            current = expected.to(tl.float32, bitcast=True)
            if REDUCE == "sum":
                updated = current + source
            elif REDUCE == "amax":
                updated = _maximum_combine(current, source)
            else:
                updated = _minimum_combine(current, source)
            bits = tl.where(pending, updated.to(tl.int32, bitcast=True), expected)
            observed = tl.atomic_cas(ptr, expected, bits, sem="relaxed")
            pending &= observed != expected
            # Retry against the value observed by CAS, not a repeated load.
            expected = observed
        if not INCLUDE_SELF:
            tl.atomic_xchg(touched_ptr + row * OUT_COLS + index, 1, mask, sem="relaxed")
    tl.debug_barrier()
    for start in range(0, OUT_COLS, BLOCK):
        columns = start + lanes
        mask = (row < ROWS) & (columns < OUT_COLS)
        bits = tl.load(accumulator_ptr + row * OUT_COLS + columns, mask, other=0)
        value = bits.to(tl.float32, bitcast=True)
        if not INCLUDE_SELF:
            touched = tl.load(touched_ptr + row * OUT_COLS + columns, mask, other=0)
            original = tl.load(inp_ptr + row * OUT_COLS + columns, mask, other=0).to(
                tl.float32
            )
            value = tl.where(touched != 0, value, original)
        tl.store(result_ptr + row * OUT_COLS + columns, value, mask)


@libentry()
@triton.jit(do_not_specialize=["heads_ptr"])
def hygon_scatter_reduce_init_heads_kernel(heads_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64) + tl.program_id(1).to(
        tl.int64
    ) * tl.num_programs(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(heads_ptr + offsets, -2, mask=offsets < N)


# HCU 3.6's buffer atomic exchange lowering references an unavailable intrinsic.
# Suppress pointer-range specialization for these kernels: this keeps global
# atomics and avoids repeated HCU pointer-range checks on the launch path.
@libentry()
@triton.jit(do_not_specialize=["index_ptr", "src_ptr", "heads_ptr", "next_ptr"])
def hygon_scatter_reduce_prod_build_lists_kernel(
    index_ptr,
    src_ptr,
    heads_ptr,
    next_ptr,
    N,
    index_ncols: tl.constexpr,
    src_ncols: tl.constexpr,
    out_ncols: tl.constexpr,
    INCLUDE_SELF: tl.constexpr,
    BLOCK: tl.constexpr,
    LOOP: tl.constexpr,
    NEXT_OFFSET: tl.constexpr = 0,
):
    """Build one lock-free source list per output using global integer exchange."""
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
        changes_value = source != 1.0
        active = mask & changes_value
        index_mask = active if INCLUDE_SELF else mask
        index = tl.load(index_ptr + offsets, mask=index_mask, other=0).to(tl.int64)
        out_offsets = row * out_ncols + index
        if not INCLUDE_SELF:
            # -2 is untouched; -1 marks identity-only updates. A real node
            # is nonnegative, so this marker never overwrites an existing list.
            tl.atomic_max(
                heads_ptr + out_offsets,
                -1,
                mask=mask & ~changes_value,
                sem="relaxed",
            )
        previous = tl.atomic_xchg(
            heads_ptr + out_offsets,
            offsets.to(tl.int32),
            mask=active,
            sem="relaxed",
        )
        tl.store(next_ptr + NEXT_OFFSET + offsets, previous, mask=active)


@libentry()
@triton.jit(
    do_not_specialize=["inp_ptr", "src_ptr", "heads_ptr", "next_ptr", "result_ptr"]
)
def hygon_scatter_reduce_prod_finalize_lists_kernel(
    inp_ptr,
    src_ptr,
    heads_ptr,
    next_ptr,
    result_ptr,
    out_numel,
    index_ncols: tl.constexpr,
    src_ncols: tl.constexpr,
    INCLUDE_SELF: tl.constexpr,
    BLOCK: tl.constexpr,
    NEXT_OFFSET: tl.constexpr = 0,
):
    """Traverse independent source lists and multiply each source exactly once."""
    pid = tl.program_id(0).to(tl.int64) + tl.program_id(1).to(
        tl.int64
    ) * tl.num_programs(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offsets < out_numel
    node = tl.load(heads_ptr + offsets, mask=mask, other=-1)
    touched = node != -2
    if INCLUDE_SELF:
        value = tl.load(inp_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    else:
        value = tl.full((BLOCK,), 1.0, tl.float32)

    active = mask & (node >= 0)
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
        node = tl.load(next_ptr + NEXT_OFFSET + safe_node, mask=active, other=-1)
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
        current_word = tl.load(word_ptr, mask=mask, other=0)
        block_done = row >= index_nrows
        while not block_done:
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
            # Retry from CAS's observed word instead of a stale loop load.
            current_word = previous_word
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
    """Accumulate directly into an alias-safe in-place result."""
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
            current_word = tl.load(word_ptr, mask=mask, other=0)
            block_done = row >= index_nrows
            while not block_done:
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
                # Preserve updates to either half of the packed word on retry.
                current_word = previous_word
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
    if (
        reduce not in ("sum", "amax", "amin")
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
        or row_extent > _PACKED16_MAX_ROW_EXTENT
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
    """Accumulate into the result without a separate initialization kernel."""
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
    return min(programs, _MAX_GRID_X), (programs + _MAX_GRID_X - 1) // _MAX_GRID_X


def _can_use_linked_product(inp, dim, index, src, result=None):
    if (
        inp.ndim not in (1, 2)
        or index.ndim != inp.ndim
        or src.ndim != inp.ndim
        or dim not in (-1, inp.ndim - 1)
        or inp.dtype not in (torch.float16, torch.float32, torch.bfloat16)
        or src.dtype != inp.dtype
        or index.dtype != torch.int64
        or not inp.is_contiguous()
        or not index.is_contiguous()
        or not src.is_contiguous()
        or inp.numel() == 0
        or index.numel() == 0
        or index.numel() >= 1 << 31
        or (inp.ndim == 2 and index.shape[0] > inp.shape[0])
        or (inp.ndim == 2 and index.shape[0] > src.shape[0])
        or index.shape[-1] > src.shape[-1]
    ):
        return False
    if result is inp:
        # Input metadata is already checked above; only source overlap remains.
        return not torch._C._overlaps(inp, src)
    if result is None:
        return True
    return (
        result.shape == inp.shape
        and result.dtype == inp.dtype
        and result.device == inp.device
        and result.is_contiguous()
        and _direct_result_is_safe(inp, src, result)
    )


def _can_use_small_reduction(inp, dim, index, src, reduce, result=None):
    return (
        reduce in ("sum", "mean", "amax", "amin")
        and inp.ndim in (1, 2)
        and index.ndim == inp.ndim
        and max(inp.shape[-1], index.shape[-1]) <= 64
        and _can_use_linked_product(inp, dim, index, src, result)
    )


def _can_use_row_atomic(inp, dim, index, src, reduce, result=None):
    return (
        (
            reduce in ("amax", "amin")
            or (
                reduce == "sum"
                and inp.dtype == torch.float32
                and result is inp
                and inp.ndim == 2
                and index.ndim == 2
                and max(inp.shape[1], index.shape[1]) <= 256
            )
        )
        and inp.ndim == 2
        and index.ndim == 2
        and 64 < max(inp.shape[1], index.shape[1]) <= 1024
        and _can_use_linked_product(inp, dim, index, src, result)
    )


def _scatter_reduce_row_atomic(inp, index, src, reduce, include_self, result=None):
    if result is None:
        result = torch.empty_like(inp)
    direct_fp32 = inp.dtype == torch.float32 and (
        include_self or result.data_ptr() != inp.data_ptr()
    )
    scratch_planes = int(not direct_fp32) + int(not include_self)
    # The scratch argument is unused for a direct FP32 reduction with self.
    scratch = (
        torch.empty(scratch_planes * inp.numel(), dtype=torch.int32, device=inp.device)
        if scratch_planes
        else index
    )
    rows, out_cols = inp.shape
    index_rows, index_cols = index.shape
    src_cols = src.shape[1]
    grid = (*_split_grid(rows), 1)
    key = (
        "row_atomic",
        inp.device,
        inp.dtype,
        rows,
        index_rows,
        out_cols,
        index_cols,
        src_cols,
        reduce,
        include_self,
        direct_fp32,
    )
    args = (
        inp,
        index,
        src,
        scratch,
        result,
        rows,
        index_rows,
        out_cols,
        index_cols,
        src_cols,
        reduce,
        include_self,
        direct_fp32,
        256,
    )
    launchers = _LIST_LAUNCHERS.get(key) if _CACHE_LIST_LAUNCHERS else None
    with torch_device_fn.device(inp.device):
        if launchers is not None:
            launchers[0](*args)
        else:
            kernel, _ = hygon_scatter_reduce_row_atomic_kernel[grid](*args, num_warps=4)
            if _CACHE_LIST_LAUNCHERS:
                if len(_LIST_LAUNCHERS) >= _MAX_LIST_LAUNCHERS:
                    _LIST_LAUNCHERS.pop(next(iter(_LIST_LAUNCHERS)), None)
                _LIST_LAUNCHERS[key] = (kernel[grid],)
    return result


def _scatter_reduce_small_reduction(inp, index, src, reduce, include_self, result=None):
    if result is None:
        result = torch.empty_like(inp)
    rows = inp.shape[0] if inp.ndim == 2 else 1
    index_rows = index.shape[0] if index.ndim == 2 else 1
    out_cols, index_cols, src_cols = inp.shape[-1], index.shape[-1], src.shape[-1]
    out_block = 1 << (out_cols - 1).bit_length()
    index_block = 1 << (index_cols - 1).bit_length()
    grid = (*_split_grid(rows), 1)
    key = (
        "small",
        inp.device,
        inp.dtype,
        rows,
        index_rows,
        out_cols,
        index_cols,
        src_cols,
        reduce,
        include_self,
    )
    args = (
        inp,
        index,
        src,
        result,
        rows,
        index_rows,
        out_cols,
        index_cols,
        src_cols,
        reduce,
        include_self,
        out_block,
        index_block,
    )
    launchers = _LIST_LAUNCHERS.get(key) if _CACHE_LIST_LAUNCHERS else None
    with torch_device_fn.device(inp.device):
        if launchers is not None:
            launchers[0](*args)
        else:
            kernel, _ = hygon_scatter_reduce_small_kernel[grid](*args, num_warps=4)
            if _CACHE_LIST_LAUNCHERS:
                if len(_LIST_LAUNCHERS) >= _MAX_LIST_LAUNCHERS:
                    _LIST_LAUNCHERS.pop(next(iter(_LIST_LAUNCHERS)), None)
                _LIST_LAUNCHERS[key] = (kernel[grid],)
    return result


def _scatter_reduce_linked_product(inp, index, src, include_self, result=None):
    # Contiguous 1-D tensors share the same flattened layout as a single row.
    if max(inp.shape[-1], index.shape[-1]) <= 64:
        return _scatter_reduce_small_reduction(
            inp, index, src, "prod", include_self, result
        )
    if result is None:
        result = torch.empty_like(inp)
    scratch = torch.empty(
        inp.numel() + index.numel(), dtype=torch.int32, device=inp.device
    )
    build_tile = _LINK_BUILD_BLOCK * _LINK_BUILD_LOOP
    build_programs = (index.numel() + build_tile - 1) // build_tile
    finalize_programs = (inp.numel() + _LINK_FINAL_BLOCK - 1) // _LINK_FINAL_BLOCK

    init_grid = (*_split_grid((inp.numel() + 1023) // 1024), 1)
    build_grid = (*_split_grid(build_programs), 1)
    final_grid = (*_split_grid(finalize_programs), 1)
    key = (
        inp.device,
        inp.dtype,
        inp.numel(),
        index.numel(),
        index.shape[-1],
        src.shape[-1],
        inp.shape[-1],
        include_self,
    )
    launchers = _LIST_LAUNCHERS.get(key) if _CACHE_LIST_LAUNCHERS else None
    with torch_device_fn.device(inp.device):
        if launchers is not None:
            # Compiled runners resolve the current stream on every call. Cache
            # neither tensor addresses nor streams, only code and launch grids.
            launchers[0](scratch, inp.numel(), 1024)
            launchers[1](
                index,
                src,
                scratch,
                scratch,
                index.numel(),
                index.shape[-1],
                src.shape[-1],
                inp.shape[-1],
                include_self,
                _LINK_BUILD_BLOCK,
                _LINK_BUILD_LOOP,
                inp.numel(),
            )
            launchers[2](
                inp,
                src,
                scratch,
                scratch,
                result,
                inp.numel(),
                index.shape[-1],
                src.shape[-1],
                include_self,
                _LINK_FINAL_BLOCK,
                inp.numel(),
            )
        else:
            init_kernel, _ = hygon_scatter_reduce_init_heads_kernel[init_grid](
                scratch, inp.numel(), BLOCK=1024
            )
            build_kernel, _ = hygon_scatter_reduce_prod_build_lists_kernel[build_grid](
                index,
                src,
                scratch,
                scratch,
                index.numel(),
                index.shape[-1],
                src.shape[-1],
                inp.shape[-1],
                include_self,
                BLOCK=_LINK_BUILD_BLOCK,
                LOOP=_LINK_BUILD_LOOP,
                NEXT_OFFSET=inp.numel(),
            )
            final_kernel, _ = hygon_scatter_reduce_prod_finalize_lists_kernel[
                final_grid
            ](
                inp,
                src,
                scratch,
                scratch,
                result,
                inp.numel(),
                index.shape[-1],
                src.shape[-1],
                include_self,
                BLOCK=_LINK_FINAL_BLOCK,
                num_warps=4,
                NEXT_OFFSET=inp.numel(),
            )
            if _CACHE_LIST_LAUNCHERS:
                if len(_LIST_LAUNCHERS) >= _MAX_LIST_LAUNCHERS:
                    _LIST_LAUNCHERS.pop(next(iter(_LIST_LAUNCHERS)), None)
                _LIST_LAUNCHERS[key] = (
                    init_kernel[init_grid],
                    build_kernel[build_grid],
                    final_kernel[final_grid],
                )
    return result


def _scatter_reduce_strided_product(inp, dim, index, src, include_self):
    """Map the active index domain to rows without floating-point CAS."""
    dim %= inp.ndim
    active_inp = inp
    active_src = src
    for axis, size in enumerate(index.shape):
        active_src = active_src.narrow(axis, 0, size)
        if axis != dim:
            active_inp = active_inp.narrow(axis, 0, size)

    row_shape = active_inp.movedim(dim, -1).shape
    inp_rows = active_inp.movedim(dim, -1).contiguous().reshape(-1, inp.shape[dim])
    index_rows = index.movedim(dim, -1).contiguous().reshape(-1, index.shape[dim])
    src_rows = active_src.movedim(dim, -1).contiguous().reshape(index_rows.shape)
    reduced = (
        _scatter_reduce_linked_product(inp_rows, index_rows, src_rows, include_self)
        .reshape(row_shape)
        .movedim(-1, dim)
    )

    if active_inp.shape == inp.shape:
        return reduced.contiguous()
    result = inp.clone(memory_format=torch.contiguous_format)
    active_result = result
    for axis, size in enumerate(index.shape):
        if axis != dim:
            active_result = active_result.narrow(axis, 0, size)
    active_result.copy_(reduced)
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
    if reduce == "prod" and _can_use_linked_product(inp, dim, index, src):
        return _scatter_reduce_linked_product(inp, index, src, include_self)
    if _can_use_small_reduction(inp, dim, index, src, reduce):
        return _scatter_reduce_small_reduction(inp, index, src, reduce, include_self)
    if _can_use_row_atomic(inp, dim, index, src, reduce):
        return _scatter_reduce_row_atomic(inp, index, src, reduce, include_self)
    if _can_use_packed16(
        inp,
        dim,
        index,
        src,
        reduce,
        include_self,
    ):
        return _scatter_reduce_packed16(inp, index, src, reduce, include_self)
    rowwise_strategy = _select_rowwise_strategy(
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
    if (
        reduce == "prod"
        and inp.ndim > 0
        and index.ndim == inp.ndim
        and src.ndim == inp.ndim
        and -inp.ndim <= dim < inp.ndim
        and inp.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and src.dtype == inp.dtype
        and index.dtype == torch.int64
        and index.numel() < 1 << 31
    ):
        return _scatter_reduce_strided_product(inp, dim, index, src, include_self)
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
    if reduce == "prod" and _can_use_linked_product(inp, dim, index, src, inp):
        return _scatter_reduce_linked_product(inp, index, src, include_self, result=inp)
    if _can_use_small_reduction(inp, dim, index, src, reduce, inp):
        return _scatter_reduce_small_reduction(
            inp, index, src, reduce, include_self, result=inp
        )
    if _can_use_row_atomic(inp, dim, index, src, reduce, inp):
        return _scatter_reduce_row_atomic(
            inp, index, src, reduce, include_self, result=inp
        )
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
        rowwise_strategy = _select_rowwise_strategy(
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
        if reduce == "prod" and _can_use_linked_product(inp, dim, index, src, out):
            return _scatter_reduce_linked_product(
                inp, index, src, include_self, result=out
            )
        if _can_use_small_reduction(inp, dim, index, src, reduce, out):
            return _scatter_reduce_small_reduction(
                inp, index, src, reduce, include_self, result=out
            )
        if _can_use_row_atomic(inp, dim, index, src, reduce, out):
            return _scatter_reduce_row_atomic(
                inp, index, src, reduce, include_self, result=out
            )
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
            rowwise_strategy = _select_rowwise_strategy(
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
    result = _scatter_reduce(inp, dim, index, src, reduce, include_self)
    if out is not None:
        out.copy_(result)
        return out
    return result

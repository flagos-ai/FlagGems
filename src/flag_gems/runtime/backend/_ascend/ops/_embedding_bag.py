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

import triton
import triton.language as tl
from packaging import version

from flag_gems.ops._embedding_bag import _embedding_bag_impl
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)
_USE_DEVICE_ASSERT = version.parse(triton.__version__) >= version.parse("3.5")


@libentry()
@triton.jit(debug=True)
def _embedding_bag_check_flags(Error, N: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    bad = tl.full((), 0, tl.int32)
    for base in range(0, N, BLOCK):
        code = tl.load(Error + base + lane, base + lane < N, other=0)
        bad |= tl.max(code, 0)
    if bad != 0:
        tl.device_assert(False, "embedding_bag: invalid index or offset")


@libentry()
@triton.jit
def _embedding_bag_one_index_kernel(
    Weight,
    Indices,
    Offsets,
    PerSample,
    Output,
    Offset2Bag,
    BagSize,
    MaxIndices,
    Error,
    N: tl.constexpr,
    O: tl.constexpr,
    B: tl.constexpr,
    D: tl.constexpr,
    V: tl.constexpr,
    STRIDE_W0: tl.constexpr,
    STRIDE_W1: tl.constexpr,
    STRIDE_I: tl.constexpr,
    STRIDE_O: tl.constexpr,
    STRIDE_P: tl.constexpr,
    PADDING: tl.constexpr,
    MODE: tl.constexpr,
    HAS_PER_SAMPLE: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ASCEND_ABI: tl.constexpr = False,
    BAG_SIZE_B: tl.constexpr = False,
    SEARCH_STEPS: tl.constexpr = 0,
    INDEX_BLOCK: tl.constexpr = 128,
    ASYNC_ASSERT: tl.constexpr = False,
    SERIAL_MAX: tl.constexpr = False,
    KEY_MAX: tl.constexpr = False,
    RECIPROCAL_MEAN: tl.constexpr = False,
    SKIP_PAD_SCALE: tl.constexpr = False,
):
    pid = tl.program_id(0)
    dim_tiles = tl.maximum(tl.cdiv(D, BLOCK_D), 1)
    idx = tl.load(Indices).to(tl.int64)
    acc_dtype: tl.constexpr = (
        tl.float64 if Weight.dtype.element_ty == tl.float64 else tl.float32
    )
    valid_idx = (idx >= 0) & (idx < V)
    cols = tl.arange(0, BLOCK_D)
    for work in range(pid, B * dim_tiles, tl.num_programs(0)):
        bag = work // dim_tiles
        tile = work % dim_tiles
        dim = tile * BLOCK_D + cols
        begin = tl.load(Offsets + bag * STRIDE_O).to(tl.int64)
        end = tl.full((), N, tl.int64)
        if bag + 1 < O:
            end = tl.load(Offsets + (bag + 1) * STRIDE_O).to(tl.int64)
        valid_offsets = (begin >= 0) & (end >= begin) & (end <= N)
        valid_offsets &= (bag != 0) | (begin == 0)
        if O > B:
            valid_offsets &= (bag != B - 1) | (end == N)
        selected = (
            valid_offsets & valid_idx & (idx != PADDING) & (begin == 0) & (end == 1)
        )
        if D > 0:
            value = tl.load(
                Weight + idx * STRIDE_W0 + dim * STRIDE_W1,
                selected & (dim < D),
                other=0.0,
            ).to(acc_dtype)
            if HAS_PER_SAMPLE:
                scale = tl.where(selected, tl.load(PerSample).to(acc_dtype), 0.0)
                value *= scale
            tl.store(Output + bag * D + dim, value, dim < D)
            if MODE == 2:
                tl.store(
                    MaxIndices + bag * D + dim, tl.where(selected, idx, 0), dim < D
                )
        if tile == 0:
            count = selected.to(tl.int32)
            tl.store(BagSize + bag, 0 if MODE == 0 else count)
            code = tl.where(valid_idx, 0, 1) | tl.where(valid_offsets, 0, 2)
            tl.store(Error + bag, code)
    if pid == 0:
        low = tl.full((), 0, tl.int32)
        high = tl.full((), B, tl.int32)
        for _ in range(SEARCH_STEPS):
            mid = (low + high) // 2
            off = tl.full((), N, Offsets.dtype.element_ty)
            if mid < B:
                off = tl.load(Offsets + mid * STRIDE_O)
            right = (mid < B) & (off <= 0)
            low = tl.where(right, mid + 1, low)
            high = tl.where(right, high, mid)
        tl.store(Offset2Bag, tl.maximum(low - 1, 0))


@triton.jit
def _embedding_bag_metadata(
    Indices,
    Offsets,
    Offset2Bag,
    bag,
    begin,
    end,
    valid_offsets,
    code,
    N: tl.constexpr,
    B: tl.constexpr,
    V: tl.constexpr,
    STRIDE_I: tl.constexpr,
    STRIDE_O: tl.constexpr,
    INDEX_BLOCK: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
):
    lane = tl.arange(0, INDEX_BLOCK)
    if N > 0:
        # Ownership is fixed by the input partition, including invalid offsets.
        # The checked hint avoids indirect offset gathers for regular bags.
        for base in range(bag * INDEX_BLOCK, N, tl.maximum(B, 1) * INDEX_BLOCK):
            pos = base + lane
            in_bounds = pos < N
            idx = tl.load(Indices + pos * STRIDE_I, in_bounds, other=0)
            invalid = in_bounds & ((idx < 0) | (idx >= V))
            code |= tl.max(invalid.to(tl.int32), 0)
            own_bag = valid_offsets & (pos >= begin) & (pos < end)
            if tl.sum((in_bounds & ~own_bag).to(tl.int32), 0) == 0:
                tl.store(Offset2Bag + pos, bag, in_bounds)
            else:
                low = tl.full((INDEX_BLOCK,), 0, tl.int64)
                high = tl.full((INDEX_BLOCK,), B, tl.int64)
                for _ in range(SEARCH_STEPS):
                    mid = (low + high) // 2
                    off = tl.load(
                        Offsets + mid * STRIDE_O,
                        in_bounds & (mid < B),
                        other=N,
                    )
                    right = (mid < B) & (off <= pos)
                    low = tl.where(right, mid + 1, low)
                    high = tl.where(right, high, mid)
                tl.store(Offset2Bag + pos, tl.maximum(low - 1, 0), in_bounds)
    return code


@libentry()
@triton.jit
def _embedding_bag_sum_kernel(
    Weight,
    Indices,
    Offsets,
    PerSample,
    Output,
    Offset2Bag,
    BagSize,
    MaxIndices,
    Error,
    N: tl.constexpr,
    O: tl.constexpr,
    B: tl.constexpr,
    D: tl.constexpr,
    V: tl.constexpr,
    STRIDE_W0: tl.constexpr,
    STRIDE_W1: tl.constexpr,
    STRIDE_I: tl.constexpr,
    STRIDE_O: tl.constexpr,
    STRIDE_P: tl.constexpr,
    PADDING: tl.constexpr,
    MODE: tl.constexpr,
    HAS_PER_SAMPLE: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ASCEND_ABI: tl.constexpr = False,
    BAG_SIZE_B: tl.constexpr = False,
    SEARCH_STEPS: tl.constexpr = 0,
    INDEX_BLOCK: tl.constexpr = 128,
    ASYNC_ASSERT: tl.constexpr = False,
    SERIAL_MAX: tl.constexpr = False,
    KEY_MAX: tl.constexpr = False,
    RECIPROCAL_MEAN: tl.constexpr = False,
    SKIP_PAD_SCALE: tl.constexpr = False,
):
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    dim_tiles = tl.maximum(tl.cdiv(D, BLOCK_D), 1)
    acc_dtype: tl.constexpr = (
        tl.float64 if Weight.dtype.element_ty == tl.float64 else tl.float32
    )
    rows = tl.arange(0, BLOCK_L)
    cols = tl.arange(0, BLOCK_D)
    total_work = B * dim_tiles
    for work in range(pid, total_work, programs):
        bag = work // dim_tiles
        dim_tile = work % dim_tiles
        dim = dim_tile * BLOCK_D + cols
        begin = tl.load(Offsets + bag * STRIDE_O).to(tl.int64)
        end = tl.load(Offsets + (bag + 1) * STRIDE_O, bag + 1 < O, other=0).to(tl.int64)
        end = tl.where(bag + 1 < O, end, N)
        valid_offsets = (begin >= 0) & (end >= begin) & (end <= N)
        valid_offsets &= (bag != 0) | (begin == 0)
        if O > B:
            valid_offsets &= (bag != B - 1) | (end == N)
        if dim_tile == 0:
            code = tl.where(valid_offsets, 0, 2)
            code = _embedding_bag_metadata(
                Indices,
                Offsets,
                Offset2Bag,
                bag,
                begin,
                end,
                valid_offsets,
                code,
                N,
                B,
                V,
                STRIDE_I,
                STRIDE_O,
                INDEX_BLOCK,
                SEARCH_STEPS,
            )
            if ASYNC_ASSERT:
                tl.device_assert(code == 0, "embedding_bag: invalid index or offset")
            else:
                tl.store(Error + bag, code)
        end = tl.where(valid_offsets, end, begin)
        begin = tl.minimum(tl.maximum(begin, 0), N)
        end = tl.minimum(tl.maximum(end, begin), N)
        count = tl.full((), 0, tl.int64)
        acc = tl.zeros((BLOCK_L, BLOCK_D), acc_dtype)
        if N > 0:
            for base in range(begin, end, BLOCK_L):
                pos = base + rows
                active = pos < end
                idx = tl.load(Indices + pos * STRIDE_I, active, other=0).to(tl.int64)
                valid = active & (idx >= 0) & (idx < V) & (idx != PADDING)
                if D > 0:
                    values = tl.load(
                        Weight + idx[:, None] * STRIDE_W0 + dim[None, :] * STRIDE_W1,
                        valid[:, None] & (dim[None, :] < D),
                        other=0.0,
                    ).to(acc_dtype)
                else:
                    values = tl.zeros((BLOCK_L, BLOCK_D), acc_dtype)
                count += tl.sum(valid.to(tl.int32), 0)
                if HAS_PER_SAMPLE:
                    scale = tl.load(
                        PerSample + pos * STRIDE_P,
                        valid if SKIP_PAD_SCALE else active,
                        other=0.0,
                    )
                    values *= scale[:, None].to(acc_dtype)
                acc += values
        result = tl.sum(acc, 0)
        if MODE == 1:
            if RECIPROCAL_MEAN:
                reciprocal = 1.0 / tl.maximum(count, 1).to(acc_dtype)
                result = result * reciprocal
            else:
                result = result / tl.maximum(count, 1).to(acc_dtype)
        if D > 0:
            tl.store(Output + bag * D + dim, result, dim < D)
        if dim_tile == 0:
            if ASCEND_ABI and MODE == 0:
                tl.store(BagSize + bag, 0)
            else:
                tl.store(BagSize + bag, count)
    if O > B and (not BAG_SIZE_B):
        if pid == 0:
            tl.store(BagSize + B, 0)
    if B == 0:
        if pid == 0:
            only_offset = tl.load(Offsets)
            if ASYNC_ASSERT:
                tl.device_assert(
                    only_offset == 0, "embedding_bag: invalid terminal offset"
                )
            else:
                tl.store(Error, tl.where(only_offset == 0, 0, 2))


@libentry()
@triton.jit
def _embedding_bag_max_kernel(
    Weight,
    Indices,
    Offsets,
    PerSample,
    Output,
    Offset2Bag,
    BagSize,
    MaxIndices,
    Error,
    N: tl.constexpr,
    O: tl.constexpr,
    B: tl.constexpr,
    D: tl.constexpr,
    V: tl.constexpr,
    STRIDE_W0: tl.constexpr,
    STRIDE_W1: tl.constexpr,
    STRIDE_I: tl.constexpr,
    STRIDE_O: tl.constexpr,
    STRIDE_P: tl.constexpr,
    PADDING: tl.constexpr,
    MODE: tl.constexpr,
    HAS_PER_SAMPLE: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ASCEND_ABI: tl.constexpr = False,
    BAG_SIZE_B: tl.constexpr = False,
    SEARCH_STEPS: tl.constexpr = 0,
    INDEX_BLOCK: tl.constexpr = 128,
    ASYNC_ASSERT: tl.constexpr = False,
    SERIAL_MAX: tl.constexpr = False,
    KEY_MAX: tl.constexpr = False,
    RECIPROCAL_MEAN: tl.constexpr = False,
    SKIP_PAD_SCALE: tl.constexpr = False,
):
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    dim_tiles = tl.maximum(tl.cdiv(D, BLOCK_D), 1)
    acc_dtype: tl.constexpr = (
        tl.float64 if Weight.dtype.element_ty == tl.float64 else tl.float32
    )
    position_dtype: tl.constexpr = tl.int32 if N <= 2147483647 - BLOCK_L else tl.int64
    winner_dtype: tl.constexpr = tl.int32 if V <= 2147483647 else tl.int64
    rows = tl.arange(0, BLOCK_L)
    cols = tl.arange(0, BLOCK_D)
    total_work = B * dim_tiles
    for work in range(pid, total_work, programs):
        bag = work // dim_tiles
        dim_tile = work % dim_tiles
        dim = dim_tile * BLOCK_D + cols
        begin = tl.load(Offsets + bag * STRIDE_O).to(tl.int64)
        end = tl.load(Offsets + (bag + 1) * STRIDE_O, bag + 1 < O, other=0).to(tl.int64)
        end = tl.where(bag + 1 < O, end, N)
        valid_offsets = (begin >= 0) & (end >= begin) & (end <= N)
        valid_offsets &= (bag != 0) | (begin == 0)
        if O > B:
            valid_offsets &= (bag != B - 1) | (end == N)
        if dim_tile == 0:
            code = tl.where(valid_offsets, 0, 2)
            code = _embedding_bag_metadata(
                Indices,
                Offsets,
                Offset2Bag,
                bag,
                begin,
                end,
                valid_offsets,
                code,
                N,
                B,
                V,
                STRIDE_I,
                STRIDE_O,
                INDEX_BLOCK,
                SEARCH_STEPS,
            )
            if ASYNC_ASSERT:
                tl.device_assert(code == 0, "embedding_bag: invalid index or offset")
            else:
                tl.store(Error + bag, code)
        end = tl.where(valid_offsets, end, begin)
        begin = tl.minimum(tl.maximum(begin, 0), N)
        end = tl.minimum(tl.maximum(end, begin), N)
        # Narrow reduction state only after validating and clamping offsets.
        # Indirect pointer arithmetic still uses the original 64-bit indices.
        begin = begin.to(position_dtype)
        end = end.to(position_dtype)
        count = tl.full((), 0, tl.int64)
        best = tl.full((BLOCK_D,), float("-inf"), acc_dtype)
        best_pos = tl.full((BLOCK_D,), N, position_dtype)
        tail_nan = tl.zeros((BLOCK_D,), tl.int32)
        bits_dtype: tl.constexpr = tl.int64 if acc_dtype == tl.float64 else tl.int32
        result_bits = tl.zeros((BLOCK_D,), bits_dtype)
        winner = tl.zeros((BLOCK_D,), winner_dtype)
        if N > 0:
            for base in range(begin, end, BLOCK_L):
                pos = base + rows
                idx = tl.load(
                    Indices + pos.to(tl.int64) * STRIDE_I, pos < end, other=0
                ).to(tl.int64)
                valid = (pos < end) & (idx >= 0) & (idx < V) & (idx != PADDING)
                if D > 0:
                    values = tl.load(
                        Weight + idx[:, None] * STRIDE_W0 + dim[None, :] * STRIDE_W1,
                        valid[:, None] & (dim[None, :] < D),
                        other=0.0,
                    ).to(acc_dtype)
                else:
                    values = tl.zeros((BLOCK_L, BLOCK_D), acc_dtype)
                tile_count = tl.sum(valid.to(tl.int32), 0)
                count += tile_count
                # Ascend resets the running maximum on every NaN. A later
                # finite value resets it again; finite ties retain the first row.
                last_valid = tl.max(tl.where(valid, pos, -1), 0)
                last_nan = tl.max(
                    tl.where(valid[:, None] & (values != values), pos[:, None], -1), 0
                )
                numeric = (
                    valid[:, None]
                    & (values == values)
                    & (pos[:, None] > last_nan[None, :])
                )
                tile_best = tl.max(tl.where(numeric, values, float("-inf")), 0)
                tile_pos = tl.min(
                    tl.where(numeric & (values == tile_best[None, :]), pos[:, None], N),
                    0,
                )
                tile_tail_nan = (last_nan >= 0) & (last_nan == last_valid)
                tile_pos = tl.where(tile_tail_nan, last_nan, tile_pos)
                chosen = valid[:, None] & (pos[:, None] == tile_pos[None, :])
                tile_winner = tl.sum(
                    tl.where(chosen, idx[:, None].to(winner_dtype), 0), 0
                )
                tile_bits = tl.sum(
                    tl.where(chosen, values.to(bits_dtype, bitcast=True), 0), 0
                )
                replace = (tile_count > 0) & (
                    (last_nan >= 0)
                    | (tail_nan != 0)
                    | (tile_best > best)
                    | ((tile_best == best) & (tile_pos < best_pos))
                )
                best = tl.where(replace, tile_best, best)
                winner = tl.where(replace, tile_winner, winner)
                result_bits = tl.where(replace, tile_bits, result_bits)
                best_pos = tl.where(replace, tile_pos, best_pos)
                tail_nan = tl.where(
                    tile_count > 0, tile_tail_nan.to(tl.int32), tail_nan
                )
        if D > 0:
            # The selected values already reside in the loaded row tile.
            # Selecting their bits preserves signed zero and avoids a final
            # per-feature indirect reload from global weight memory.
            result = result_bits.to(acc_dtype, bitcast=True)
            tl.store(Output + bag * D + dim, result, dim < D)
            tl.store(MaxIndices + bag * D + dim, winner, dim < D)
        if dim_tile == 0:
            tl.store(BagSize + bag, count)
    if B == 0 and pid == 0:
        only_offset = tl.load(Offsets)
        tl.store(Error, tl.where(only_offset == 0, 0, 2))


def _check_error(error, count):
    # Complex gather/reduction kernels cannot always lower inline assertions.
    # A separate check preserves validation without a host synchronization.
    _embedding_bag_check_flags[(1,)](error, count, 128, debug=True)


def _auxiliary_result(result, mode, padding_idx):
    # Ascend's SUM fast path omits offset2bag when padding is disabled.
    if mode == 0 and padding_idx == -1:
        output, mapping, bag_size, maximum = result
        return output, mapping[:0], bag_size, maximum
    return result


def _embedding_bag(
    weight,
    indices,
    offsets,
    scale_grad_by_freq=False,
    mode=0,
    sparse=False,
    per_sample_weights=None,
    include_last_offset=False,
    padding_idx=-1,
):
    logger.debug("GEMS _EMBEDDING_BAG")
    # Strided gathers need smaller tiles to fit the vector-core UB.
    config = None
    if weight.ndim == 2:
        bags = max(offsets.numel() - int(include_last_offset), 1)
        index_block = min(
            128, triton.next_power_of_2(max(triton.cdiv(indices.numel(), bags), 1))
        )
        config = (
            16 if indices.numel() <= bags * 16 else 32,
            min(64, triton.next_power_of_2(max(weight.shape[1], 1))),
            index_block,
            4,
        )
    if weight.ndim == 2 and weight.stride(1) != 1:
        config = (4, min(32, triton.next_power_of_2(max(weight.shape[1], 1))), 32, 4)
    result = _embedding_bag_impl(
        weight,
        indices,
        offsets,
        scale_grad_by_freq,
        mode,
        sparse,
        per_sample_weights,
        include_last_offset,
        padding_idx,
        _kernel=(
            _embedding_bag_one_index_kernel
            if indices.numel() == 1
            else _embedding_bag_max_kernel if mode == 2 else _embedding_bag_sum_kernel
        ),
        _launch_kwargs={"SKIP_PAD_SCALE": True},
        _async_assert=False,
        _check_error=_check_error if _USE_DEVICE_ASSERT else None,
        _block_config=config,
    )
    return _auxiliary_result(result, mode, padding_idx)


def _embedding_bag_forward_only(
    weight,
    indices,
    offsets,
    scale_grad_by_freq=False,
    mode=0,
    sparse=False,
    per_sample_weights=None,
    include_last_offset=False,
    padding_idx=-1,
):
    logger.debug("GEMS _EMBEDDING_BAG_FORWARD_ONLY")
    config = None
    if weight.ndim == 2:
        bags = max(offsets.numel() - int(include_last_offset), 1)
        index_block = min(
            128, triton.next_power_of_2(max(triton.cdiv(indices.numel(), bags), 1))
        )
        config = (
            16 if indices.numel() <= bags * 16 else 32,
            min(64, triton.next_power_of_2(max(weight.shape[1], 1))),
            index_block,
            4,
        )
    if weight.ndim == 2 and weight.stride(1) != 1:
        config = (4, min(32, triton.next_power_of_2(max(weight.shape[1], 1))), 32, 4)
    result = _embedding_bag_impl(
        weight,
        indices,
        offsets,
        scale_grad_by_freq,
        mode,
        sparse,
        per_sample_weights,
        include_last_offset,
        padding_idx,
        forward_only=True,
        _kernel=(
            _embedding_bag_one_index_kernel
            if indices.numel() == 1
            else _embedding_bag_max_kernel if mode == 2 else _embedding_bag_sum_kernel
        ),
        _launch_kwargs={"SKIP_PAD_SCALE": True},
        _async_assert=False,
        _check_error=_check_error if _USE_DEVICE_ASSERT else None,
        _block_config=config,
    )
    return _auxiliary_result(result, mode, padding_idx)

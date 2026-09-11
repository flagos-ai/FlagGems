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
def _embedding_bag_check_flags(
    error,
    num_error_flags: tl.constexpr,
    block: tl.constexpr,  # element block size
):
    lane = tl.arange(0, block)
    bad = tl.full((), 0, tl.int32)
    for base in range(0, num_error_flags, block):
        code = tl.load(error + base + lane, base + lane < num_error_flags, other=0)
        bad |= tl.max(code, 0)
    if bad != 0:
        tl.device_assert(False, "embedding_bag: invalid index or offset")


@libentry()
@triton.jit
def _embedding_bag_one_index_kernel(
    weight,
    indices,
    offsets,
    per_sample_weights,
    output,
    offset_to_bag,
    bag_size,
    max_indices,  # maximum-value embedding indices
    error,
    num_indices: tl.constexpr,
    num_offsets: tl.constexpr,
    num_bags: tl.constexpr,
    embedding_dim: tl.constexpr,
    num_weights: tl.constexpr,
    stride_w0: tl.constexpr,  # weight row stride
    stride_w1: tl.constexpr,  # weight feature stride
    stride_i: tl.constexpr,  # indices stride
    stride_o: tl.constexpr,  # offsets stride
    stride_p: tl.constexpr,  # per-sample-weight stride
    padding: tl.constexpr,
    mode: tl.constexpr,
    has_per_sample: tl.constexpr,
    block_l: tl.constexpr,  # bag-length block size
    block_d: tl.constexpr,  # embedding-dimension block size
    ascend_abi: tl.constexpr = False,  # Ascend auxiliary-output application binary interface
    bag_size_b: tl.constexpr = False,  # bag-size output uses num_bags entries
    search_steps: tl.constexpr = 0,
    index_block: tl.constexpr = 128,
    async_assert: tl.constexpr = False,  # asynchronous device assertion
    serial_max: tl.constexpr = False,  # serial maximum reduction
    key_max: tl.constexpr = False,  # maximum reduction using ordered keys
    reciprocal_mean: tl.constexpr = False,
    skip_pad_scale: tl.constexpr = False,  # skip padding-scale adjustment
):
    pid = tl.program_id(0)
    dim_tiles = tl.maximum(tl.cdiv(embedding_dim, block_d), 1)
    idx = tl.load(indices).to(tl.int64)
    acc_dtype: tl.constexpr = (
        tl.float64 if weight.dtype.element_ty == tl.float64 else tl.float32
    )
    valid_idx = (idx >= 0) & (idx < num_weights)
    cols = tl.arange(0, block_d)
    for work in range(pid, num_bags * dim_tiles, tl.num_programs(0)):
        bag = work // dim_tiles
        tile = work % dim_tiles
        dim = tile * block_d + cols
        begin = tl.load(offsets + bag * stride_o).to(tl.int64)
        end = tl.full((), num_indices, tl.int64)
        if bag + 1 < num_offsets:
            end = tl.load(offsets + (bag + 1) * stride_o).to(tl.int64)
        valid_offsets = (begin >= 0) & (end >= begin) & (end <= num_indices)
        valid_offsets &= (bag != 0) | (begin == 0)
        if num_offsets > num_bags:
            valid_offsets &= (bag != num_bags - 1) | (end == num_indices)
        selected = (
            valid_offsets & valid_idx & (idx != padding) & (begin == 0) & (end == 1)
        )
        if embedding_dim > 0:
            value = tl.load(
                weight + idx * stride_w0 + dim * stride_w1,
                selected & (dim < embedding_dim),
                other=0.0,
            ).to(acc_dtype)
            if has_per_sample:
                scale = tl.where(
                    selected, tl.load(per_sample_weights).to(acc_dtype), 0.0
                )
                value *= scale
            tl.store(output + bag * embedding_dim + dim, value, dim < embedding_dim)
            if mode == 2:
                tl.store(
                    max_indices + bag * embedding_dim + dim,
                    tl.where(selected, idx, 0),
                    dim < embedding_dim,
                )
        if tile == 0:
            count = selected.to(tl.int32)
            tl.store(bag_size + bag, 0 if mode == 0 else count)
            code = tl.where(valid_idx, 0, 1) | tl.where(valid_offsets, 0, 2)
            tl.store(error + bag, code)
    if pid == 0:
        low = tl.full((), 0, tl.int32)
        high = tl.full((), num_bags, tl.int32)
        for _ in range(search_steps):
            mid = (low + high) // 2
            off = tl.full((), num_indices, offsets.dtype.element_ty)
            if mid < num_bags:
                off = tl.load(offsets + mid * stride_o)
            right = (mid < num_bags) & (off <= 0)
            low = tl.where(right, mid + 1, low)
            high = tl.where(right, high, mid)
        tl.store(offset_to_bag, tl.maximum(low - 1, 0))


@triton.jit
def _embedding_bag_metadata(
    indices,
    offsets,
    offset_to_bag,
    bag,
    begin,
    end,
    valid_offsets,
    code,
    num_indices: tl.constexpr,
    num_bags: tl.constexpr,
    num_weights: tl.constexpr,
    stride_i: tl.constexpr,  # indices stride
    stride_o: tl.constexpr,  # offsets stride
    index_block: tl.constexpr,
    search_steps: tl.constexpr,
):
    lane = tl.arange(0, index_block)
    if num_indices > 0:
        # Ownership is fixed by the input partition, including invalid offsets.
        # The checked hint avoids indirect offset gathers for regular bags.
        for base in range(
            bag * index_block, num_indices, tl.maximum(num_bags, 1) * index_block
        ):
            pos = base + lane
            in_bounds = pos < num_indices
            idx = tl.load(indices + pos * stride_i, in_bounds, other=0)
            invalid = in_bounds & ((idx < 0) | (idx >= num_weights))
            code |= tl.max(invalid.to(tl.int32), 0)
            own_bag = valid_offsets & (pos >= begin) & (pos < end)
            if tl.sum((in_bounds & ~own_bag).to(tl.int32), 0) == 0:
                tl.store(offset_to_bag + pos, bag, in_bounds)
            else:
                low = tl.full((index_block,), 0, tl.int64)
                high = tl.full((index_block,), num_bags, tl.int64)
                for _ in range(search_steps):
                    mid = (low + high) // 2
                    off = tl.load(
                        offsets + mid * stride_o,
                        in_bounds & (mid < num_bags),
                        other=num_indices,
                    )
                    right = (mid < num_bags) & (off <= pos)
                    low = tl.where(right, mid + 1, low)
                    high = tl.where(right, high, mid)
                tl.store(offset_to_bag + pos, tl.maximum(low - 1, 0), in_bounds)
    return code


@libentry()
@triton.jit
def _embedding_bag_sum_kernel(
    weight,
    indices,
    offsets,
    per_sample_weights,
    output,
    offset_to_bag,
    bag_size,
    max_indices,  # maximum-value embedding indices
    error,
    num_indices: tl.constexpr,
    num_offsets: tl.constexpr,
    num_bags: tl.constexpr,
    embedding_dim: tl.constexpr,
    num_weights: tl.constexpr,
    stride_w0: tl.constexpr,  # weight row stride
    stride_w1: tl.constexpr,  # weight feature stride
    stride_i: tl.constexpr,  # indices stride
    stride_o: tl.constexpr,  # offsets stride
    stride_p: tl.constexpr,  # per-sample-weight stride
    padding: tl.constexpr,
    mode: tl.constexpr,
    has_per_sample: tl.constexpr,
    block_l: tl.constexpr,  # bag-length block size
    block_d: tl.constexpr,  # embedding-dimension block size
    ascend_abi: tl.constexpr = False,  # Ascend auxiliary-output application binary interface
    bag_size_b: tl.constexpr = False,  # bag-size output uses num_bags entries
    search_steps: tl.constexpr = 0,
    index_block: tl.constexpr = 128,
    async_assert: tl.constexpr = False,  # asynchronous device assertion
    serial_max: tl.constexpr = False,  # serial maximum reduction
    key_max: tl.constexpr = False,  # maximum reduction using ordered keys
    reciprocal_mean: tl.constexpr = False,
    skip_pad_scale: tl.constexpr = False,  # skip padding-scale adjustment
):
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    dim_tiles = tl.maximum(tl.cdiv(embedding_dim, block_d), 1)
    acc_dtype: tl.constexpr = (
        tl.float64 if weight.dtype.element_ty == tl.float64 else tl.float32
    )
    rows = tl.arange(0, block_l)
    cols = tl.arange(0, block_d)
    total_work = num_bags * dim_tiles
    for work in range(pid, total_work, programs):
        bag = work // dim_tiles
        dim_tile = work % dim_tiles
        dim = dim_tile * block_d + cols
        begin = tl.load(offsets + bag * stride_o).to(tl.int64)
        end = tl.load(
            offsets + (bag + 1) * stride_o, bag + 1 < num_offsets, other=0
        ).to(tl.int64)
        end = tl.where(bag + 1 < num_offsets, end, num_indices)
        valid_offsets = (begin >= 0) & (end >= begin) & (end <= num_indices)
        valid_offsets &= (bag != 0) | (begin == 0)
        if num_offsets > num_bags:
            valid_offsets &= (bag != num_bags - 1) | (end == num_indices)
        if dim_tile == 0:
            code = tl.where(valid_offsets, 0, 2)
            code = _embedding_bag_metadata(
                indices,
                offsets,
                offset_to_bag,
                bag,
                begin,
                end,
                valid_offsets,
                code,
                num_indices,
                num_bags,
                num_weights,
                stride_i,
                stride_o,
                index_block,
                search_steps,
            )
            if async_assert:
                tl.device_assert(code == 0, "embedding_bag: invalid index or offset")
            else:
                tl.store(error + bag, code)
        end = tl.where(valid_offsets, end, begin)
        begin = tl.minimum(tl.maximum(begin, 0), num_indices)
        end = tl.minimum(tl.maximum(end, begin), num_indices)
        count = tl.full((), 0, tl.int64)
        acc = tl.zeros((block_l, block_d), acc_dtype)
        if num_indices > 0:
            for base in range(begin, end, block_l):
                pos = base + rows
                active = pos < end
                idx = tl.load(indices + pos * stride_i, active, other=0).to(tl.int64)
                valid = active & (idx >= 0) & (idx < num_weights) & (idx != padding)
                if embedding_dim > 0:
                    values = tl.load(
                        weight + idx[:, None] * stride_w0 + dim[None, :] * stride_w1,
                        valid[:, None] & (dim[None, :] < embedding_dim),
                        other=0.0,
                    ).to(acc_dtype)
                else:
                    values = tl.zeros((block_l, block_d), acc_dtype)
                count += tl.sum(valid.to(tl.int32), 0)
                if has_per_sample:
                    scale = tl.load(
                        per_sample_weights + pos * stride_p,
                        valid if skip_pad_scale else active,
                        other=0.0,
                    )
                    values *= scale[:, None].to(acc_dtype)
                acc += values
        result = tl.sum(acc, 0)
        if mode == 1:
            if reciprocal_mean:
                reciprocal = 1.0 / tl.maximum(count, 1).to(acc_dtype)
                result = result * reciprocal
            else:
                result = result / tl.maximum(count, 1).to(acc_dtype)
        if embedding_dim > 0:
            tl.store(output + bag * embedding_dim + dim, result, dim < embedding_dim)
        if dim_tile == 0:
            if ascend_abi and mode == 0:
                tl.store(bag_size + bag, 0)
            else:
                tl.store(bag_size + bag, count)
    if num_offsets > num_bags and (not bag_size_b):
        if pid == 0:
            tl.store(bag_size + num_bags, 0)
    if num_bags == 0:
        if pid == 0:
            only_offset = tl.load(offsets)
            if async_assert:
                tl.device_assert(
                    only_offset == 0, "embedding_bag: invalid terminal offset"
                )
            else:
                tl.store(error, tl.where(only_offset == 0, 0, 2))


@libentry()
@triton.jit
def _embedding_bag_max_kernel(
    weight,
    indices,
    offsets,
    per_sample_weights,
    output,
    offset_to_bag,
    bag_size,
    max_indices,  # maximum-value embedding indices
    error,
    num_indices: tl.constexpr,
    num_offsets: tl.constexpr,
    num_bags: tl.constexpr,
    embedding_dim: tl.constexpr,
    num_weights: tl.constexpr,
    stride_w0: tl.constexpr,  # weight row stride
    stride_w1: tl.constexpr,  # weight feature stride
    stride_i: tl.constexpr,  # indices stride
    stride_o: tl.constexpr,  # offsets stride
    stride_p: tl.constexpr,  # per-sample-weight stride
    padding: tl.constexpr,
    mode: tl.constexpr,
    has_per_sample: tl.constexpr,
    block_l: tl.constexpr,  # bag-length block size
    block_d: tl.constexpr,  # embedding-dimension block size
    ascend_abi: tl.constexpr = False,  # Ascend auxiliary-output application binary interface
    bag_size_b: tl.constexpr = False,  # bag-size output uses num_bags entries
    search_steps: tl.constexpr = 0,
    index_block: tl.constexpr = 128,
    async_assert: tl.constexpr = False,  # asynchronous device assertion
    serial_max: tl.constexpr = False,  # serial maximum reduction
    key_max: tl.constexpr = False,  # maximum reduction using ordered keys
    reciprocal_mean: tl.constexpr = False,
    skip_pad_scale: tl.constexpr = False,  # skip padding-scale adjustment
):
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    dim_tiles = tl.maximum(tl.cdiv(embedding_dim, block_d), 1)
    acc_dtype: tl.constexpr = (
        tl.float64 if weight.dtype.element_ty == tl.float64 else tl.float32
    )
    position_dtype: tl.constexpr = (
        tl.int32 if num_indices <= 2147483647 - block_l else tl.int64
    )
    winner_dtype: tl.constexpr = tl.int32 if num_weights <= 2147483647 else tl.int64
    rows = tl.arange(0, block_l)
    cols = tl.arange(0, block_d)
    total_work = num_bags * dim_tiles
    for work in range(pid, total_work, programs):
        bag = work // dim_tiles
        dim_tile = work % dim_tiles
        dim = dim_tile * block_d + cols
        begin = tl.load(offsets + bag * stride_o).to(tl.int64)
        end = tl.load(
            offsets + (bag + 1) * stride_o, bag + 1 < num_offsets, other=0
        ).to(tl.int64)
        end = tl.where(bag + 1 < num_offsets, end, num_indices)
        valid_offsets = (begin >= 0) & (end >= begin) & (end <= num_indices)
        valid_offsets &= (bag != 0) | (begin == 0)
        if num_offsets > num_bags:
            valid_offsets &= (bag != num_bags - 1) | (end == num_indices)
        if dim_tile == 0:
            code = tl.where(valid_offsets, 0, 2)
            code = _embedding_bag_metadata(
                indices,
                offsets,
                offset_to_bag,
                bag,
                begin,
                end,
                valid_offsets,
                code,
                num_indices,
                num_bags,
                num_weights,
                stride_i,
                stride_o,
                index_block,
                search_steps,
            )
            if async_assert:
                tl.device_assert(code == 0, "embedding_bag: invalid index or offset")
            else:
                tl.store(error + bag, code)
        end = tl.where(valid_offsets, end, begin)
        begin = tl.minimum(tl.maximum(begin, 0), num_indices)
        end = tl.minimum(tl.maximum(end, begin), num_indices)
        # Narrow reduction state only after validating and clamping offsets.
        # Indirect pointer arithmetic still uses the original 64-bit indices.
        begin = begin.to(position_dtype)
        end = end.to(position_dtype)
        count = tl.full((), 0, tl.int64)
        best = tl.full((block_d,), float("-inf"), acc_dtype)
        best_pos = tl.full((block_d,), num_indices, position_dtype)
        tail_nan = tl.zeros((block_d,), tl.int32)
        bits_dtype: tl.constexpr = tl.int64 if acc_dtype == tl.float64 else tl.int32
        result_bits = tl.zeros((block_d,), bits_dtype)
        winner = tl.zeros((block_d,), winner_dtype)
        if num_indices > 0:
            for base in range(begin, end, block_l):
                pos = base + rows
                idx = tl.load(
                    indices + pos.to(tl.int64) * stride_i, pos < end, other=0
                ).to(tl.int64)
                valid = (
                    (pos < end) & (idx >= 0) & (idx < num_weights) & (idx != padding)
                )
                if embedding_dim > 0:
                    values = tl.load(
                        weight + idx[:, None] * stride_w0 + dim[None, :] * stride_w1,
                        valid[:, None] & (dim[None, :] < embedding_dim),
                        other=0.0,
                    ).to(acc_dtype)
                else:
                    values = tl.zeros((block_l, block_d), acc_dtype)
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
                    tl.where(
                        numeric & (values == tile_best[None, :]),
                        pos[:, None],
                        num_indices,
                    ),
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
        if embedding_dim > 0:
            # The selected values already reside in the loaded row tile.
            # Selecting their bits preserves signed zero and avoids a final
            # per-feature indirect reload from global weight memory.
            result = result_bits.to(acc_dtype, bitcast=True)
            tl.store(output + bag * embedding_dim + dim, result, dim < embedding_dim)
            tl.store(
                max_indices + bag * embedding_dim + dim, winner, dim < embedding_dim
            )
        if dim_tile == 0:
            tl.store(bag_size + bag, count)
    if num_bags == 0 and pid == 0:
        only_offset = tl.load(offsets)
        tl.store(error, tl.where(only_offset == 0, 0, 2))


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
    logger.debug("GEMS_ASCEND _EMBEDDING_BAG")
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
        _launch_kwargs={"skip_pad_scale": True},
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
    logger.debug("GEMS_ASCEND _EMBEDDING_BAG_FORWARD_ONLY")
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
        _launch_kwargs={"skip_pad_scale": True},
        _async_assert=False,
        _check_error=_check_error if _USE_DEVICE_ASSERT else None,
        _block_config=config,
    )
    return _auxiliary_result(result, mode, padding_idx)

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

from flag_gems.ops._embedding_bag import _embedding_bag_impl
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle


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
def _embedding_bag_forward_mthreads(
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
    mapping_hint: tl.constexpr = False,
):
    pid = tle.program_id(0)
    programs = tle.num_programs(0)
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
            if mapping_hint:
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
            else:
                index_lane = tl.arange(0, index_block)
                # Fixed partitions keep mapping writes disjoint even when a
                # malformed offset sequence makes different bags overlap.
                if num_indices > 0:
                    for input_base in range(
                        bag * index_block,
                        num_indices,
                        tl.maximum(num_bags, 1) * index_block,
                    ):
                        pos = input_base + index_lane
                        in_bounds = pos < num_indices
                        idx = tl.load(indices + pos * stride_i, in_bounds, other=0)
                        invalid = in_bounds & ((idx < 0) | (idx >= num_weights))
                        code |= tl.max(invalid.to(tl.int32), 0)
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
            if async_assert:
                tl.device_assert(code == 0, "embedding_bag: invalid index or offset")
            else:
                tl.store(error + bag, code)
        end = tl.where(valid_offsets, end, begin)
        begin = tl.minimum(tl.maximum(begin, 0), num_indices)
        end = tl.minimum(tl.maximum(end, begin), num_indices)
        if mode == 2 and serial_max:
            count = tl.full((), 0, tl.int64)
            best = tl.full((block_d,), 0, acc_dtype)
            winner = tl.full((block_d,), -1, tl.int64)
            # Unrolling keeps the original sequential comparison order. Restrict
            # it to the measured BF16/FP32 medium-width path; other paths retain
            # the scalar loop below.
            if (embedding_dim >= 256 and embedding_dim < 512) and (
                weight.dtype.element_ty == tl.bfloat16
                or weight.dtype.element_ty == tl.float32
            ):
                for base in range(begin, end, 4):
                    # Preserve input order and first-value NaN/tie semantics. Each
                    # indirect access retains its own bounds and padding mask.
                    for shift in tl.static_range(4):
                        pos = base + shift
                        active = (pos >= 0) & (pos < end)
                        idx = tl.load(indices + pos * stride_i, active, other=0).to(
                            tl.int64
                        )
                        valid = (
                            active & (idx >= 0) & (idx < num_weights) & (idx != padding)
                        )
                        values = tl.load(
                            weight + idx * stride_w0 + dim * stride_w1,
                            valid & (dim < embedding_dim),
                            other=0.0,
                        ).to(acc_dtype)
                        replace = valid & ((count == 0) | (values > best))
                        best = tl.where(replace, values, best)
                        winner = tl.where(replace, idx, winner)
                        count += valid.to(tl.int64)
            else:
                for pos in range(begin, end):
                    idx = tl.load(indices + pos * stride_i).to(tl.int64)
                    valid = (idx >= 0) & (idx < num_weights) & (idx != padding)
                    values = tl.load(
                        weight + idx * stride_w0 + dim * stride_w1,
                        valid & (dim < embedding_dim),
                        other=0.0,
                    ).to(acc_dtype)
                    replace = valid & ((count == 0) | (values > best))
                    best = tl.where(replace, values, best)
                    winner = tl.where(replace, idx, winner)
                    count += valid.to(tl.int64)
            result = best
            tl.store(
                max_indices + bag * embedding_dim + dim, winner, dim < embedding_dim
            )
        else:
            count = tl.full((), 0, tl.int64)
            if mode == 2:
                if key_max:
                    best = tl.full((block_d,), -9223372036854775808, tl.int64)
                else:
                    best = tl.full((block_d,), float("-inf"), acc_dtype)
                # Position values never exceed the N sentinel. Pointer arithmetic
                # is promoted back to int64 before applying arbitrary input strides.
                position_dtype: tl.constexpr = (
                    tl.int32 if num_indices < 2147483647 else tl.int64
                )
                best_pos = tl.full((block_d,), num_indices, position_dtype)
                first = tl.full((), num_indices, position_dtype)
            else:
                acc = tl.zeros((block_l, block_d), acc_dtype)
            if num_indices > 0:
                for base in range(begin, end, block_l):
                    pos = base + rows
                    active = pos < end
                    idx = tl.load(indices + pos * stride_i, active, other=0).to(
                        tl.int64
                    )
                    valid = active & (idx >= 0) & (idx < num_weights) & (idx != padding)
                    if embedding_dim > 0:
                        values = tl.load(
                            weight
                            + idx[:, None] * stride_w0
                            + dim[None, :] * stride_w1,
                            valid[:, None] & (dim[None, :] < embedding_dim),
                            other=0.0,
                        ).to(acc_dtype)
                    else:
                        values = tl.zeros((block_l, block_d), acc_dtype)
                    count += tl.sum(valid.to(tl.int32), 0)
                    if mode == 2:
                        position = pos.to(position_dtype)
                        first = tl.minimum(
                            first, tl.min(tl.where(valid, position, num_indices), 0)
                        )
                        if key_max:
                            bits = values.to(tl.int64, bitcast=True)
                            magnitude = bits & 0x7FFFFFFFFFFFFFFF
                            numeric = valid[:, None] & (magnitude <= 0x7FF0000000000000)
                            ordered = bits ^ tl.where(bits < 0, 0x7FFFFFFFFFFFFFFF, 0)
                            ordered = tl.where(magnitude == 0, 0, ordered)
                            candidate = tl.where(numeric, ordered, -9223372036854775808)
                            # Ordered IEEE-754 bits permit an integer argmax.
                            # Canonical zero keys preserve the first signed-zero tie;
                            # the original winning value is loaded below.
                            tile_best, relative = tl.max(
                                candidate, 0, return_indices=True
                            )
                            tile_pos = tl.where(
                                tile_best > -9223372036854775808,
                                base + relative,
                                num_indices,
                            ).to(tl.int32 if num_indices < 2147483647 else tl.int64)
                        else:
                            numeric = valid[:, None] & (values == values)
                            candidate = tl.where(numeric, values, float("-inf"))
                            tile_best = tl.max(candidate, 0)
                            tile_pos = tl.min(
                                tl.where(
                                    numeric & (values == tile_best[None, :]),
                                    position[:, None],
                                    num_indices,
                                ),
                                0,
                            )
                        replace = (tile_best > best) | (
                            (tile_best == best) & (tile_pos < best_pos)
                        )
                        best = tl.where(replace, tile_best, best)
                        best_pos = tl.where(replace, tile_pos, best_pos)
                    else:
                        if has_per_sample:
                            # CUDA ATen multiplies zero padding values by their sample
                            # weights, preserving NaN from a nonfinite padding weight.
                            scale = tl.load(
                                per_sample_weights + pos * stride_p,
                                valid if skip_pad_scale else active,
                                other=0.0,
                            )
                            values *= scale[:, None].to(acc_dtype)
                        acc += values
            if mode == 2:
                if num_indices == 0 or embedding_dim == 0:
                    winner = tl.full((block_d,), -1, tl.int64)
                    result = tl.zeros((block_d,), acc_dtype)
                else:
                    first_idx = tl.load(
                        indices + first.to(tl.int64) * stride_i,
                        first < num_indices,
                        other=0,
                    )
                    first_value = tl.load(
                        weight + first_idx.to(tl.int64) * stride_w0 + dim * stride_w1,
                        (first < num_indices)
                        & (first_idx >= 0)
                        & (first_idx < num_weights)
                        & (dim < embedding_dim),
                        other=0.0,
                    )
                    if key_max:
                        first_nan = (
                            first_value.to(tl.int64, bitcast=True) & 0x7FFFFFFFFFFFFFFF
                        ) > 0x7FF0000000000000
                        best_pos = tl.where(
                            first_nan, first.to(best_pos.dtype), best_pos
                        )
                    else:
                        best_pos = tl.where(first_value != first_value, first, best_pos)
                    winner = tl.load(
                        indices + best_pos.to(tl.int64) * stride_i,
                        best_pos < num_indices,
                        other=-1,
                    )
                    result = tl.load(
                        weight + winner.to(tl.int64) * stride_w0 + dim * stride_w1,
                        (winner >= 0) & (winner < num_weights) & (dim < embedding_dim),
                        other=0.0,
                    )
                if ascend_abi:
                    winner = tl.where(count == 0, 0, winner)
                if embedding_dim > 0:
                    tl.store(
                        max_indices + bag * embedding_dim + dim,
                        winner,
                        dim < embedding_dim,
                    )
            else:
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
    if num_offsets > num_bags and not bag_size_b:
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


logger = logging.getLogger(__name__)


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


def _check_error(error, count):
    # Keeping assertions out of the gather/reduction kernel avoids the MUSA
    # compiler's large valid-input slowdown while retaining device validation.
    _embedding_bag_check_flags[(1,)](error, count, 128, debug=True)


class _ReductionKernel:
    def __init__(self, serial, hint=False):
        self.serial = serial
        self.hint = hint
        self.entry = libentry()(_embedding_bag_forward_mthreads.jit_function)

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            kwargs["serial_max"] = self.serial
            kwargs["mapping_hint"] = self.hint
            return self.entry[grid](*args, **kwargs)

        return run


_KERNELS = {
    (serial, hint): _ReductionKernel(serial, hint)
    for serial in (False, True)
    for hint in (False, True)
}


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
    bags = offsets.numel() - int(include_last_offset)
    wide = (
        weight.ndim == 2
        and 512 <= weight.shape[1] <= 1024
        and bags >= 128
        and indices.numel() <= 16 * bags
    )
    medium = (
        weight.ndim == 2
        and 256 <= weight.shape[1] < 512
        and bags >= 128
        and indices.numel() <= 32 * bags
    )
    average = triton.cdiv(indices.numel(), max(bags, 1))
    mapping_hint = (
        weight.ndim == 2
        and weight.shape[1] <= 128
        and bags >= 64
        and (8 < average <= 16 or (mode == 2 and average >= 32))
    )
    block_config = None
    if mapping_hint:
        block_config = (
            16 if indices.numel() <= bags * 16 else 32,
            64,
            min(128, triton.next_power_of_2(max(average, 1))),
            4,
        )
    if wide:
        block_config = (2, 1024, 128, 4)
    elif medium:
        block_config = (4 if mode == 2 else 8, 256, 128, 4)
    return _embedding_bag_impl(
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
            _KERNELS[(mode == 2 if wide or medium else False, mapping_hint)]
            if mode == 2 or mapping_hint
            else None
        ),
        _block_config=block_config,
        _async_assert=False,
        _launch_kwargs={"skip_pad_scale": True},
        _check_error=_check_error,
    )


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
    return _embedding_bag_impl(
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
        _async_assert=False,
        _check_error=_check_error,
    )

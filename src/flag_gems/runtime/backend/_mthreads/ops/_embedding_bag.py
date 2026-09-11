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
def _embedding_bag_forward_mthreads(
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
    MAPPING_HINT: tl.constexpr = False,
):
    pid = tle.program_id(0)
    programs = tle.num_programs(0)
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
            if MAPPING_HINT:
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
            else:
                index_lane = tl.arange(0, INDEX_BLOCK)
                # Fixed partitions keep mapping writes disjoint even when a
                # malformed offset sequence makes different bags overlap.
                if N > 0:
                    for input_base in range(
                        bag * INDEX_BLOCK, N, tl.maximum(B, 1) * INDEX_BLOCK
                    ):
                        pos = input_base + index_lane
                        in_bounds = pos < N
                        idx = tl.load(Indices + pos * STRIDE_I, in_bounds, other=0)
                        invalid = in_bounds & ((idx < 0) | (idx >= V))
                        code |= tl.max(invalid.to(tl.int32), 0)
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
            if ASYNC_ASSERT:
                tl.device_assert(code == 0, "embedding_bag: invalid index or offset")
            else:
                tl.store(Error + bag, code)
        end = tl.where(valid_offsets, end, begin)
        begin = tl.minimum(tl.maximum(begin, 0), N)
        end = tl.minimum(tl.maximum(end, begin), N)
        if MODE == 2 and SERIAL_MAX:
            count = tl.full((), 0, tl.int64)
            best = tl.full((BLOCK_D,), 0, acc_dtype)
            winner = tl.full((BLOCK_D,), -1, tl.int64)
            # Unrolling keeps the original sequential comparison order. Restrict
            # it to the measured BF16/FP32 medium-width path; other paths retain
            # the scalar loop below.
            if (D >= 256 and D < 512) and (
                Weight.dtype.element_ty == tl.bfloat16
                or Weight.dtype.element_ty == tl.float32
            ):
                for base in range(begin, end, 4):
                    # Preserve input order and first-value NaN/tie semantics. Each
                    # indirect access retains its own bounds and padding mask.
                    for shift in tl.static_range(4):
                        pos = base + shift
                        active = (pos >= 0) & (pos < end)
                        idx = tl.load(Indices + pos * STRIDE_I, active, other=0).to(
                            tl.int64
                        )
                        valid = active & (idx >= 0) & (idx < V) & (idx != PADDING)
                        values = tl.load(
                            Weight + idx * STRIDE_W0 + dim * STRIDE_W1,
                            valid & (dim < D),
                            other=0.0,
                        ).to(acc_dtype)
                        replace = valid & ((count == 0) | (values > best))
                        best = tl.where(replace, values, best)
                        winner = tl.where(replace, idx, winner)
                        count += valid.to(tl.int64)
            else:
                for pos in range(begin, end):
                    idx = tl.load(Indices + pos * STRIDE_I).to(tl.int64)
                    valid = (idx >= 0) & (idx < V) & (idx != PADDING)
                    values = tl.load(
                        Weight + idx * STRIDE_W0 + dim * STRIDE_W1,
                        valid & (dim < D),
                        other=0.0,
                    ).to(acc_dtype)
                    replace = valid & ((count == 0) | (values > best))
                    best = tl.where(replace, values, best)
                    winner = tl.where(replace, idx, winner)
                    count += valid.to(tl.int64)
            result = best
            tl.store(MaxIndices + bag * D + dim, winner, dim < D)
        else:
            count = tl.full((), 0, tl.int64)
            if MODE == 2:
                if KEY_MAX:
                    best = tl.full((BLOCK_D,), -9223372036854775808, tl.int64)
                else:
                    best = tl.full((BLOCK_D,), float("-inf"), acc_dtype)
                # Position values never exceed the N sentinel. Pointer arithmetic
                # is promoted back to int64 before applying arbitrary input strides.
                position_dtype: tl.constexpr = tl.int32 if N < 2147483647 else tl.int64
                best_pos = tl.full((BLOCK_D,), N, position_dtype)
                first = tl.full((), N, position_dtype)
            else:
                acc = tl.zeros((BLOCK_L, BLOCK_D), acc_dtype)
            if N > 0:
                for base in range(begin, end, BLOCK_L):
                    pos = base + rows
                    active = pos < end
                    idx = tl.load(Indices + pos * STRIDE_I, active, other=0).to(
                        tl.int64
                    )
                    valid = active & (idx >= 0) & (idx < V) & (idx != PADDING)
                    if D > 0:
                        values = tl.load(
                            Weight
                            + idx[:, None] * STRIDE_W0
                            + dim[None, :] * STRIDE_W1,
                            valid[:, None] & (dim[None, :] < D),
                            other=0.0,
                        ).to(acc_dtype)
                    else:
                        values = tl.zeros((BLOCK_L, BLOCK_D), acc_dtype)
                    count += tl.sum(valid.to(tl.int32), 0)
                    if MODE == 2:
                        position = pos.to(position_dtype)
                        first = tl.minimum(
                            first, tl.min(tl.where(valid, position, N), 0)
                        )
                        if KEY_MAX:
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
                                tile_best > -9223372036854775808, base + relative, N
                            ).to(tl.int32 if N < 2147483647 else tl.int64)
                        else:
                            numeric = valid[:, None] & (values == values)
                            candidate = tl.where(numeric, values, float("-inf"))
                            tile_best = tl.max(candidate, 0)
                            tile_pos = tl.min(
                                tl.where(
                                    numeric & (values == tile_best[None, :]),
                                    position[:, None],
                                    N,
                                ),
                                0,
                            )
                        replace = (tile_best > best) | (
                            (tile_best == best) & (tile_pos < best_pos)
                        )
                        best = tl.where(replace, tile_best, best)
                        best_pos = tl.where(replace, tile_pos, best_pos)
                    else:
                        if HAS_PER_SAMPLE:
                            # CUDA ATen multiplies zero padding values by their sample
                            # weights, preserving NaN from a nonfinite padding weight.
                            scale = tl.load(
                                PerSample + pos * STRIDE_P,
                                valid if SKIP_PAD_SCALE else active,
                                other=0.0,
                            )
                            values *= scale[:, None].to(acc_dtype)
                        acc += values
            if MODE == 2:
                if N == 0 or D == 0:
                    winner = tl.full((BLOCK_D,), -1, tl.int64)
                    result = tl.zeros((BLOCK_D,), acc_dtype)
                else:
                    first_idx = tl.load(
                        Indices + first.to(tl.int64) * STRIDE_I, first < N, other=0
                    )
                    first_value = tl.load(
                        Weight + first_idx.to(tl.int64) * STRIDE_W0 + dim * STRIDE_W1,
                        (first < N) & (first_idx >= 0) & (first_idx < V) & (dim < D),
                        other=0.0,
                    )
                    if KEY_MAX:
                        first_nan = (
                            first_value.to(tl.int64, bitcast=True) & 0x7FFFFFFFFFFFFFFF
                        ) > 0x7FF0000000000000
                        best_pos = tl.where(
                            first_nan, first.to(best_pos.dtype), best_pos
                        )
                    else:
                        best_pos = tl.where(first_value != first_value, first, best_pos)
                    winner = tl.load(
                        Indices + best_pos.to(tl.int64) * STRIDE_I,
                        best_pos < N,
                        other=-1,
                    )
                    result = tl.load(
                        Weight + winner.to(tl.int64) * STRIDE_W0 + dim * STRIDE_W1,
                        (winner >= 0) & (winner < V) & (dim < D),
                        other=0.0,
                    )
                if ASCEND_ABI:
                    winner = tl.where(count == 0, 0, winner)
                if D > 0:
                    tl.store(MaxIndices + bag * D + dim, winner, dim < D)
            else:
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
    if O > B and not BAG_SIZE_B:
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


logger = logging.getLogger(__name__)


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
            kwargs["SERIAL_MAX"] = self.serial
            kwargs["MAPPING_HINT"] = self.hint
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
        _launch_kwargs={"SKIP_PAD_SCALE": True},
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

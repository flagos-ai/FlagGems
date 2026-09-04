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

"""THead / PPU W8A16 TopK.

Input is grouped FP8 E5M2 plus per-group scale (group_size=128). The kernel
dequantizes on the fly and returns BF16 values and int64 indices along the
last dimension.
"""

import logging
import math

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.ops.topk import (
    _MAX_INT32_VAL,
    _MIN_INT32_VAL,
    _MIN_INT64_VAL,
    argsort,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

try:
    import triton.experimental.tle.language as tle_gpu

    HAS_TLE_GPU = hasattr(tle_gpu, "gpu")
except ImportError:
    tle_gpu = None
    HAS_TLE_GPU = False

logger = logging.getLogger(__name__)


@triton.jit
def _fp8_e5_to_f32(x):
    return x.to(tl.float8e5, bitcast=True).to(tl.float32)


@libentry()
@triton.jit
def dequant_fp8_e5_kernel(
    out_ptr,
    x_ptr,
    scale_ptr,
    N,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tle.program_id(0)
    cols = tle.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    x_q = _fp8_e5_to_f32(tl.load(x_ptr + row * N + cols, mask=mask, other=0))
    x_scale = tl.load(
        scale_ptr + row * tl.cdiv(N, GROUP_SIZE) + cols // GROUP_SIZE,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(out_ptr + row * N + cols, x_q * x_scale, mask=mask)


@libentry()
@triton.jit
def topk_fp8_single_stage_kernel(
    y_ptr,
    index_ptr,
    x_ptr,
    scale_ptr,
    k: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DESCENDING: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
):
    cur_batch = tle.program_id(0)
    x_ptr += cur_batch * N
    scale_ptr += cur_batch * NUM_GROUPS
    y_ptr += cur_batch * k
    index_ptr += cur_batch * k

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    pad_val = float("-inf") if DESCENDING else float("inf")
    mask_index_val = _MIN_INT32_VAL if DESCENDING else _MAX_INT32_VAL

    x_q = _fp8_e5_to_f32(tl.load(x_ptr + cols, mask=mask, other=0))
    x_scale = tl.load(scale_ptr + cols // GROUP_SIZE, mask=mask, other=0.0).to(
        tl.float32
    )
    x_val = tl.where(mask, x_q * x_scale, pad_val)
    idx_val = tl.where(mask, cols, mask_index_val).to(tl.int32)

    sorted_x, sorted_idx = argsort(x_val, idx_val, dim=0, descending=DESCENDING)

    out_mask = cols < k
    tl.store(y_ptr + cols, sorted_x, mask=out_mask)
    tl.store(index_ptr + cols, sorted_idx.to(tl.int64), mask=out_mask)


@triton.jit
def _merge_sorted_topk(
    best_val,
    best_idx,
    tile_val,
    tile_idx,
    BLOCK: tl.constexpr,
    DESCENDING: tl.constexpr,
):
    sval, sidx = argsort(tile_val, tile_idx, 0, DESCENDING)
    row0 = tl.arange(0, 2)[:, None] == 0
    mval = tl.where(row0, best_val.reshape(1, BLOCK), sval.reshape(1, BLOCK)).reshape(
        2 * BLOCK
    )
    midx = tl.where(row0, best_idx.reshape(1, BLOCK), sidx.reshape(1, BLOCK)).reshape(
        2 * BLOCK
    )
    mval, midx = argsort(mval, midx, 0, DESCENDING)
    mval2 = mval.reshape(2, BLOCK)
    midx2 = midx.reshape(2, BLOCK)
    out_val = tl.sum(tl.where(row0, mval2, 0.0), axis=0)
    out_idx = tl.sum(tl.where(row0, midx2.to(tl.int64), 0), axis=0).to(tl.int32)
    return out_val, out_idx


@libentry()
@triton.jit
def topk_fp8_running_merge_kernel(
    y_ptr,
    index_ptr,
    x_ptr,
    scale_ptr,
    N,
    k: tl.constexpr,
    BLOCK: tl.constexpr,
    DESCENDING: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    x_ptr += pid * N
    scale_ptr += pid * tl.cdiv(N, GROUP_SIZE)
    y_ptr += pid * k
    index_ptr += pid * k

    pad_val = float("-inf") if DESCENDING else float("inf")
    pad_idx = _MIN_INT32_VAL if DESCENDING else _MAX_INT32_VAL
    offs = tl.arange(0, BLOCK)
    best_val = tl.full([BLOCK], pad_val, dtype=tl.float32)
    best_idx = tl.full([BLOCK], pad_idx, dtype=tl.int32)

    n_tiles = tl.cdiv(N, BLOCK)
    for t in tl.range(0, n_tiles):
        cols = t * BLOCK + offs
        mask = cols < N
        x_q = _fp8_e5_to_f32(tl.load(x_ptr + cols, mask=mask, other=0))
        x_scale = tl.load(scale_ptr + cols // GROUP_SIZE, mask=mask, other=0.0).to(
            tl.float32
        )
        tile_val = tl.where(mask, x_q * x_scale, pad_val)
        tile_idx = tl.where(mask, cols, pad_idx).to(tl.int32)
        best_val, best_idx = _merge_sorted_topk(
            best_val, best_idx, tile_val, tile_idx, BLOCK, DESCENDING
        )

    out_mask = offs < k
    tl.store(y_ptr + offs, best_val, mask=out_mask)
    tl.store(index_ptr + offs, best_idx.to(tl.int64), mask=out_mask)


@libentry()
@triton.jit
def topk_fp8_stage1_kernel(
    y_ptr,
    index_ptr,
    x_ptr,
    scale_ptr,
    k,
    N: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    DESCENDING: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
):
    cur_batch = tle.program_id(0)
    cur_chunk_idx = tle.program_id(1)
    chunk_num = tle.num_programs(1)

    y_ptr += cur_batch * chunk_num * k + cur_chunk_idx * k
    index_ptr += cur_batch * chunk_num * k + cur_chunk_idx * k

    chunk_offset = cur_chunk_idx * CHUNK_SIZE
    x_ptr += cur_batch * N + chunk_offset
    scale_ptr += cur_batch * NUM_GROUPS

    cols = tl.arange(0, CHUNK_SIZE)
    global_cols = chunk_offset + cols
    mask = global_cols < N

    mask_val = float("-inf") if DESCENDING else float("inf")
    x_q = _fp8_e5_to_f32(tl.load(x_ptr + cols, mask=mask, other=0))
    x_scale = tl.load(scale_ptr + global_cols // GROUP_SIZE, mask=mask, other=0.0).to(
        tl.float32
    )
    x_val = tl.where(mask, x_q * x_scale, mask_val)
    available = mask

    for k_idx in range(k):
        if DESCENDING:
            chunk_select_val = tl.max(x_val)
        else:
            chunk_select_val = tl.min(x_val)
        is_candidate = available & (x_val == chunk_select_val)
        candidate_indices = tl.where(is_candidate, cols, CHUNK_SIZE)
        chunk_select_idx = tl.argmin(candidate_indices, axis=0)

        tl.store(y_ptr + k_idx, chunk_select_val)
        tl.store(index_ptr + k_idx, chunk_select_idx + chunk_offset)
        if DESCENDING:
            x_val = tl.where(cols == chunk_select_idx, float("-inf"), x_val)
        else:
            x_val = tl.where(cols == chunk_select_idx, float("inf"), x_val)
        available = available & (cols != chunk_select_idx)


@triton.jit
def _fp8_bits_to_ordered_key(bits):
    sign = bits & 0x80
    flip_mask = tl.where(sign != 0, 0xFF, 0x80).to(tl.uint8)
    return bits ^ flip_mask


@libentry()
@triton.jit
def topk_fp8_one_group_packed_kernel(
    y_ptr,
    index_ptr,
    x_ptr,
    scale_ptr,
    k: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    bits = tl.load(x_ptr + pid * N + cols, mask=mask, other=0).to(tl.uint8)
    values = _fp8_e5_to_f32(bits)
    ordered_key = _fp8_bits_to_ordered_key(bits)
    ordered_key = tl.where(values != values, 0xFF, ordered_key).to(tl.uint8)
    index_key = (0xFFFF - cols).to(tl.uint32)
    packed = (ordered_key.to(tl.uint32) << 16) | index_key
    packed = tl.where(mask, packed, 0)
    packed = tl.sort(packed, dim=0, descending=True)

    out_mask = cols < k
    selected_key = (packed >> 16).to(tl.uint8)
    selected_flip = tl.where((selected_key & 0x80) != 0, 0x80, 0xFF).to(tl.uint8)
    selected_bits = selected_key ^ selected_flip
    selected_values = _fp8_e5_to_f32(selected_bits)
    scale = tl.load(scale_ptr + pid).to(tl.float32)
    selected_indices = (0xFFFF - (packed & 0xFFFF)).to(tl.int64)
    tl.store(
        y_ptr + pid * k + cols,
        selected_values * scale,
        mask=out_mask,
    )
    tl.store(index_ptr + pid * k + cols, selected_indices, mask=out_mask)


@libentry()
@triton.jit
def topk_fp8_two_group_packed_kernel(
    y_ptr,
    index_ptr,
    x_ptr,
    scale_ptr,
    k: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    local = tl.arange(0, GROUP_SIZE)
    row_start = pid * 2 * GROUP_SIZE

    bits_0 = tl.load(x_ptr + row_start + local).to(tl.uint8)
    bits_1 = tl.load(x_ptr + row_start + GROUP_SIZE + local).to(tl.uint8)
    values_0 = _fp8_e5_to_f32(bits_0)
    values_1 = _fp8_e5_to_f32(bits_1)
    key_0 = _fp8_bits_to_ordered_key(bits_0)
    key_1 = _fp8_bits_to_ordered_key(bits_1)
    key_0 = tl.where(values_0 != values_0, 0xFF, key_0).to(tl.uint8)
    key_1 = tl.where(values_1 != values_1, 0xFF, key_1).to(tl.uint8)
    packed_0 = (key_0.to(tl.uint32) << 16) | (0xFFFF - local).to(tl.uint32)
    packed_1 = (key_1.to(tl.uint32) << 16) | (0xFFFF - GROUP_SIZE - local).to(tl.uint32)
    packed_0 = tl.sort(packed_0, dim=0, descending=True)
    packed_1 = tl.sort(packed_1, dim=0, descending=True)

    candidate_offsets = tl.arange(0, 2 * k)
    group_ranks = candidate_offsets % k
    selected_0 = tl.gather(packed_0, group_ranks, axis=0)
    selected_1 = tl.gather(packed_1, group_ranks, axis=0)
    selected = tl.where(candidate_offsets < k, selected_0, selected_1)

    raw_key = (selected >> 16).to(tl.uint8)
    raw_flip = tl.where((raw_key & 0x80) != 0, 0x80, 0xFF).to(tl.uint8)
    raw_bits = raw_key ^ raw_flip
    raw_values = _fp8_e5_to_f32(raw_bits)
    group_idx = (candidate_offsets >= k).to(tl.int32)
    scale = tl.load(scale_ptr + pid * 2 + group_idx).to(tl.float32)
    values = (raw_values * scale).to(tl.bfloat16)
    indices = (0xFFFF - (selected & 0xFFFF)).to(tl.int32)

    value_bits = values.to(tl.uint16, bitcast=True)
    value_flip = tl.where((value_bits & 0x8000) != 0, 0xFFFF, 0x8000).to(tl.uint16)
    value_key = value_bits ^ value_flip
    value_key = tl.where(values != values, 0xFFFF, value_key).to(tl.uint16)
    merged = (value_key.to(tl.uint32) << 16) | (0xFFFF - indices).to(tl.uint32)
    merged = tl.sort(merged, dim=0, descending=True)

    out_mask = candidate_offsets < k
    out_key = (merged >> 16).to(tl.uint16)
    out_flip = tl.where((out_key & 0x8000) != 0, 0x8000, 0xFFFF).to(tl.uint16)
    out_bits = out_key ^ out_flip
    final_values = out_bits.to(tl.bfloat16, bitcast=True)
    final_indices = (0xFFFF - (merged & 0xFFFF)).to(tl.int64)
    tl.store(
        y_ptr + pid * k + candidate_offsets,
        final_values,
        mask=out_mask,
    )
    tl.store(
        index_ptr + pid * k + candidate_offsets,
        final_indices,
        mask=out_mask,
    )


@libentry()
@triton.jit
def topk_fp8_row_radix_threshold_kernel(
    x_ptr,
    threshold_key_ptr,
    counter_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    bins = tl.arange(0, 16)
    desired = tl.full((), 0, dtype=tl.uint8)
    desired_mask = tl.full((), 0, dtype=tl.uint8)
    k_to_find = tl.full((), K, dtype=tl.int32)
    n_tiles = tl.cdiv(N, BLOCK_N)

    for digit_iter in tl.static_range(0, 2):
        shift = 4 - digit_iter * 4
        counts = tl.zeros((16,), dtype=tl.int32)

        for tile in tl.range(0, n_tiles):
            offs = tile * BLOCK_N + tl.arange(0, BLOCK_N)
            mask = offs < N
            q = tl.load(x_ptr + pid * N + offs, mask=mask, other=0)
            bits = q.to(tl.uint8, bitcast=True)
            key = _fp8_bits_to_ordered_key(bits)
            matches = (key & desired_mask) == desired
            digit = ((key >> shift) & 0xF).to(tl.int32)
            valid = mask & matches
            counts += tl.sum(
                tl.where(
                    (digit[None, :] == bins[:, None]) & valid[None, :],
                    1,
                    0,
                ),
                axis=1,
            )

        cumsum_desc = tl.cumsum(counts, axis=0, reverse=True)
        selected = tl.full((), 0, dtype=tl.int32)
        counts_gt = tl.full((), 0, dtype=tl.int32)
        found = tl.full((), 0, dtype=tl.int32)

        for rev in tl.static_range(0, 16):
            digit_value = 15 - rev
            cum_d = tl.sum(tl.where(bins == digit_value, cumsum_desc, 0))
            if digit_value + 1 < 16:
                cum_next = tl.sum(tl.where(bins == digit_value + 1, cumsum_desc, 0))
            else:
                cum_next = tl.full((), 0, dtype=tl.int32)
            take = (found == 0) & (cum_d >= k_to_find) & (cum_next < k_to_find)
            selected = tl.where(take, digit_value, selected)
            counts_gt = tl.where(take, cum_next, counts_gt)
            found = tl.where(take, 1, found)

        selected_u8 = selected.to(tl.uint8)
        desired = desired | (selected_u8 << shift)
        desired_mask = desired_mask | (tl.full((), 0xF, dtype=tl.uint8) << shift)
        k_to_find = k_to_find - counts_gt

    tl.store(threshold_key_ptr + pid, desired)
    tl.store(counter_ptr + pid, 0)


@libentry()
@triton.jit
def topk_fp8_row_radix_high_hist_kernel(
    high_hist_ptr,
    x_ptr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_TILES: tl.constexpr,
):
    pid = tle.program_id(0)
    tile = tle.program_id(1)
    bins = tl.arange(0, 16)
    offs = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N

    q = tl.load(x_ptr + pid * N + offs, mask=mask, other=0)
    bits = q.to(tl.uint8, bitcast=True)
    key = _fp8_bits_to_ordered_key(bits)
    digit = ((key >> 4) & 0xF).to(tl.int32)
    counts = tl.sum(
        tl.where((digit[None, :] == bins[:, None]) & mask[None, :], 1, 0),
        axis=1,
    )
    tl.store(high_hist_ptr + (pid * N_TILES + tile) * 16 + bins, counts)


@libentry()
@triton.jit
def topk_fp8_row_radix_high_reduce_kernel(
    selected_high_ptr,
    k_remaining_ptr,
    high_hist_ptr,
    K: tl.constexpr,
    N_TILES: tl.constexpr,
):
    pid = tle.program_id(0)
    bins = tl.arange(0, 16)
    counts = tl.zeros((16,), dtype=tl.int32)

    for tile in tl.range(0, N_TILES):
        counts += tl.load(high_hist_ptr + (pid * N_TILES + tile) * 16 + bins)

    cumsum_desc = tl.cumsum(counts, axis=0, reverse=True)
    selected = tl.full((), 0, dtype=tl.int32)
    counts_gt = tl.full((), 0, dtype=tl.int32)
    found = tl.full((), 0, dtype=tl.int32)

    for rev in tl.static_range(0, 16):
        digit_value = 15 - rev
        cum_d = tl.sum(tl.where(bins == digit_value, cumsum_desc, 0))
        if digit_value + 1 < 16:
            cum_next = tl.sum(tl.where(bins == digit_value + 1, cumsum_desc, 0))
        else:
            cum_next = tl.full((), 0, dtype=tl.int32)
        take = (found == 0) & (cum_d >= K) & (cum_next < K)
        selected = tl.where(take, digit_value, selected)
        counts_gt = tl.where(take, cum_next, counts_gt)
        found = tl.where(take, 1, found)

    tl.store(selected_high_ptr + pid, selected.to(tl.uint8))
    tl.store(k_remaining_ptr + pid, K - counts_gt)


@libentry()
@triton.jit
def topk_fp8_row_radix_low_hist_kernel(
    low_hist_ptr,
    x_ptr,
    selected_high_ptr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_TILES: tl.constexpr,
):
    pid = tle.program_id(0)
    tile = tle.program_id(1)
    bins = tl.arange(0, 16)
    offs = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N

    selected_high = tl.load(selected_high_ptr + pid).to(tl.int32)
    q = tl.load(x_ptr + pid * N + offs, mask=mask, other=0)
    bits = q.to(tl.uint8, bitcast=True)
    key = _fp8_bits_to_ordered_key(bits)
    high = ((key >> 4) & 0xF).to(tl.int32)
    low = (key & 0xF).to(tl.int32)
    valid = mask & (high == selected_high)
    counts = tl.sum(
        tl.where((low[None, :] == bins[:, None]) & valid[None, :], 1, 0),
        axis=1,
    )
    tl.store(low_hist_ptr + (pid * N_TILES + tile) * 16 + bins, counts)


@libentry()
@triton.jit
def topk_fp8_row_radix_low_reduce_kernel(
    threshold_key_ptr,
    counter_ptr,
    low_hist_ptr,
    selected_high_ptr,
    k_remaining_ptr,
    N_TILES: tl.constexpr,
):
    pid = tle.program_id(0)
    bins = tl.arange(0, 16)
    counts = tl.zeros((16,), dtype=tl.int32)

    for tile in tl.range(0, N_TILES):
        counts += tl.load(low_hist_ptr + (pid * N_TILES + tile) * 16 + bins)

    k_to_find = tl.load(k_remaining_ptr + pid)
    cumsum_desc = tl.cumsum(counts, axis=0, reverse=True)
    selected_low = tl.full((), 0, dtype=tl.int32)
    found = tl.full((), 0, dtype=tl.int32)

    for rev in tl.static_range(0, 16):
        digit_value = 15 - rev
        cum_d = tl.sum(tl.where(bins == digit_value, cumsum_desc, 0))
        if digit_value + 1 < 16:
            cum_next = tl.sum(tl.where(bins == digit_value + 1, cumsum_desc, 0))
        else:
            cum_next = tl.full((), 0, dtype=tl.int32)
        take = (found == 0) & (cum_d >= k_to_find) & (cum_next < k_to_find)
        selected_low = tl.where(take, digit_value, selected_low)
        found = tl.where(take, 1, found)

    high = tl.load(selected_high_ptr + pid).to(tl.uint8)
    threshold = (high << 4) | selected_low.to(tl.uint8)
    tl.store(threshold_key_ptr + pid, threshold)
    tl.store(counter_ptr + pid, 0)


@libentry()
@triton.jit
def topk_fp8_row_radix_collect_kernel(
    candidate_val_ptr,
    candidate_idx_ptr,
    counter_ptr,
    x_ptr,
    threshold_key_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TAKE_EQUAL: tl.constexpr,
):
    pid = tle.program_id(0)
    tile = tle.program_id(1)
    offs = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N

    q = tl.load(x_ptr + pid * N + offs, mask=mask, other=0)
    bits = q.to(tl.uint8, bitcast=True)
    key = _fp8_bits_to_ordered_key(bits)
    threshold = tl.load(threshold_key_ptr + pid)
    if TAKE_EQUAL:
        take = mask & (key == threshold)
    else:
        take = mask & (key > threshold)

    counter_offsets = tl.zeros((BLOCK_N,), dtype=tl.int64)
    old_pos = tl.atomic_add(
        counter_ptr + pid + counter_offsets, 1, sem="relaxed", mask=take
    )
    store_mask = take & (old_pos < K)
    tl.store(
        candidate_val_ptr + pid * K + old_pos,
        _fp8_e5_to_f32(q),
        mask=store_mask,
    )
    tl.store(
        candidate_idx_ptr + pid * K + old_pos,
        offs.to(tl.int64),
        mask=store_mask,
    )


@libentry()
@triton.jit
def topk_fp8_row_radix_sort_kernel(
    y_ptr,
    index_ptr,
    candidate_val_ptr,
    candidate_idx_ptr,
    scale_ptr,
    K: tl.constexpr,
    K_PAD: tl.constexpr,
):
    pid = tle.program_id(0)
    offs = tl.arange(0, K_PAD)
    mask = offs < K
    vals = tl.load(
        candidate_val_ptr + pid * K + offs,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    idx = tl.load(
        candidate_idx_ptr + pid * K + offs,
        mask=mask,
        other=_MIN_INT64_VAL,
    ).to(tl.int64)
    sorted_vals, sorted_idx = argsort(vals, idx, dim=0, descending=True)
    scale = tl.load(scale_ptr + pid).to(tl.float32)
    tl.store(y_ptr + pid * K + offs, sorted_vals * scale, mask=mask)
    tl.store(index_ptr + pid * K + offs, sorted_idx, mask=mask)


if HAS_TLE_GPU:

    @triton.jit
    def _get_topmask_and_fullmask(x):
        tl.static_assert(
            x.dtype.is_int_unsigned(),
            "floating-point value must be passed as bits",
        )
        tm: tl.constexpr = 1 << (-1 + x.dtype.primitive_bitwidth)
        fm: tl.constexpr = (1 << x.dtype.primitive_bitwidth) - 1
        tm_arr = tl.full(x.shape, tm, dtype=x.dtype)
        fm_arr = tl.full(x.shape, fm, dtype=x.dtype)
        return tm_arr, fm_arr

    @triton.jit
    def _key_to_fpval(x):
        tm, fm = _get_topmask_and_fullmask(x)
        mask = tl.where((x & tm) != 0, tm, fm)
        return x ^ mask

    @triton.jit
    def _fpval_to_key_with_nan(x, x_bits):
        tm, fm = _get_topmask_and_fullmask(x_bits)
        mask = tl.where((x_bits & tm) != 0, fm, tm)
        key = x_bits ^ mask
        return tl.where(x == x, key, fm)

    @triton.jit
    def _load_dequant_fp8_row(
        X,
        Scale,
        pid,
        stride_xm,
        stride_sm,
        offs_n,
        mask_n,
        GROUP_SIZE: tl.constexpr,
    ):
        q = tl.load(X + pid * stride_xm + offs_n, mask=mask_n, other=0)
        scale = tl.load(
            Scale + pid * stride_sm + offs_n // GROUP_SIZE, mask=mask_n, other=0.0
        ).to(tl.float32)
        x = (_fp8_e5_to_f32(q) * scale).to(tl.bfloat16)
        return tl.where(mask_n, x, float("-inf")).to(tl.bfloat16)

    @triton.jit
    def _row_keys(
        X,
        Scale,
        Key,
        pid,
        stride_xm,
        stride_sm,
        stride_km,
        offs_n,
        mask_n,
        GROUP_SIZE: tl.constexpr,
        HAS_KEY_BUF: tl.constexpr,
        WRITE_KEY: tl.constexpr,
    ):
        if HAS_KEY_BUF and not WRITE_KEY:
            return tl.load(Key + pid * stride_km + offs_n, mask=mask_n, other=0)
        x = _load_dequant_fp8_row(
            X, Scale, pid, stride_xm, stride_sm, offs_n, mask_n, GROUP_SIZE
        )
        x_key = _fpval_to_key_with_nan(x, x.to(tl.uint16, bitcast=True))
        if HAS_KEY_BUF and WRITE_KEY:
            tl.store(Key + pid * stride_km + offs_n, x_key, mask=mask_n)
        return x_key

    @libentry()
    @triton.jit
    def topk_fp8_row_radix_tle_kernel(
        X,
        Scale,
        Key,
        Yv,
        Yi,
        stride_xm,
        stride_sm,
        stride_km,
        stride_ym,
        n_cols,
        K: tl.constexpr,
        K_PAD: tl.constexpr,
        BLOCK_N: tl.constexpr,
        RADIX_BITS: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        HAS_KEY_BUF: tl.constexpr,
    ):
        pid = tl.program_id(0)
        x_utype = tl.uint16
        x_ultype = tl.uint32

        RADIX_SIZE: tl.constexpr = 1 << RADIX_BITS
        RADIX_MASK: tl.constexpr = RADIX_SIZE - 1
        bins = tl.arange(0, RADIX_SIZE)
        one = tl.full([BLOCK_N], 1, tl.int32)

        desired = tl.full((), 0, dtype=x_utype)
        desired_mask = tl.full((), 0, dtype=x_utype)
        k_to_find = tl.full((), K, dtype=tl.int32)
        n_tiles = tl.cdiv(n_cols, BLOCK_N)

        if HAS_KEY_BUF:
            for t in tl.range(0, n_tiles):
                offs_n = t * BLOCK_N + tl.arange(0, BLOCK_N)
                mask_n = offs_n < n_cols
                _row_keys(
                    X,
                    Scale,
                    Key,
                    pid,
                    stride_xm,
                    stride_sm,
                    stride_km,
                    offs_n,
                    mask_n,
                    GROUP_SIZE,
                    HAS_KEY_BUF,
                    True,
                )

        smem_counts = tle_gpu.gpu.alloc(
            [RADIX_SIZE],
            dtype=tl.int32,
            layout=None,
            scope=tle_gpu.gpu.smem,
            nv_mma_shared_layout=False,
        )
        smem_count_ptrs = tle_gpu.gpu.local_ptr(smem_counts, (bins,))

        for digit_pos in tl.static_range(16 - RADIX_BITS, -1, -RADIX_BITS):
            tl.store(smem_count_ptrs, tl.zeros([RADIX_SIZE], dtype=tl.int32))
            for t in tl.range(0, n_tiles):
                offs_n = t * BLOCK_N + tl.arange(0, BLOCK_N)
                mask_n = offs_n < n_cols
                x_key = _row_keys(
                    X,
                    Scale,
                    Key,
                    pid,
                    stride_xm,
                    stride_sm,
                    stride_km,
                    offs_n,
                    mask_n,
                    GROUP_SIZE,
                    HAS_KEY_BUF,
                    False,
                )
                matches = (x_key & desired_mask) == desired
                digit = ((x_key >> digit_pos) & RADIX_MASK).to(tl.int32)
                valid = mask_n & matches
                count_addrs = tle_gpu.gpu.local_ptr(smem_counts, (digit,))
                tl.atomic_add(count_addrs, one, mask=valid, sem="relaxed", scope="cta")

            counts = tl.load(smem_count_ptrs)
            cumsum_desc = tl.cumsum(counts, axis=0, reverse=True)
            tl.store(smem_count_ptrs, cumsum_desc)

            selected_scalar = 0
            counts_gt_scalar = 0
            found = 0
            for rev in tl.static_range(RADIX_SIZE):
                d = RADIX_SIZE - 1 - rev
                cum_d = tl.load(tle_gpu.gpu.local_ptr(smem_counts, (d,)))
                if d + 1 < RADIX_SIZE:
                    cum_next = tl.load(tle_gpu.gpu.local_ptr(smem_counts, (d + 1,)))
                else:
                    cum_next = 0
                take = (found == 0) & (cum_d >= k_to_find) & (cum_next < k_to_find)
                selected_scalar = tl.where(take, d, selected_scalar)
                counts_gt_scalar = tl.where(take, cum_next, counts_gt_scalar)
                found = tl.where(take, 1, found)

            selected_u = selected_scalar.to(x_utype)
            desired = desired | (selected_u << digit_pos)
            desired_mask = desired_mask | (
                tl.full((), RADIX_MASK, dtype=x_utype) << digit_pos
            )
            k_to_find = k_to_find - counts_gt_scalar

        thr_key = desired
        min_val = tl.full((), float("-inf"), tl.bfloat16)
        min_bits = min_val.to(x_utype, bitcast=True)
        min_key = _fpval_to_key_with_nan(min_val, min_bits)
        min_packed = min_key.to(x_ultype) << 16
        offs_k = tl.arange(0, K_PAD)

        smem_selected = tle_gpu.gpu.alloc(
            [K_PAD],
            dtype=x_ultype,
            layout=None,
            scope=tle_gpu.gpu.smem,
            nv_mma_shared_layout=False,
        )
        smem_selected_ptrs = tle_gpu.gpu.local_ptr(smem_selected, (offs_k,))
        tl.store(smem_selected_ptrs, tl.full([K_PAD], min_packed, dtype=x_ultype))

        smem_write_count = tle_gpu.gpu.alloc(
            [1],
            dtype=tl.int32,
            layout=None,
            scope=tle_gpu.gpu.smem,
            nv_mma_shared_layout=False,
        )
        tl.store(tle_gpu.gpu.local_ptr(smem_write_count, (0,)), 0)
        write_count_ptrs = tle_gpu.gpu.local_ptr(
            smem_write_count, (tl.zeros([BLOCK_N], dtype=tl.int32),)
        )

        for t in tl.range(0, n_tiles):
            offs_n = t * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_n = offs_n < n_cols
            x_key = _row_keys(
                X,
                Scale,
                Key,
                pid,
                stride_xm,
                stride_sm,
                stride_km,
                offs_n,
                mask_n,
                GROUP_SIZE,
                HAS_KEY_BUF,
                False,
            )
            idx_key = (n_cols - offs_n).to(x_ultype)
            packed = (x_key.to(x_ultype) << 16) | idx_key
            take_gt = mask_n & (x_key > thr_key)
            pos = tl.atomic_add(
                write_count_ptrs, one, mask=take_gt, sem="relaxed", scope="cta"
            )
            write_mask = take_gt & (pos < K_PAD)
            dst_ptrs = tle_gpu.gpu.local_ptr(smem_selected, (pos.to(tl.int32),))
            tl.store(dst_ptrs, packed, mask=write_mask)

        for t in tl.range(0, n_tiles):
            offs_n = t * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_n = offs_n < n_cols
            x_key = _row_keys(
                X,
                Scale,
                Key,
                pid,
                stride_xm,
                stride_sm,
                stride_km,
                offs_n,
                mask_n,
                GROUP_SIZE,
                HAS_KEY_BUF,
                False,
            )
            idx_key = (n_cols - offs_n).to(x_ultype)
            packed = (x_key.to(x_ultype) << 16) | idx_key
            take_eq = mask_n & (x_key == thr_key)
            pos = tl.atomic_add(
                write_count_ptrs, one, mask=take_eq, sem="relaxed", scope="cta"
            )
            write_mask = take_eq & (pos < K_PAD)
            dst_ptrs = tle_gpu.gpu.local_ptr(smem_selected, (pos.to(tl.int32),))
            tl.store(dst_ptrs, packed, mask=write_mask)

        selected_packed = tl.load(smem_selected_ptrs)
        topk = tl.sort(selected_packed, dim=0, descending=True)
        idx_mask = tl.full(topk.shape, (1 << 16) - 1, dtype=topk.dtype)
        idx_raw = (topk & idx_mask).to(tl.uint32)
        y_indices = (n_cols - idx_raw.to(tl.int32)).to(tl.int64)
        y_keys = (topk >> 16).to(x_utype)
        y_values = _key_to_fpval(y_keys).to(tl.bfloat16, bitcast=True)

        mask_k = offs_k < K
        tl.store(Yv + pid * stride_ym + offs_k, y_values, mask=mask_k)
        tl.store(Yi + pid * stride_ym + offs_k, y_indices, mask=mask_k)

    @triton.jit
    def _select_desc_bin(smem_counts, k_to_find, RADIX_SIZE: tl.constexpr):
        selected_scalar = 0
        counts_gt_scalar = 0
        found = 0
        for rev in tl.static_range(RADIX_SIZE):
            d = RADIX_SIZE - 1 - rev
            cum_d = tl.load(tle_gpu.gpu.local_ptr(smem_counts, (d,)))
            if d + 1 < RADIX_SIZE:
                cum_next = tl.load(tle_gpu.gpu.local_ptr(smem_counts, (d + 1,)))
            else:
                cum_next = 0
            take = (found == 0) & (cum_d >= k_to_find) & (cum_next < k_to_find)
            selected_scalar = tl.where(take, d, selected_scalar)
            counts_gt_scalar = tl.where(take, cum_next, counts_gt_scalar)
            found = tl.where(take, 1, found)
        return selected_scalar, counts_gt_scalar

    @libentry()
    @triton.jit
    def topk_fp8_row_radix_tle_large_kernel(
        X,
        Scale,
        Yv,
        Yi,
        stride_xm,
        stride_sm,
        stride_ym,
        n_cols,
        K: tl.constexpr,
        K_PAD: tl.constexpr,
        BLOCK_N: tl.constexpr,
        PART_N: tl.constexpr,
        RADIX_BITS: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        pid_p = tl.program_id(1)
        col_start = pid_p * PART_N
        part_cols = tl.minimum(n_cols - col_start, PART_N)
        x_utype = tl.uint16
        x_ultype = tl.uint32
        RADIX_SIZE: tl.constexpr = 1 << RADIX_BITS
        RADIX_MASK: tl.constexpr = RADIX_SIZE - 1
        bins = tl.arange(0, RADIX_SIZE)
        one = tl.full([BLOCK_N], 1, tl.int32)
        n_tiles = tl.cdiv(part_cols, BLOCK_N)

        smem_keys = tle_gpu.gpu.alloc(
            [PART_N],
            dtype=x_utype,
            layout=None,
            scope=tle_gpu.gpu.smem,
            nv_mma_shared_layout=False,
        )
        for t in tl.range(0, n_tiles):
            local = t * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_n = local < part_cols
            offs_n = col_start + local
            x = _load_dequant_fp8_row(
                X, Scale, pid, stride_xm, stride_sm, offs_n, mask_n, GROUP_SIZE
            )
            x_key = _fpval_to_key_with_nan(x, x.to(x_utype, bitcast=True))
            tl.store(
                tle_gpu.gpu.local_ptr(smem_keys, (local,)),
                x_key,
                mask=mask_n,
            )

        smem_counts = tle_gpu.gpu.alloc(
            [RADIX_SIZE],
            dtype=tl.int32,
            layout=None,
            scope=tle_gpu.gpu.smem,
            nv_mma_shared_layout=False,
        )
        smem_count_ptrs = tle_gpu.gpu.local_ptr(smem_counts, (bins,))

        desired = tl.full((), 0, dtype=x_utype)
        desired_mask = tl.full((), 0, dtype=x_utype)
        k_to_find = tl.full((), K, dtype=tl.int32)
        for digit_pos in tl.static_range(16 - RADIX_BITS, -1, -RADIX_BITS):
            tl.store(smem_count_ptrs, tl.zeros([RADIX_SIZE], dtype=tl.int32))
            for t in tl.range(0, n_tiles):
                local = t * BLOCK_N + tl.arange(0, BLOCK_N)
                mask_n = local < part_cols
                x_key = tl.load(
                    tle_gpu.gpu.local_ptr(smem_keys, (local,)),
                    mask=mask_n,
                    other=0,
                )
                matches = (x_key & desired_mask) == desired
                digit = ((x_key >> digit_pos) & RADIX_MASK).to(tl.int32)
                valid = mask_n & matches
                count_addrs = tle_gpu.gpu.local_ptr(smem_counts, (digit,))
                tl.atomic_add(count_addrs, one, mask=valid, sem="relaxed", scope="cta")

            counts = tl.load(smem_count_ptrs)
            tl.store(smem_count_ptrs, tl.cumsum(counts, axis=0, reverse=True))
            selected_scalar, counts_gt_scalar = _select_desc_bin(
                smem_counts, k_to_find, RADIX_SIZE
            )
            selected_u = selected_scalar.to(x_utype)
            desired = desired | (selected_u << digit_pos)
            desired_mask = desired_mask | (
                tl.full((), RADIX_MASK, dtype=x_utype) << digit_pos
            )
            k_to_find = k_to_find - counts_gt_scalar

        thr_key = desired
        min_val = tl.full((), float("-inf"), tl.bfloat16)
        min_key = _fpval_to_key_with_nan(min_val, min_val.to(x_utype, bitcast=True))
        min_packed = min_key.to(x_ultype) << 16
        offs_k = tl.arange(0, K_PAD)

        smem_selected = tle_gpu.gpu.alloc(
            [K_PAD],
            dtype=x_ultype,
            layout=None,
            scope=tle_gpu.gpu.smem,
            nv_mma_shared_layout=False,
        )
        smem_selected_ptrs = tle_gpu.gpu.local_ptr(smem_selected, (offs_k,))
        tl.store(smem_selected_ptrs, tl.full([K_PAD], min_packed, dtype=x_ultype))

        smem_write_count = tle_gpu.gpu.alloc(
            [2],
            dtype=tl.int32,
            layout=None,
            scope=tle_gpu.gpu.smem,
            nv_mma_shared_layout=False,
        )
        tl.store(tle_gpu.gpu.local_ptr(smem_write_count, (0,)), 0)
        tl.store(
            tle_gpu.gpu.local_ptr(smem_write_count, (1,)),
            K - k_to_find,
        )
        counter_offsets = tl.zeros([BLOCK_N], dtype=tl.int32)
        gt_count_ptrs = tle_gpu.gpu.local_ptr(smem_write_count, (counter_offsets,))
        eq_count_ptrs = tle_gpu.gpu.local_ptr(smem_write_count, (counter_offsets + 1,))

        for t in tl.range(0, n_tiles):
            local = t * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_n = local < part_cols
            offs_n = col_start + local
            x_key = tl.load(
                tle_gpu.gpu.local_ptr(smem_keys, (local,)),
                mask=mask_n,
                other=0,
            )
            packed = (x_key.to(x_ultype) << 16) | (n_cols - offs_n).to(x_ultype)
            take_gt = mask_n & (x_key > thr_key)
            take_eq = mask_n & (x_key == thr_key)
            pos_gt = tl.atomic_add(
                gt_count_ptrs, one, mask=take_gt, sem="relaxed", scope="cta"
            )
            pos_eq = tl.atomic_add(
                eq_count_ptrs, one, mask=take_eq, sem="relaxed", scope="cta"
            )
            pos = tl.where(take_gt, pos_gt, pos_eq)
            tl.store(
                tle_gpu.gpu.local_ptr(smem_selected, (pos.to(tl.int32),)),
                packed,
                mask=(take_gt | take_eq) & (pos < K_PAD),
            )

        selected_packed = tl.load(smem_selected_ptrs)
        topk = tl.sort(selected_packed, dim=0, descending=True)
        idx_mask = tl.full(topk.shape, (1 << 16) - 1, dtype=topk.dtype)
        idx_raw = (topk & idx_mask).to(tl.uint32)
        y_indices = (n_cols - idx_raw.to(tl.int32)).to(tl.int64)
        y_keys = (topk >> 16).to(x_utype)
        y_values = _key_to_fpval(y_keys).to(tl.bfloat16, bitcast=True)
        mask_k = offs_k < K
        out_base = pid * stride_ym + pid_p * K
        tl.store(Yv + out_base + offs_k, y_values, mask=mask_k)
        tl.store(Yi + out_base + offs_k, y_indices, mask=mask_k)

    @libentry()
    @triton.jit
    def topk_merge_cand_kernel(
        Yv,
        Yi,
        Cv,
        Ci,
        n_cand,
        K: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < n_cand
        vals = tl.load(Cv + pid * n_cand + offs, mask=mask, other=float("-inf")).to(
            tl.bfloat16
        )
        idx = tl.load(Ci + pid * n_cand + offs, mask=mask, other=0)
        keys = _fpval_to_key_with_nan(vals, vals.to(tl.uint16, bitcast=True))
        packed = (keys.to(tl.uint32) << 16) | (65535 - idx).to(tl.uint32)
        packed = tl.where(mask, packed, 0)
        topk = tl.sort(packed, dim=0, descending=True)
        y_keys = (topk >> 16).to(tl.uint16)
        y_values = _key_to_fpval(y_keys).to(tl.bfloat16, bitcast=True)
        y_indices = (65535 - (topk & 0xFFFF)).to(tl.int64)
        tl.store(Yv + pid * K + offs, y_values, mask=offs < K)
        tl.store(Yi + pid * K + offs, y_indices, mask=offs < K)


def topk_w8a16_fp8(
    x_fp8,
    x_scale,
    k,
    dim=-1,
    largest=True,
    sorted=True,
    group_size=128,
    out_dtype=torch.bfloat16,
):
    logger.debug("GEMS TOPK FP8E5 W8A16")
    if dim < 0:
        dim = dim + x_fp8.ndim

    assert dim == x_fp8.ndim - 1, "Currently only support topk in last dimension"
    assert sorted, "Currently only support sorted == True"
    if x_fp8.dtype != torch.float8_e5m2:
        if x_fp8.itemsize != 1:
            raise TypeError(
                f"topk_w8a16_fp8 expects float8_e5m2 (or 8-bit) storage, got {x_fp8.dtype}"
            )
        x_fp8 = x_fp8.view(torch.float8_e5m2)

    if k == 0:
        out_shape = list(x_fp8.shape[:-1]) + [0]
        return (
            torch.empty(out_shape, device=x_fp8.device, dtype=out_dtype),
            torch.empty(out_shape, device=x_fp8.device, dtype=torch.int64),
        )

    topk_elem_cnt = x_fp8.shape[dim]
    batch_size = math.prod(x_fp8.shape) // topk_elem_cnt
    num_groups = triton.cdiv(topk_elem_cnt, group_size)
    expected_scale_shape = x_fp8.shape[:-1] + (num_groups,)
    assert (
        x_scale.shape == expected_scale_shape
    ), f"x_scale shape should be {expected_scale_shape}, got {x_scale.shape}"

    x_2d = x_fp8.view(torch.uint8).reshape(batch_size, topk_elem_cnt)
    scale_2d = x_scale.reshape(batch_size, num_groups)
    descending = True if largest else False

    out_shape = x_fp8.shape[:-1] + (k,)
    y_vals = torch.empty(out_shape, device=x_fp8.device, dtype=out_dtype)
    y_idx = torch.empty(out_shape, device=x_fp8.device, dtype=torch.int64)
    y_vals_2d = y_vals.reshape(batch_size, k)
    y_idx_2d = y_idx.reshape(batch_size, k)

    # A single non-negative quantization scale preserves the raw FP8 order.
    # Sort packed 8-bit keys and only dequantize the selected values.
    if descending and sorted and num_groups == 1 and topk_elem_cnt <= 128:
        block_size = triton.next_power_of_2(topk_elem_cnt)
        with torch_device_fn.device(x_fp8.device):
            topk_fp8_one_group_packed_kernel[(batch_size,)](
                y_vals_2d,
                y_idx_2d,
                x_2d,
                scale_2d,
                k,
                topk_elem_cnt,
                block_size,
                num_warps=8,
                num_stages=1,
            )
        return (y_vals, y_idx)

    # For two 128-element groups, keep only each group's K best raw FP8
    # candidates, dequantize 2K values, then merge those candidates in BF16.
    if (
        descending
        and sorted
        and out_dtype == torch.bfloat16
        and group_size == 128
        and topk_elem_cnt == 2 * group_size
        and k == triton.next_power_of_2(k)
    ):
        with torch_device_fn.device(x_fp8.device):
            topk_fp8_two_group_packed_kernel[(batch_size,)](
                y_vals_2d,
                y_idx_2d,
                x_2d,
                scale_2d,
                k,
                group_size,
                num_warps=2,
                num_stages=1,
            )
        return (y_vals, y_idx)

    if (
        HAS_TLE_GPU
        and descending
        and sorted
        and x_fp8.device.type == flag_gems.device
        and k >= 8
        and topk_elem_cnt >= 128
        and topk_elem_cnt <= 65535
        and triton.next_power_of_2(k) <= 1024
    ):
        k_pad = triton.next_power_of_2(k)
        if topk_elem_cnt >= 32768:
            part_n = 4096
            n_parts = triton.cdiv(topk_elem_cnt, part_n)
            cand_v = torch.empty(
                (batch_size, n_parts * k),
                device=x_fp8.device,
                dtype=out_dtype,
            )
            cand_i = torch.empty(
                (batch_size, n_parts * k),
                device=x_fp8.device,
                dtype=torch.int64,
            )
            with torch_device_fn.device(x_fp8.device):
                topk_fp8_row_radix_tle_large_kernel[(batch_size, n_parts)](
                    x_2d,
                    scale_2d,
                    cand_v,
                    cand_i,
                    x_2d.stride(0),
                    scale_2d.stride(0),
                    cand_v.stride(0),
                    topk_elem_cnt,
                    K=k,
                    K_PAD=k_pad,
                    BLOCK_N=512,
                    PART_N=part_n,
                    # On PPU, four 16-bin histogram passes beat two
                    # 256-bin passes by reducing shared-atomic overhead.
                    RADIX_BITS=4,
                    GROUP_SIZE=group_size,
                    num_warps=8,
                    num_stages=1,
                )
                topk_merge_cand_kernel[(batch_size,)](
                    y_vals_2d,
                    y_idx_2d,
                    cand_v,
                    cand_i,
                    n_parts * k,
                    K=k,
                    BLOCK=triton.next_power_of_2(n_parts * k),
                    num_warps=8,
                    num_stages=1,
                )
            return (y_vals, y_idx)

        if topk_elem_cnt >= 4096:
            key_2d = torch.empty(
                (batch_size, topk_elem_cnt),
                device=x_fp8.device,
                dtype=torch.uint16,
            )
            with torch_device_fn.device(x_fp8.device):
                topk_fp8_row_radix_tle_kernel[(batch_size,)](
                    x_2d,
                    scale_2d,
                    key_2d,
                    y_vals_2d,
                    y_idx_2d,
                    x_2d.stride(0),
                    scale_2d.stride(0),
                    key_2d.stride(0),
                    y_vals_2d.stride(0),
                    topk_elem_cnt,
                    K=k,
                    K_PAD=k_pad,
                    BLOCK_N=1024,
                    RADIX_BITS=8,
                    GROUP_SIZE=group_size,
                    HAS_KEY_BUF=True,
                    num_warps=8,
                    num_stages=1,
                )
            return (y_vals, y_idx)

        block_n_radix = max(k_pad, min(256, triton.next_power_of_2(topk_elem_cnt)))
        with torch_device_fn.device(x_fp8.device):
            topk_fp8_row_radix_tle_kernel[(batch_size,)](
                x_2d,
                scale_2d,
                x_2d,
                y_vals_2d,
                y_idx_2d,
                x_2d.stride(0),
                scale_2d.stride(0),
                0,
                y_vals_2d.stride(0),
                topk_elem_cnt,
                K=k,
                K_PAD=k_pad,
                BLOCK_N=block_n_radix,
                RADIX_BITS=4,
                GROUP_SIZE=group_size,
                HAS_KEY_BUF=False,
                num_warps=8,
                num_stages=1,
            )
        return (y_vals, y_idx)

    if topk_elem_cnt <= 512:
        block_size = triton.next_power_of_2(topk_elem_cnt)
        with torch_device_fn.device(x_fp8.device):
            topk_fp8_single_stage_kernel[(batch_size,)](
                y_vals_2d,
                y_idx_2d,
                x_2d,
                scale_2d,
                k,
                topk_elem_cnt,
                block_size,
                descending,
                group_size,
                num_groups,
            )
        return (y_vals, y_idx)

    k_pad = triton.next_power_of_2(max(k, 1))
    block = min(max(k_pad, 128), 256)
    with torch_device_fn.device(x_fp8.device):
        topk_fp8_running_merge_kernel[(batch_size,)](
            y_vals_2d,
            y_idx_2d,
            x_2d,
            scale_2d,
            topk_elem_cnt,
            k,
            block,
            descending,
            group_size,
            num_warps=8,
        )
    return (y_vals, y_idx)

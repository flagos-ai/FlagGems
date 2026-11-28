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

"""Triton top_k_per_row_prefill for DeepSeek V4 prefill-phase topk selection.

Implement based on file python/tutorials/tle/deepseek_v32/01-topk_selector.py from repo
https://github.com/flagos-ai/FlagTree.git, align with vLLM implementation.

"""

import logging
import sys

import torch
import triton
import triton.language as tl

from flag_gems.utils.triton_version_utils import has_triton_tle

if has_triton_tle(3, 6, 0):
    try:
        import triton.experimental.tle.language as tle

        HAS_TLE = True
    except ImportError:
        tle = None
        HAS_TLE = False
else:
    tle = None
    HAS_TLE = False


logger = logging.getLogger(__name__)

# Start of shared implementation code for top_k_per_row_decode and top_k_per_row_prefill

SORTING_ALGORITHM_THRESHOLD = 12288
SPLIT_WORK_THRESHOLD = 200 * 1000
NUM_THREADS_PER_BLOCK = 512
MULTIPLE_BLOCKS_PER_ROW_CONFIG = 10
NUM_THREADS_PER_BLOCK_MERGE = 1024
NUM_FILNAL_ITEMS = 2048
NUM_BINS = 2048
RADIX_BITS_FINAL = 8
RADIX_SIZE_FINAL = 1 << RADIX_BITS_FINAL
RADIX_FINAL_PREFILL_VOCAB_THRESHOLD = 65536


def _use_radix_final_for_prefill(vocab_size):
    return vocab_size >= RADIX_FINAL_PREFILL_VOCAB_THRESHOLD


@triton.jit
def _convert_to_uint32(x):
    bits = x.to(tl.uint32, bitcast=True)
    sign_mask = tl.full(bits.shape, 0x80000000, tl.uint32)
    sign_set = (bits & sign_mask) != 0
    inv = (~bits) & tl.full(bits.shape, 0x7FFFFFFF, tl.uint32)
    return tl.where(sign_set, bits, inv)


@triton.jit
def _extract_bin_idx(x, in_range, pattern, STEP: tl.constexpr):
    is_partial_match = in_range
    if STEP == 0:
        h = x.to(tl.float16)
        bits = h.to(tl.uint16, bitcast=True)
        sign_mask = tl.full(bits.shape, 0x8000, tl.uint16)
        sign_set = (bits & sign_mask) != 0
        inv = (~bits) & tl.full(bits.shape, 0x7FFF, tl.uint16)
        mapped = tl.where(sign_set, bits, inv)
        bin_idx = (mapped >> 5).to(tl.uint32)
    else:
        bits = _convert_to_uint32(x)
        if STEP == 1:
            bin_idx = bits >> 21
        elif STEP == 2:
            bin_idx = (bits >> 10) & 0x7FF
            is_partial_match &= ((bits ^ pattern) >> 21) == 0
        elif STEP == 3:
            bin_idx = bits & 0x3FF
            is_partial_match &= ((bits ^ pattern) >> 10) == 0
    return bin_idx, is_partial_match


@triton.jit
def _convert_to_trt_uint16_hi11(x):
    h = x.to(tl.float16)
    bits = h.to(tl.uint16, bitcast=True)
    sign_mask = tl.full(bits.shape, 0x8000, tl.uint16)
    sign_set = (bits & sign_mask) != 0
    inv = (~bits) & tl.full(bits.shape, 0x7FFF, tl.uint16)
    mapped = tl.where(sign_set, bits, inv)
    return (mapped >> 5).to(tl.int32)


@triton.jit
def _distribute_to_bins(
    logits,
    in_range,
    ones,
    logit_pattern,
    s_histogram_ptr,
    STEP: tl.constexpr,
):
    bin_idx, is_partial_match = _extract_bin_idx(
        logits,
        in_range,
        logit_pattern,
        STEP=STEP,
    )
    tl.atomic_add(
        s_histogram_ptr + bin_idx,
        ones,
        mask=is_partial_match,
        sem="relaxed",
        scope="cta",
    )


@triton.jit
def _process_bins(
    logits,
    in_range,
    ones,
    offs,  # row_start based
    found_topk_values_ptrs,
    final_cnt_ptrs,
    logit_pattern,
    threshold_bin_idx,
    write_directly,
    use_final,
    row_start,
    indices_ptr,
    s_histogram_ptr,
    s_final_logits_ptr,
    s_out_indices_ptr,
    s_out_logits_ptr,
    STEP: tl.constexpr,
    TOPK: tl.constexpr,
    MULTIPLE_BLOCKS_PER_ROW: tl.constexpr,
    MERGE_BLOCKS: tl.constexpr,
):
    NUM_FINAL_ITEMS: tl.constexpr = 2048

    bin_idx, is_partial_match = _extract_bin_idx(
        logits,
        in_range,
        logit_pattern,
        STEP=STEP,
    )
    take_lt = is_partial_match & (bin_idx < threshold_bin_idx) & write_directly
    out_pos_lt = tl.atomic_add(
        found_topk_values_ptrs,
        ones,
        mask=take_lt,
        sem="relaxed",
        scope="cta",
    )
    if MERGE_BLOCKS:
        indices = tl.load(
            indices_ptr + offs,
            mask=take_lt,
        )
        tl.store(
            s_out_indices_ptr + out_pos_lt,
            indices,
            mask=take_lt,
        )
    elif MULTIPLE_BLOCKS_PER_ROW:
        tl.store(
            s_out_indices_ptr + out_pos_lt,
            (offs + row_start).to(tl.int32),
            mask=take_lt,
        )
        tl.store(
            s_out_logits_ptr + out_pos_lt,
            logits,
            mask=take_lt,
        )
    else:
        tl.store(
            s_out_indices_ptr + out_pos_lt,
            offs.to(tl.int32),
            mask=take_lt,
        )

    if STEP < 3:
        if use_final:
            take_eq_final = is_partial_match & (bin_idx == threshold_bin_idx)
            final_pos = tl.atomic_add(
                final_cnt_ptrs,
                ones,
                mask=take_eq_final,
                sem="relaxed",
                scope="cta",
            )
            tl.store(
                s_final_logits_ptr + final_pos,
                logits,
                mask=take_eq_final & (final_pos < NUM_FINAL_ITEMS),
            )
            # s_histogram_ptr being used for indices in final sort
            if MERGE_BLOCKS:
                indices = tl.load(
                    indices_ptr + offs,
                    mask=take_eq_final & (final_pos < NUM_FINAL_ITEMS),
                )
                tl.store(
                    s_histogram_ptr + final_pos,
                    indices,
                    mask=take_eq_final & (final_pos < NUM_FINAL_ITEMS),
                )
            elif MULTIPLE_BLOCKS_PER_ROW:
                tl.store(
                    s_histogram_ptr + final_pos,
                    (offs + row_start).to(tl.int32),
                    mask=take_eq_final & (final_pos < NUM_FINAL_ITEMS),
                )
            else:
                tl.store(
                    s_histogram_ptr + final_pos,
                    offs.to(tl.int32),
                    mask=take_eq_final & (final_pos < NUM_FINAL_ITEMS),
                )
    else:
        take_eq = is_partial_match & (bin_idx == threshold_bin_idx)
        # s_histogram_ptr being used for exclude prefix sum
        out_pos_eq = tl.atomic_add(
            s_histogram_ptr + bin_idx,
            ones,
            mask=take_eq,
            sem="relaxed",
            scope="cta",
        )
        if MERGE_BLOCKS:
            indices = tl.load(
                indices_ptr + offs,
                mask=take_eq & (out_pos_eq < TOPK),
            )
            tl.store(
                s_out_indices_ptr + out_pos_eq,
                indices,
                mask=take_eq & (out_pos_eq < TOPK),
            )
        elif MULTIPLE_BLOCKS_PER_ROW:
            tl.store(
                s_out_indices_ptr + out_pos_eq,
                (offs + row_start).to(tl.int32),
                mask=take_eq & (out_pos_eq < TOPK),
            )
            tl.store(
                s_out_logits_ptr + out_pos_eq,
                logits,
                mask=take_eq & (out_pos_eq < TOPK),
            )
        else:
            tl.store(
                s_out_indices_ptr + out_pos_eq,
                offs.to(tl.int32),
                mask=take_eq & (out_pos_eq < TOPK),
            )


@triton.jit
def _process_histogram_step(
    logits_ptr,
    row_start,
    row_end,
    stride1,
    vocab_size,
    skip_elems,
    indices_ptr,
    logit_pattern,
    threshold_bin_idx,
    assume_aligned,
    s_histogram_ptr,
    s_final_logits_ptr,
    s_final_cnt_ptr,
    s_threshold_bin_idx_ptr,
    s_final_bin_size_ptr,
    s_found_topk_values_ptr,
    s_out_indices_ptr,
    s_out_logits_ptr,
    STEP: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_TLE: tl.constexpr,
    MULTIPLE_BLOCKS_PER_ROW: tl.constexpr,
    MERGE_BLOCKS: tl.constexpr,
):
    VEC: tl.constexpr = 4
    NUM_FINAL_ITEMS: tl.constexpr = 2048
    RADIX11_SIZE: tl.constexpr = 2048
    RADIX11_MASK: tl.constexpr = 0x7FF
    RADIX10_SIZE: tl.constexpr = 1024

    lane = tl.arange(0, BLOCK_SIZE)
    vec = tl.arange(0, VEC)
    ones = tl.full([BLOCK_SIZE], 1, tl.int32)
    ones_vec_2d = tl.full([BLOCK_SIZE, VEC], 1, tl.int32)
    zeros = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    zeros_vec_2d = tl.zeros([BLOCK_SIZE, VEC], dtype=tl.int32)

    threshold_rounds: tl.constexpr = (
        RADIX10_SIZE // BLOCK_SIZE if STEP == 3 else RADIX11_SIZE // BLOCK_SIZE
    )
    for clear_round in tl.static_range(0, threshold_rounds):
        clear_bins = clear_round * BLOCK_SIZE + lane
        tl.store(s_histogram_ptr + clear_bins, 0)
    tl.debug_barrier()

    if STEP == 2:
        logit_pattern = (threshold_bin_idx.to(tl.uint32) & RADIX11_MASK) << 21
    elif STEP == 3:
        logit_pattern |= (threshold_bin_idx.to(tl.uint32) & RADIX11_MASK) << 10

    if assume_aligned:
        n_tiles = tl.cdiv(vocab_size, BLOCK_SIZE)
        n_vec_full = vocab_size // (BLOCK_SIZE * VEC)
        rem_tiles = (vocab_size - n_vec_full * BLOCK_SIZE * VEC) // BLOCK_SIZE
        for t in tl.range(0, n_vec_full):
            base = t * BLOCK_SIZE * VEC + lane * VEC
            offs = base[:, None] + vec[None, :]
            x_vec = tl.load(logits_ptr + offs)
            _distribute_to_bins(
                x_vec,
                True,
                ones_vec_2d,
                logit_pattern,
                s_histogram_ptr,
                STEP=STEP,
            )
        for t in tl.range(0, rem_tiles):
            offs = (n_vec_full * VEC + t) * BLOCK_SIZE + lane
            x = tl.load(logits_ptr + offs)
            _distribute_to_bins(
                x,
                True,
                ones,
                logit_pattern,
                s_histogram_ptr,
                STEP=STEP,
            )
    elif stride1 == 1:
        aligned_row_ptr = tl.multiple_of(logits_ptr + row_start + skip_elems, VEC * 4)
        row_len = row_end - row_start - skip_elems
        n_vec_full = row_len // (BLOCK_SIZE * VEC)
        rem_tiles = (row_len - n_vec_full * BLOCK_SIZE * VEC) // BLOCK_SIZE
        rem_elems = row_len % BLOCK_SIZE
        for t in tl.range(0, n_vec_full):
            base = t * BLOCK_SIZE * VEC + lane * VEC
            offs = base[:, None] + vec[None, :]
            x_vec = tl.load(aligned_row_ptr + offs)
            _distribute_to_bins(
                x_vec,
                True,
                ones_vec_2d,
                logit_pattern,
                s_histogram_ptr,
                STEP=STEP,
            )
        for t in tl.range(0, rem_tiles):
            offs = (n_vec_full * VEC + t) * BLOCK_SIZE + lane
            x = tl.load(aligned_row_ptr + offs)
            _distribute_to_bins(
                x,
                True,
                ones,
                logit_pattern,
                s_histogram_ptr,
                STEP=STEP,
            )
        if skip_elems > 0:
            offs = lane
            in_range = lane < skip_elems
            x = tl.load(
                logits_ptr + row_start + offs, mask=in_range, other=float("-inf")
            )
            _distribute_to_bins(
                x,
                in_range,
                ones,
                logit_pattern,
                s_histogram_ptr,
                STEP=STEP,
            )
        if rem_elems > 0:
            offs = (n_vec_full * VEC + rem_tiles) * BLOCK_SIZE + lane
            in_range = lane < rem_elems
            x = tl.load(aligned_row_ptr + offs, mask=in_range, other=float("-inf"))
            _distribute_to_bins(
                x,
                in_range,
                ones,
                logit_pattern,
                s_histogram_ptr,
                STEP=STEP,
            )
    else:
        row_len = row_end - row_start
        n_tiles = tl.cdiv(row_len, BLOCK_SIZE)
        for t in tl.range(0, n_tiles):
            offs = t * BLOCK_SIZE + lane
            in_range = offs < row_len
            x = tl.load(
                logits_ptr + row_start + offs * stride1,
                mask=in_range,
                other=float("-inf"),
            )
            _distribute_to_bins(
                x,
                in_range,
                ones,
                logit_pattern,
                s_histogram_ptr,
                STEP=STEP,
            )
    last_value = tl.load(s_found_topk_values_ptr)
    tl.debug_barrier()

    threshold_bin_ptrs = s_threshold_bin_idx_ptr + zeros
    final_bin_size_ptrs = s_final_bin_size_ptr + zeros
    threshold_found = tl.full((), False, dtype=tl.int1)
    for round_idx in tl.static_range(0, threshold_rounds):
        if not threshold_found:
            bins = round_idx * BLOCK_SIZE + lane
            counts = tl.load(s_histogram_ptr + bins)
            if HAS_TLE:
                prefix_sum, counts_total = tle.cumsum(counts, axis=0, reverse=False)
            else:
                counts_total = tl.sum(counts)
                prefix_sum = counts_total - tl.cumsum(counts, axis=0, reverse=True)
            prefix_sum = prefix_sum + last_value
            total_sum = last_value + counts_total
            next_prefix_sum = prefix_sum + counts
            threshold_mask = (prefix_sum < TOPK) & (next_prefix_sum >= TOPK)
            threshold_bin = bins
            threshold_bin_size = next_prefix_sum - prefix_sum
            if STEP == 3:
                tl.store(s_histogram_ptr + bins, prefix_sum)
            tl.store(threshold_bin_ptrs, threshold_bin, mask=threshold_mask)
            tl.store(final_bin_size_ptrs, threshold_bin_size, mask=threshold_mask)
            found_round = tl.reduce_or(threshold_mask, axis=0)
            threshold_found = found_round
            last_value = total_sum

    tl.debug_barrier()
    threshold_bin_idx = tl.load(s_threshold_bin_idx_ptr)
    final_bin_size = tl.load(s_final_bin_size_ptr)
    use_final = final_bin_size <= NUM_FINAL_ITEMS
    write_directly = ((STEP == 0) & (final_bin_size <= NUM_FINAL_ITEMS)) | (STEP >= 1)

    found_ptrs = s_found_topk_values_ptr + zeros
    final_cnt_ptrs = s_final_cnt_ptr + zeros
    if assume_aligned:
        found_ptrs_vec_2d = s_found_topk_values_ptr + zeros_vec_2d
        final_cnt_ptrs_vec_2d = s_final_cnt_ptr + zeros_vec_2d
        n_tiles = tl.cdiv(vocab_size, BLOCK_SIZE)
        n_vec_full = vocab_size // (BLOCK_SIZE * VEC)
        rem_tiles = (vocab_size - n_vec_full * BLOCK_SIZE * VEC) // BLOCK_SIZE
        for t in tl.range(0, n_vec_full):
            base = t * BLOCK_SIZE * VEC + lane * VEC
            offs = base[:, None] + vec[None, :]
            x_vec = tl.load(logits_ptr + offs)
            _process_bins(
                x_vec,
                True,
                ones_vec_2d,
                offs,
                found_ptrs_vec_2d,
                final_cnt_ptrs_vec_2d,
                logit_pattern,
                threshold_bin_idx,
                write_directly,
                use_final,
                row_start,
                indices_ptr,
                s_histogram_ptr,
                s_final_logits_ptr,
                s_out_indices_ptr,
                s_out_logits_ptr,
                STEP=STEP,
                TOPK=TOPK,
                MULTIPLE_BLOCKS_PER_ROW=MULTIPLE_BLOCKS_PER_ROW,
                MERGE_BLOCKS=MERGE_BLOCKS,
            )
        for t in tl.range(0, rem_tiles):
            offs = (n_vec_full * VEC + t) * BLOCK_SIZE + lane
            x = tl.load(logits_ptr + offs)
            _process_bins(
                x,
                True,
                ones,
                offs,
                found_ptrs,
                final_cnt_ptrs,
                logit_pattern,
                threshold_bin_idx,
                write_directly,
                use_final,
                row_start,
                indices_ptr,
                s_histogram_ptr,
                s_final_logits_ptr,
                s_out_indices_ptr,
                s_out_logits_ptr,
                STEP=STEP,
                TOPK=TOPK,
                MULTIPLE_BLOCKS_PER_ROW=MULTIPLE_BLOCKS_PER_ROW,
                MERGE_BLOCKS=MERGE_BLOCKS,
            )
    elif stride1 == 1:
        found_ptrs_vec_2d = s_found_topk_values_ptr + zeros_vec_2d
        final_cnt_ptrs_vec_2d = s_final_cnt_ptr + zeros_vec_2d
        aligned_row_ptr = tl.multiple_of(logits_ptr + row_start + skip_elems, VEC * 4)
        row_len = row_end - row_start - skip_elems
        n_vec_full = row_len // (BLOCK_SIZE * VEC)
        rem_tiles = (row_len - n_vec_full * BLOCK_SIZE * VEC) // BLOCK_SIZE
        rem_elems = row_len % BLOCK_SIZE
        for t in tl.range(0, n_vec_full):
            base = t * BLOCK_SIZE * VEC + lane * VEC
            offs = base[:, None] + vec[None, :]
            x_vec = tl.load(aligned_row_ptr + offs)
            _process_bins(
                x_vec,
                True,
                ones_vec_2d,
                offs + skip_elems,
                found_ptrs_vec_2d,
                final_cnt_ptrs_vec_2d,
                logit_pattern,
                threshold_bin_idx,
                write_directly,
                use_final,
                row_start,
                indices_ptr,
                s_histogram_ptr,
                s_final_logits_ptr,
                s_out_indices_ptr,
                s_out_logits_ptr,
                STEP=STEP,
                TOPK=TOPK,
                MULTIPLE_BLOCKS_PER_ROW=MULTIPLE_BLOCKS_PER_ROW,
                MERGE_BLOCKS=MERGE_BLOCKS,
            )
        for t in tl.range(0, rem_tiles):
            offs = (n_vec_full * VEC + t) * BLOCK_SIZE + lane
            x = tl.load(aligned_row_ptr + offs)
            _process_bins(
                x,
                True,
                ones,
                offs + skip_elems,
                found_ptrs,
                final_cnt_ptrs,
                logit_pattern,
                threshold_bin_idx,
                write_directly,
                use_final,
                row_start,
                indices_ptr,
                s_histogram_ptr,
                s_final_logits_ptr,
                s_out_indices_ptr,
                s_out_logits_ptr,
                STEP=STEP,
                TOPK=TOPK,
                MULTIPLE_BLOCKS_PER_ROW=MULTIPLE_BLOCKS_PER_ROW,
                MERGE_BLOCKS=MERGE_BLOCKS,
            )
        if skip_elems > 0:
            offs = lane
            in_range = lane < skip_elems
            x = tl.load(
                logits_ptr + row_start + offs, mask=in_range, other=float("-inf")
            )
            _process_bins(
                x,
                in_range,
                ones,
                offs,
                found_ptrs,
                final_cnt_ptrs,
                logit_pattern,
                threshold_bin_idx,
                write_directly,
                use_final,
                row_start,
                indices_ptr,
                s_histogram_ptr,
                s_final_logits_ptr,
                s_out_indices_ptr,
                s_out_logits_ptr,
                STEP=STEP,
                TOPK=TOPK,
                MULTIPLE_BLOCKS_PER_ROW=MULTIPLE_BLOCKS_PER_ROW,
                MERGE_BLOCKS=MERGE_BLOCKS,
            )
        if rem_elems > 0:
            offs = (n_vec_full * VEC + rem_tiles) * BLOCK_SIZE + lane
            in_range = lane < rem_elems
            x = tl.load(aligned_row_ptr + offs, mask=in_range, other=float("-inf"))
            _process_bins(
                x,
                in_range,
                ones,
                offs + skip_elems,
                found_ptrs,
                final_cnt_ptrs,
                logit_pattern,
                threshold_bin_idx,
                write_directly,
                use_final,
                row_start,
                indices_ptr,
                s_histogram_ptr,
                s_final_logits_ptr,
                s_out_indices_ptr,
                s_out_logits_ptr,
                STEP=STEP,
                TOPK=TOPK,
                MULTIPLE_BLOCKS_PER_ROW=MULTIPLE_BLOCKS_PER_ROW,
                MERGE_BLOCKS=MERGE_BLOCKS,
            )
    else:
        row_len = row_end - row_start
        n_tiles = tl.cdiv(row_len, BLOCK_SIZE)
        for t in tl.range(0, n_tiles):
            offs = t * BLOCK_SIZE + lane
            in_range = offs < row_len
            x = tl.load(
                logits_ptr + row_start + offs * stride1,
                mask=in_range,
                other=float("-inf"),
            )
            _process_bins(
                x,
                in_range,
                ones,
                offs,
                found_ptrs,
                final_cnt_ptrs,
                logit_pattern,
                threshold_bin_idx,
                write_directly,
                use_final,
                row_start,
                indices_ptr,
                s_histogram_ptr,
                s_final_logits_ptr,
                s_out_indices_ptr,
                s_out_logits_ptr,
                STEP=STEP,
                TOPK=TOPK,
                MULTIPLE_BLOCKS_PER_ROW=MULTIPLE_BLOCKS_PER_ROW,
                MERGE_BLOCKS=MERGE_BLOCKS,
            )
    tl.debug_barrier()
    return final_bin_size > NUM_FINAL_ITEMS, logit_pattern, threshold_bin_idx


@triton.jit
def _final_select_radix(
    s_histogram_ptr,
    s_final_logits_ptr,
    s_final_cnt_ptr,
    s_found_topk_values_ptr,
    s_out_indices_ptr,
    s_out_logits_ptr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MULTIPLE_BLOCKS_PER_ROW: tl.constexpr,
):
    NUM_FINAL_ITEMS: tl.constexpr = 2048
    RADIX_BITS_FINAL: tl.constexpr = 8
    RADIX_SIZE_FINAL: tl.constexpr = 1 << RADIX_BITS_FINAL
    RADIX_MASK_FINAL: tl.constexpr = RADIX_SIZE_FINAL - 1
    DIGIT_START: tl.constexpr = 32 - RADIX_BITS_FINAL

    lane = tl.arange(0, BLOCK_SIZE)
    ones = tl.full([BLOCK_SIZE], 1, tl.int32)
    zeros = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    bins = tl.arange(0, RADIX_SIZE_FINAL)

    s_radix_counts = tle.gpu.alloc(
        [RADIX_SIZE_FINAL],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_radix_count_ptr = tle.gpu.local_ptr(s_radix_counts, (0,))
    radix_count_vec_ptr = s_radix_count_ptr + bins
    base_idx = tl.load(s_found_topk_values_ptr)
    final_cnt = tl.minimum(tl.load(s_final_cnt_ptr), NUM_FINAL_ITEMS)
    remain = tl.minimum(TOPK - base_idx, final_cnt)
    tl.debug_barrier()

    if remain > 0:
        desired = tl.zeros((), dtype=tl.uint32)
        desired_mask = tl.zeros((), dtype=tl.uint32)
        k_to_find = remain + 1

        for digit_pos in tl.static_range(DIGIT_START, -1, -RADIX_BITS_FINAL):
            if k_to_find > 1:
                tl.store(s_radix_count_ptr + lane, 0, mask=lane < RADIX_SIZE_FINAL)
                tl.debug_barrier()

                cnt_tiles = tl.cdiv(final_cnt, BLOCK_SIZE)
                for t in tl.range(0, cnt_tiles):
                    pos = t * BLOCK_SIZE + lane
                    valid = pos < final_cnt
                    x = tl.load(
                        s_final_logits_ptr + pos,
                        mask=valid,
                        other=0,
                    )
                    key = _convert_to_uint32(x)
                    matches = (key & desired_mask) == desired
                    digit = ((key >> digit_pos) & RADIX_MASK_FINAL).to(tl.int32)
                    take = valid & matches
                    tl.atomic_add(
                        s_radix_count_ptr + digit,
                        ones,
                        mask=take,
                        sem="relaxed",
                        scope="cta",
                    )

                tl.debug_barrier()
                counts = tl.load(radix_count_vec_ptr)
                prefix_sum, _ = tle.cumsum(counts, axis=0, reverse=False)
                next_prefix_sum = prefix_sum + counts
                threshold_mask = (prefix_sum < k_to_find) & (
                    next_prefix_sum >= k_to_find
                )
                threshold_init = tl.full((), RADIX_SIZE_FINAL, dtype=tl.int32)
                threshold_bin = tl.min(
                    tl.where(threshold_mask, bins, threshold_init), axis=0
                ).to(tl.int32)
                threshold_bin = tl.where(
                    threshold_bin == RADIX_SIZE_FINAL,
                    RADIX_SIZE_FINAL - 1,
                    threshold_bin,
                )
                counts_lt = tl.max(
                    tl.where(bins == threshold_bin, prefix_sum, 0), axis=0
                ).to(tl.int32)

                desired = desired | (threshold_bin.to(tl.uint32) << digit_pos)
                desired_mask = desired_mask | (
                    tl.full((), RADIX_MASK_FINAL, dtype=tl.uint32) << digit_pos
                )
                k_to_find = k_to_find - counts_lt

        thr_key = desired
        found_ptrs = s_found_topk_values_ptr + zeros
        cnt_tiles = tl.cdiv(final_cnt, BLOCK_SIZE)
        for t in tl.range(0, cnt_tiles):
            pos = t * BLOCK_SIZE + lane
            valid = pos < final_cnt
            idx = tl.load(s_histogram_ptr + pos, mask=valid, other=0)
            x = tl.load(
                s_final_logits_ptr + pos,
                mask=valid,
                other=0,
            )
            key = _convert_to_uint32(x)
            take_lt = valid & (key < thr_key)
            out_pos_gt = tl.atomic_add(
                found_ptrs,
                ones,
                mask=take_lt,
                sem="relaxed",
                scope="cta",
            )
            tl.store(
                s_out_indices_ptr + out_pos_gt,
                idx,
                mask=take_lt & (out_pos_gt < TOPK),
            )
            if MULTIPLE_BLOCKS_PER_ROW:
                tl.store(
                    s_out_logits_ptr + out_pos_gt,
                    x,
                    mask=take_lt & (out_pos_gt < TOPK),
                )

        tl.debug_barrier()
        cur = tl.load(s_found_topk_values_ptr)
        if cur < TOPK:
            for t in tl.range(0, cnt_tiles):
                cur = tl.load(s_found_topk_values_ptr)
                if cur < TOPK:
                    pos = t * BLOCK_SIZE + lane
                    valid = pos < final_cnt
                    idx = tl.load(s_histogram_ptr + pos, mask=valid, other=0)
                    x = tl.load(
                        s_final_logits_ptr + pos,
                        mask=valid,
                        other=0,
                    )
                    key = _convert_to_uint32(x)
                    take_eq = valid & (key == thr_key)
                    out_pos_eq = tl.atomic_add(
                        found_ptrs,
                        ones,
                        mask=take_eq,
                        sem="relaxed",
                        scope="cta",
                    )
                    tl.store(
                        s_out_indices_ptr + out_pos_eq,
                        idx,
                        mask=take_eq & (out_pos_eq < TOPK),
                    )
                    if MULTIPLE_BLOCKS_PER_ROW:
                        tl.store(
                            s_out_logits_ptr + out_pos_eq,
                            x,
                            mask=take_eq & (out_pos_eq < TOPK),
                        )
        tl.debug_barrier()


@triton.jit
def _top_k_per_row_job(
    logits_ptr,
    out_indices_ptr,
    row_start,
    row_end,
    stride1,
    vocab_size,
    skip_elems,
    out_logits_ptr,
    indices_ptr,
    s_histogram_ptr,
    s_final_logits_ptr,
    s_final_cnt_ptr,
    s_threshold_bin_idx_ptr,
    s_final_bin_size_ptr,
    s_found_topk_values_ptr,
    s_out_indices_ptr,
    s_out_logits_ptr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_RADIX_FINAL: tl.constexpr,
    HAS_TLE: tl.constexpr,
    MULTIPLE_BLOCKS_PER_ROW: tl.constexpr,
    MERGE_BLOCKS: tl.constexpr,
):
    NUM_FINAL_ITEMS: tl.constexpr = 2048

    assume_aligned = (
        (row_start == 0)
        & (row_end == vocab_size)
        & (stride1 == 1)
        & ((vocab_size % BLOCK_SIZE) == 0)
    )
    if assume_aligned:
        tl.assume(row_start == 0)
        tl.assume(row_end == vocab_size)
        tl.assume(stride1 == 1)
        vocab_size = tl.multiple_of(vocab_size, BLOCK_SIZE)
    elif stride1 == 1:
        tl.assume(stride1 == 1)

    lane = tl.arange(0, BLOCK_SIZE)
    row_len = row_end - row_start
    if row_len <= TOPK:
        chunks: tl.constexpr = (TOPK + BLOCK_SIZE - 1) // BLOCK_SIZE
        for chunk_idx in tl.range(0, chunks):
            pos = chunk_idx * BLOCK_SIZE + lane
            take_row = pos < row_len
            if MULTIPLE_BLOCKS_PER_ROW:
                tl.store(
                    out_indices_ptr + pos,
                    (pos + row_start).to(tl.int32),
                    mask=take_row,
                )
                logits = tl.load(logits_ptr + pos + row_start, mask=take_row)
                tl.store(out_logits_ptr + pos, logits, mask=take_row)
            else:
                tl.store(
                    out_indices_ptr + pos,
                    pos.to(tl.int32),
                    mask=take_row,
                )
            take_pad = (pos >= row_len) & (pos < TOPK)
            tl.store(out_indices_ptr + pos, -1, mask=take_pad)
            if MULTIPLE_BLOCKS_PER_ROW:
                tl.store(out_logits_ptr + pos, float("-inf"), mask=take_pad)
        return
    tl.store(s_final_cnt_ptr, 0)
    tl.store(s_found_topk_values_ptr, 0)
    tl.debug_barrier()
    logit_pattern = tl.zeros((), dtype=tl.uint32)
    continue_to_next_step = tl.full((), True, dtype=tl.int1)
    threshold_bin_idx = tl.full((), -1, dtype=tl.int32)
    for step_idx in tl.static_range(0, 4):
        if continue_to_next_step:
            (
                continue_to_next_step,
                logit_pattern,
                threshold_bin_idx,
            ) = _process_histogram_step(
                logits_ptr,
                row_start,
                row_end,
                stride1,
                vocab_size,
                skip_elems,
                indices_ptr,
                logit_pattern,
                threshold_bin_idx,
                assume_aligned,
                s_histogram_ptr,
                s_final_logits_ptr,
                s_final_cnt_ptr,
                s_threshold_bin_idx_ptr,
                s_final_bin_size_ptr,
                s_found_topk_values_ptr,
                s_out_indices_ptr,
                s_out_logits_ptr,
                STEP=step_idx,
                TOPK=TOPK,
                BLOCK_SIZE=BLOCK_SIZE,
                HAS_TLE=HAS_TLE,
                MULTIPLE_BLOCKS_PER_ROW=MULTIPLE_BLOCKS_PER_ROW,
                MERGE_BLOCKS=MERGE_BLOCKS,
            )

    if not continue_to_next_step:
        if USE_RADIX_FINAL and HAS_TLE:
            _final_select_radix(
                s_histogram_ptr,
                s_final_logits_ptr,
                s_final_cnt_ptr,
                s_found_topk_values_ptr,
                s_out_indices_ptr,
                s_out_logits_ptr,
                TOPK=TOPK,
                BLOCK_SIZE=BLOCK_SIZE,
                MULTIPLE_BLOCKS_PER_ROW=MULTIPLE_BLOCKS_PER_ROW,
            )
        else:
            base_idx = tl.load(s_found_topk_values_ptr)
            # Guard against stale/oversized counts to avoid out-of-bounds accesses
            # in the shared-memory final buffers.
            final_cnt = tl.minimum(tl.load(s_final_cnt_ptr), NUM_FINAL_ITEMS)
            sort_chunks = tl.cdiv(final_cnt, BLOCK_SIZE)
            for sort_chunk in tl.range(0, sort_chunks):
                pos = sort_chunk * BLOCK_SIZE + lane
                valid = pos < final_cnt
                logit_i = tl.load(
                    s_final_logits_ptr + pos,
                    mask=valid,
                    other=0,
                )
                out_rank = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
                for j in tl.range(0, final_cnt):
                    logit_j = tl.load(s_final_logits_ptr + j)
                    better = (logit_i < logit_j) | ((logit_i == logit_j) & (pos < j))
                    out_rank = out_rank + (valid & better).to(tl.int32)
                dst_pos = base_idx + out_rank
                take = valid & (dst_pos < TOPK)
                idx_i = tl.load(
                    s_histogram_ptr + pos,
                    mask=take,
                    other=0,
                )
                tl.store(s_out_indices_ptr + dst_pos, idx_i, mask=take)
                if MULTIPLE_BLOCKS_PER_ROW:
                    tl.store(s_out_logits_ptr + dst_pos, logit_i, mask=take)
            tl.debug_barrier()

    # out_indices_ptr is identical to s_out_indices_ptr for non-tle
    if HAS_TLE:
        flush_chunks: tl.constexpr = (TOPK + BLOCK_SIZE - 1) // BLOCK_SIZE
        for flush_chunk in tl.static_range(flush_chunks):
            pos = flush_chunk * BLOCK_SIZE + lane
            mask = pos < TOPK
            out_vals = tl.load(s_out_indices_ptr + pos, mask=mask, other=-1)
            tl.store(out_indices_ptr + pos, out_vals, mask=mask)
            if MULTIPLE_BLOCKS_PER_ROW:
                split_logits = tl.load(
                    s_out_logits_ptr + pos, mask=mask, other=float("-inf")
                )
                tl.store(out_logits_ptr + pos, split_logits, mask=mask)


# End of shared implementation code for top_k_per_row_decode and top_k_per_row_prefill


@triton.jit
def tle_top_k_per_row_prefill(
    logits_ptr,
    out_indices_ptr,
    row_starts,
    row_ends,
    stride0,
    stride1,
    vocab_size,
    TOPK: tl.constexpr,
    TOPKP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_RADIX_FINAL: tl.constexpr,
    ROW_OFFSET: tl.constexpr,
):
    NUM_FILNAL_ITEMS: tl.constexpr = 2048
    NUM_BINS: tl.constexpr = 2048
    VEC: tl.constexpr = 4

    row_id = tl.program_id(0) + ROW_OFFSET
    row_start = tl.load(row_starts + row_id)
    row_end = tl.load(row_ends + row_id)
    logits_ptr += row_id * stride0
    # float4 align
    x_off_mod = (row_id * stride0 + row_start) % VEC
    skip_elems = 0 if x_off_mod == 0 else VEC - x_off_mod
    out_indices_ptr += row_id * TOPK

    # used for histogram, indices in final sort and exclude prefix_sum
    s_histogram = tle.gpu.alloc(
        [NUM_BINS],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_final_logits = tle.gpu.alloc(
        [NUM_FILNAL_ITEMS],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_out_indices = tle.gpu.alloc(
        [TOPKP],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_final_cnt = tle.gpu.alloc(
        [1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_threshold_bin_idx = tle.gpu.alloc(
        [1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_final_bin_size = tle.gpu.alloc(
        [1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_found_topk_values = tle.gpu.alloc(
        [1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_histogram_ptr = tle.gpu.local_ptr(s_histogram, (0,))
    s_final_logits_ptr = tle.gpu.local_ptr(s_final_logits, (0,))
    s_out_indices_ptr = tle.gpu.local_ptr(s_out_indices, (0,))
    s_final_cnt_ptr = tle.gpu.local_ptr(s_final_cnt, (0,))
    s_threshold_bin_idx_ptr = tle.gpu.local_ptr(s_threshold_bin_idx, (0,))
    s_final_bin_size_ptr = tle.gpu.local_ptr(s_final_bin_size, (0,))
    s_found_topk_values_ptr = tle.gpu.local_ptr(s_found_topk_values, (0,))

    _top_k_per_row_job(
        logits_ptr,
        out_indices_ptr,
        row_start,
        row_end,
        stride1,
        vocab_size,
        skip_elems,
        None,
        None,
        s_histogram_ptr,
        s_final_logits_ptr,
        s_final_cnt_ptr,
        s_threshold_bin_idx_ptr,
        s_final_bin_size_ptr,
        s_found_topk_values_ptr,
        s_out_indices_ptr,
        None,
        TOPK=TOPK,
        BLOCK_SIZE=BLOCK_SIZE,
        USE_RADIX_FINAL=USE_RADIX_FINAL,
        HAS_TLE=True,
        MULTIPLE_BLOCKS_PER_ROW=False,
        MERGE_BLOCKS=False,
    )


# Tile width of the monotone-key bisection-phase kernel (number of columns covered by one scan).
BISECT_BLOCK_SIZE = 2048


@triton.jit
def _to_order_key(x):
    """float32 -> monotone uint32 order key: larger values yield larger keys (including
    negatives and +/-0).

    key = bits ^ 0x80000000 (non-negative) / ~bits (negative), so integer comparison is
    equivalent to floating-point comparison and the subsequent bisection only needs an
    integer count(key >= mid).
    """
    bits = x.to(tl.uint32, bitcast=True)
    negative = (bits >> 31) == 1
    return tl.where(negative, ~bits, bits | 0x80000000)


@triton.jit
def _bisect_row_into(
    row_ptr,
    out_ptr,
    row_len,
    stride1,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply "32 rounds of counting bisection to find the topk-th largest key + two passes
    to write the output" to a single row.

    Factored into a device function so the candidate-set kernel can fall back to this
    definitely-correct slow path in place when the candidate count is abnormal (too few
    collected or overflow), without an extra kernel launch or host-side synchronization.

    Cost: 34 scans per row, taken only when that row's candidate set is unavailable.
    """
    lane = tl.arange(0, BLOCK_SIZE)
    k = tl.minimum(row_len, TOPK)

    # First fill the tail of the output with -1 (placeholder after valid entries when row_len < topk)
    for fill_start in tl.range(0, TOPK, BLOCK_SIZE):
        fill_idx = fill_start + lane
        tl.store(
            out_ptr + fill_idx,
            tl.full([BLOCK_SIZE], -1, tl.int32),
            mask=fill_idx < TOPK,
        )

    # 32 rounds of counting bisection: lo ends up as the k-th largest key.
    # Use a dynamic range instead of static_range: this function is inlined as a fallback branch in the candidate-set
    # kernel,
    # and unrolling all 32 rounds would blow up both kernels' code size and register pressure (measured 3x slower).
    lo = tl.zeros((), dtype=tl.uint32)
    hi = tl.full((), 0xFFFFFFFF, dtype=tl.uint32)
    for bisect_round in tl.range(0, 32):
        span = hi - lo
        mid = lo + (span >> 1) + (span & 1)
        count = tl.zeros((), dtype=tl.int32)
        for off in tl.range(0, row_len, BLOCK_SIZE):
            idx = off + lane
            in_range = idx < row_len
            x = tl.load(row_ptr + idx * stride1, mask=in_range, other=float("-inf"))
            key = _to_order_key(x)
            count += tl.sum((key >= mid).to(tl.int32))
        upper_half = count >= k
        lo = tl.where(upper_half, mid, lo)
        hi = tl.where(upper_half, hi, mid - 1)

    thr = lo
    written = tl.zeros((), dtype=tl.int32)

    # First pass: write out all elements strictly greater than the threshold (their count must be < topk)
    for off in tl.range(0, row_len, BLOCK_SIZE):
        idx = off + lane
        in_range = idx < row_len
        x = tl.load(row_ptr + idx * stride1, mask=in_range, other=float("-inf"))
        take = in_range & (_to_order_key(x) > thr)
        take_i32 = take.to(tl.int32)
        exclusive = tl.cumsum(take_i32, axis=0) - take_i32
        tl.store(out_ptr + written + exclusive, idx, mask=take)
        written += tl.sum(take_i32)

    # Second pass: fill remaining slots in scan order with elements equal to the threshold (any of the tied elements may
    # be chosen)
    remaining = k - written
    filled = tl.zeros((), dtype=tl.int32)
    for off in tl.range(0, row_len, BLOCK_SIZE):
        idx = off + lane
        in_range = idx < row_len
        x = tl.load(row_ptr + idx * stride1, mask=in_range, other=float("-inf"))
        take = in_range & (_to_order_key(x) == thr)
        take_i32 = take.to(tl.int32)
        exclusive = tl.cumsum(take_i32, axis=0) - take_i32
        slot = filled + exclusive
        tl.store(out_ptr + written + slot, idx, mask=take & (slot < remaining))
        filled += tl.sum(take_i32)


# Upper vocab limit for keeping a whole row resident in registers (above this, use the candidate-set approach).
ROW_TILE_MAX = 4096


@triton.jit
def _gcu_rowwise_top_k_per_row(
    logits_ptr,
    out_indices_ptr,
    row_starts,
    row_ends,
    stride0,
    stride1,
    TOPK: tl.constexpr,
    TILE: tl.constexpr,
    PAD_TILE: tl.constexpr,
):
    """Each program handles one row: the whole row of logits is loaded into registers at
    once, and the 32 bisection rounds and output placement all happen in registers, with
    only 1 read + 1 write (of output indices).

    Much faster than ``_bisect_row_into``, which rescans the whole row on every round
    (34 passes).

    GCU measurements that determined this implementation:
      * ``tl.histogram`` / global ``tl.atomic_add`` reach only 1.4 / 1.7 GB/s, so
        histogram-based algorithms are impractical on this device;
      * each ``tl.sum`` reduction costs about 50ns, and unrolled ``static_range`` blows up
        register pressure (58ms for 32 rounds vs 25ms with dynamic ``range``), so
        ``tl.range`` is used;
      * cross-warp reduction synchronization dominates, so the minimum number of warps is
        chosen with "no more than 128 lanes per thread" (1 warp degrades to intra-warp
        shuffle and is measured 2x faster than 8 warps).
    """
    row_id = tl.program_id(0)
    row_start = tl.load(row_starts + row_id)
    row_len = tl.load(row_ends + row_id) - row_start
    row_ptr = logits_ptr + row_id * stride0 + row_start * stride1
    out_ptr = out_indices_ptr + row_id * TOPK

    lane = tl.arange(0, TILE)
    in_range = lane < row_len
    x = tl.load(row_ptr + lane * stride1, mask=in_range, other=float("-inf"))
    key = _to_order_key(x)
    k = tl.minimum(row_len, TOPK)

    # 32 rounds of counting bisection: lo converges to the k-th largest key (lanes outside in_range do not participate
    # in counting)
    lo = tl.zeros((), dtype=tl.uint32)
    hi = tl.full((), 0xFFFFFFFF, dtype=tl.uint32)
    for bisect_round in tl.range(0, 32):
        span = hi - lo
        mid = lo + (span >> 1) + (span & 1)
        count = tl.sum((in_range & (key >= mid)).to(tl.int32))
        upper_half = count >= k
        lo = tl.where(upper_half, mid, lo)
        hi = tl.where(upper_half, hi, mid - 1)
    thr = lo

    # Fill slots after valid entries with -1 (tail placeholder when row_len < TOPK)
    pad = tl.arange(0, PAD_TILE)
    tl.store(out_ptr + pad, tl.full([PAD_TILE], -1, tl.int32), mask=pad >= k)

    # First pass: elements with key > thr are placed directly in scan order (their count must be < k)
    take_hi = in_range & (key > thr)
    take_hi_i32 = take_hi.to(tl.int32)
    rank_hi = tl.cumsum(take_hi_i32, axis=0) - take_hi_i32
    tl.store(out_ptr + rank_hi, lane, mask=take_hi)
    written = tl.sum(take_hi_i32)

    # Second pass: elements with key == thr fill remaining slots in scan order (any of the tied elements may be chosen)
    take_eq = in_range & (key == thr)
    take_eq_i32 = take_eq.to(tl.int32)
    rank_eq = tl.cumsum(take_eq_i32, axis=0) - take_eq_i32
    slot = written + rank_eq
    tl.store(out_ptr + slot, lane, mask=take_eq & (slot < k))


# ---------------------------------------------------------------------------
# Parameters of the candidate-set approach
#
# Pure bisection needs one reduction per lane per round, while on GCU reduction cost is nearly independent of the lane
# count and dominated by
# synchronization depth, so for a large vocab (a row does not fit in registers) first compress the candidates down to
# [CAP]:
#   1) Estimate a lower-bound threshold thr_lo by block sampling (16 bisection rounds on the sample, costing about 1/16
# of a full row);
#   2) Do one full scan to collect elements with key >= thr_lo into the [CAP] candidate buffer;
#   3) If the candidate count falls in [top_k, CAP], run 32 rounds of exact bisection on the candidate set and place the
# output,
#      otherwise fall back in place to _bisect_row_into with 34 scan passes (a single row has only vocab elements,
#      so the absolute cost of a single-row fallback is small).
# ---------------------------------------------------------------------------
CANDIDATE_SAMPLE_SMALL = 256  # Number of sampled elements for a small vocab
# Number of sampled elements for a large vocab (small top_k/vocab requires more samples to reduce estimator variance)
CANDIDATE_SAMPLE_LARGE = 2048
CANDIDATE_SAMPLE_CHUNKS = 8  # Number of sampling blocks (each block is 128B contiguous, exactly one cache line)
# Number of bisection rounds on the sample: only a coarse threshold is needed, no need to resolve down to the lowest bit
CANDIDATE_SAMPLE_ROUNDS = 16


@triton.jit
def _gcu_candidate_top_k_per_row(
    logits_ptr,
    out_indices_ptr,
    cand_key_ptr,
    cand_idx_ptr,
    row_starts,
    row_ends,
    stride0,
    stride1,
    TOPK: tl.constexpr,
    CAP: tl.constexpr,
    PAD_TILE: tl.constexpr,
    BLOCK: tl.constexpr,
    SAMPLE: tl.constexpr,
    CHUNKS: tl.constexpr,
    SAMPLE_ROUNDS: tl.constexpr,
):
    """Each program handles one row: sample to set a lower bound, one full scan to collect
    candidates, then exact bisection and placement on the candidate set."""
    row_id = tl.program_id(0)
    row_start = tl.load(row_starts + row_id)
    row_end = tl.load(row_ends + row_id)
    row_len = row_end - row_start
    row_ptr = logits_ptr + row_id * stride0 + row_start * stride1
    out_ptr = out_indices_ptr + row_id * TOPK
    k = tl.minimum(row_len, TOPK)

    # ---- 1) Block sampling, bisect on the sample to obtain the candidate-collection lower bound thr_lo ----
    chunk_len: tl.constexpr = SAMPLE // CHUNKS
    s_lane = tl.arange(0, SAMPLE)
    s_off = (s_lane // chunk_len) * (row_len // CHUNKS) + (s_lane % chunk_len)
    s_in = s_off < row_len
    s_x = tl.load(row_ptr + s_off * stride1, mask=s_in, other=float("-inf"))
    zero_u32 = tl.zeros([SAMPLE], dtype=tl.uint32)
    s_key = tl.where(s_in, _to_order_key(s_x), zero_u32)

    # The r-th largest key in the sample ≈ the 1.5*top_k-th largest key in the population: this both ensures the
    # candidate count >= top_k
    # and leaves enough headroom for CAP (sampling noise is far smaller than this 2x safety factor).
    s_count = tl.sum(s_in.to(tl.int32))
    r = (3 * k * s_count + 2 * row_len - 1) // (2 * row_len)
    r = tl.minimum(tl.maximum(r, 1), s_count)
    lo = tl.zeros((), dtype=tl.uint32)
    hi = tl.full((), 0xFFFFFFFF, dtype=tl.uint32)
    for sample_round in tl.static_range(0, SAMPLE_ROUNDS):
        span = hi - lo
        mid = lo + (span >> 1) + (span & 1)
        count = tl.sum((s_key >= mid).to(tl.int32))
        upper_half = count >= r
        lo = tl.where(upper_half, mid, lo)
        hi = tl.where(upper_half, hi, mid - 1)
    thr_lo = lo

    # ---- 2) One full scan: collect elements with key >= thr_lo into the candidate buffer in scan order ----
    lane = tl.arange(0, BLOCK)
    cand_key_row = cand_key_ptr + row_id * CAP
    cand_idx_row = cand_idx_ptr + row_id * CAP
    written = tl.zeros((), dtype=tl.int32)
    for off in tl.range(0, row_len, BLOCK):
        idx = off + lane
        in_range = idx < row_len
        x = tl.load(row_ptr + idx * stride1, mask=in_range, other=float("-inf"))
        key = _to_order_key(x)
        take = in_range & (key >= thr_lo)
        take_i32 = take.to(tl.int32)
        rank = written + tl.cumsum(take_i32, axis=0) - take_i32
        keep = take & (rank < CAP)
        tl.store(cand_key_row + rank, key, mask=keep)
        tl.store(cand_idx_row + rank, idx, mask=keep)
        written += tl.sum(take_i32)

    # ---- 3) If the candidate set is unusable (fewer than top_k or truncated by CAP), fall back in place for that row
    # ----
    if (written < k) | (written > CAP):
        _bisect_row_into(
            row_ptr, out_ptr, row_len, stride1, TOPK=TOPK, BLOCK_SIZE=BLOCK
        )
    else:
        # ---- 4) 32 rounds of exact bisection on the candidate set, then place the output in scan order ----
        c_lane = tl.arange(0, CAP)
        c_in = c_lane < written
        c_key = tl.load(cand_key_row + c_lane, mask=c_in, other=0)
        lo2 = tl.zeros((), dtype=tl.uint32)
        hi2 = tl.full((), 0xFFFFFFFF, dtype=tl.uint32)
        for select_round in tl.range(0, 32):
            span = hi2 - lo2
            mid = lo2 + (span >> 1) + (span & 1)
            count = tl.sum((c_in & (c_key >= mid)).to(tl.int32))
            upper_half = count >= k
            lo2 = tl.where(upper_half, mid, lo2)
            hi2 = tl.where(upper_half, hi2, mid - 1)
        thr = lo2

        # Fill slots after valid entries with -1 (tail placeholder when row_len < TOPK)
        pad = tl.arange(0, PAD_TILE)
        tl.store(out_ptr + pad, tl.full([PAD_TILE], -1, tl.int32), mask=pad >= k)

        c_pos = tl.load(cand_idx_row + c_lane, mask=c_in, other=0)
        take_hi = c_in & (c_key > thr)
        take_hi_i32 = take_hi.to(tl.int32)
        rank_hi = tl.cumsum(take_hi_i32, axis=0) - take_hi_i32
        tl.store(out_ptr + rank_hi, c_pos, mask=take_hi)
        n_hi = tl.sum(take_hi_i32)

        # Tied elements fill remaining slots in scan order (any of the tied elements may be chosen)
        take_eq = c_in & (c_key == thr)
        take_eq_i32 = take_eq.to(tl.int32)
        rank_eq = tl.cumsum(take_eq_i32, axis=0) - take_eq_i32
        slot = n_hi + rank_eq
        tl.store(out_ptr + slot, c_pos, mask=take_eq & (slot < k))


@triton.jit
def non_tle_top_k_per_row_prefill(
    logits_ptr,
    out_indices_ptr,
    row_starts,
    row_ends,
    stride0,
    stride1,
    vocab_size,
    s_histogram_ptr,
    s_final_logits_ptr,
    s_final_cnt_ptr,
    s_threshold_bin_idx_ptr,
    s_final_bin_size_ptr,
    s_found_topk_values_ptr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROW_OFFSET: tl.constexpr,
):
    VEC: tl.constexpr = 4
    NUM_BINS: tl.constexpr = 2048
    NUM_FILNAL_ITEMS: tl.constexpr = 2048

    row_id = tl.program_id(0) + ROW_OFFSET
    row_start = tl.load(row_starts + row_id)
    row_end = tl.load(row_ends + row_id)
    logits_ptr += row_id * stride0
    # float4 align
    x_off_mod = (row_id * stride0 + row_start) % VEC
    skip_elems = 0 if x_off_mod == 0 else VEC - x_off_mod
    out_indices_ptr += row_id * TOPK

    s_histogram_ptr += row_id * NUM_BINS
    s_final_logits_ptr += row_id * NUM_FILNAL_ITEMS
    s_final_cnt_ptr += row_id
    s_threshold_bin_idx_ptr += row_id
    s_final_bin_size_ptr += row_id
    s_found_topk_values_ptr += row_id

    _top_k_per_row_job(
        logits_ptr,
        out_indices_ptr,
        row_start,
        row_end,
        stride1,
        vocab_size,
        skip_elems,
        None,
        None,
        s_histogram_ptr,
        s_final_logits_ptr,
        s_final_cnt_ptr,
        s_threshold_bin_idx_ptr,
        s_final_bin_size_ptr,
        s_found_topk_values_ptr,
        out_indices_ptr,
        None,
        TOPK=TOPK,
        BLOCK_SIZE=BLOCK_SIZE,
        USE_RADIX_FINAL=False,
        HAS_TLE=False,
        MULTIPLE_BLOCKS_PER_ROW=False,
        MERGE_BLOCKS=False,
    )


def top_k_per_row_prefill(
    logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
):
    """Top-K per row for prefill phase of DeepSeek V4 sparse attention.

    Selects top_k indices from a single row of logits using radix-based
    selection. Only valid elements within [row_start, row_end) are considered.

    Args:
        logits: [num_rows, vocab_size] float32 tensor.
        row_starts: [num_rows] int32 — start of valid range per row (inclusive).
        row_ends: [num_rows] int32 — end of valid range per row (exclusive).
        indices: [num_rows, top_k] int32 — output buffer, filled with 0-based indices
                 relative to row_starts[i]. Caller pre-allocates this.
        num_rows: number of rows.
        stride0: logits.stride(0), typically == vocab_size for contiguous tensor.
        stride1: logits.stride(1), typically == 1 for contiguous tensor.
        top_k: number of top elements per row.

    GCU400 note: the TLE shared-memory kernel is not used on this target. For
    variable-length rows it mis-fills output slots (uninitialized smem surfaces
    as garbage indices, e.g. raw float bit patterns) and can select wrong
    values -- the same known issue the gcu400 ``bin_topk`` TLE kernel papers
    over by recomputing broken rows with aten ops on the host. The gcu400 copy
    of the sibling ``top_k_per_row_decode`` already dropped its TLE path for
    the same reason. Do the same here and always launch the non-TLE kernel,
    which uses per-row global scratch buffers instead of ``tle.gpu.alloc``
    shared memory and passes every accuracy shape on this device.

    The upstream non-TLE kernel (histogram + ``tl.atomic_add`` threshold
    search) aborts the enflame compiler, and both ``tl.histogram`` and global
    atomics run at ~1.5 GB/s on GCU, so the gcu400 copy replaces it with a
    candidate-set scheme: sample a lower-bound threshold, collect candidates
    in one full sweep, then run the exact bisect on that small candidate set.
    Rows whose candidate set is unusable fall back to the 34-sweep bisect.
    """
    logger.debug("GEMS_ENFLAME TOP_K_PER_ROW_PREFILL")

    assert num_rows == logits.shape[0]
    vocab_size = logits.shape[1]
    if vocab_size <= ROW_TILE_MAX:
        # A row fits in registers: all 32 bisection rounds and placement happen in registers, with only 1 memory pass.
        row_tile = triton.next_power_of_2(vocab_size)
        _gcu_rowwise_top_k_per_row[(num_rows,)](
            logits,
            indices,
            row_starts,
            row_ends,
            stride0,
            stride1,
            TOPK=top_k,
            TILE=row_tile,
            PAD_TILE=triton.next_power_of_2(top_k),
            num_warps=1,
        )
        return

    # Candidate buffer capacity: 4x top_k (the sampling lower bound is scaled to 1.5*top_k, leaving half the margin on
    # each side),
    # no need to exceed the full row length when the vocab itself is smaller.
    cap = min(triton.next_power_of_2(4 * top_k), triton.next_power_of_2(vocab_size))
    sample = CANDIDATE_SAMPLE_LARGE if vocab_size >= 65536 else CANDIDATE_SAMPLE_SMALL
    device = logits.device
    cand_key = torch.empty(num_rows * cap, device=device, dtype=torch.int32)
    cand_idx = torch.empty(num_rows * cap, device=device, dtype=torch.int32)
    _gcu_candidate_top_k_per_row[(num_rows,)](
        logits,
        indices,
        cand_key,
        cand_idx,
        row_starts,
        row_ends,
        stride0,
        stride1,
        TOPK=top_k,
        CAP=cap,
        PAD_TILE=triton.next_power_of_2(top_k),
        BLOCK=BISECT_BLOCK_SIZE,
        SAMPLE=sample,
        CHUNKS=CANDIDATE_SAMPLE_CHUNKS,
        SAMPLE_ROUNDS=CANDIDATE_SAMPLE_ROUNDS,
        # At most 64 lanes per thread: too few warps would spill the two large candidate/sampling tiles out of
        # registers,
        # measurements show that at 64 rows x 129280 this drops from 5.6ms to sub-millisecond.
        num_warps=max(1, cap // 2048),
    )


# ---------------------------------------------------------------------------
# Generic-module rebinding (test / bench / direct-import routing)
#
# SpecOpRegistrar only replaces the top-level ``flag_gems`` entry.
# ``from flag_gems.fused import top_k_per_row_prefill`` (used by the accuracy
# test and the benchmark) and ``import flag_gems.deprecated.top_k_per_row_prefill``
# keep resolving to the generic implementation, whose histogram+atomic kernel
# aborts the enflame compiler (``llvm::cast<DenseElementsAttr>`` assertion in
# make_gcuir). Rebind the generic module's public entry to this copy, the same
# way ``_install_gcu400_bin_topk`` routes ``DSA.bin_topk``. The generic module
# is always loaded first (``flag_gems/__init__`` imports it before registering
# the backend), so the lookups below find it; when absent the rebinding is a
# no-op. The vendor module can execute twice under two module names, and the
# identity guard only prevents pathological self-rebinding.
# ---------------------------------------------------------------------------
_generic_topk_module = sys.modules.get("flag_gems.deprecated.top_k_per_row_prefill")
if _generic_topk_module is not None and _generic_topk_module is not sys.modules.get(
    __name__
):
    _generic_topk_module.top_k_per_row_prefill = top_k_per_row_prefill

_generic_deprecated_package = sys.modules.get("flag_gems.deprecated")
if _generic_deprecated_package is not None:
    _generic_deprecated_package.top_k_per_row_prefill = top_k_per_row_prefill

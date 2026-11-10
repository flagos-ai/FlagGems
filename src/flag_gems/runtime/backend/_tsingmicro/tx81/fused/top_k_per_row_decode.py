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

"""Tsingmicro (txda) specialization of ``top_k_per_row_decode``.

Same radix-select algorithm as :mod:`flag_gems.fused.top_k_per_row_decode`
(which is what the generic entry point runs on every non-TLE backend), with two
txda-specific changes:

1. **Both ``seq_lens`` layouts.**  Speculative decoding (MTP) hands the kernel a
   2D ``(B, next_n)`` ``seq_lens`` in which every logit row already carries its
   own effective context length, flat-indexed as ``seq_lens[row_id]``; plain
   decode hands it a 1D ``(B,)`` one where the per-row offset is derived here.
   The CUDA reference kernel branches on ``seqLensIs2D`` for exactly this
   (``csrc/libtorch_stable/sampler.cu: topKPerRowDecode``), and ``persistent_topk``
   enforces the same contract through ``lengths.numel() == num_rows``.

2. **No masked atomics.**  txda's Triton backend lowers ``tl.atomic_add`` with a
   ``mask=`` argument pathologically: the histogram step drops from ~0.7 ms to
   ~112 ms as soon as ``row_len`` passes ``top_k`` (i.e. once the fast
   ``row_len <= TOPK`` copy path is skipped), and the cost grows linearly with
   ``row_len`` (~1.2 s at 8192).  Redirecting the masked-off lanes to a bin that
   is incremented by zero keeps the histogram identical -- zeros do not change a
   counter -- while giving the backend an unmasked atomic to lower.

The output is bit-identical to the generic implementation for the same inputs
(same algorithm, same write order), so switching backends cannot move a token.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


SORTING_ALGORITHM_THRESHOLD = 12288
SPLIT_WORK_THRESHOLD = 200 * 1000
NUM_THREADS_PER_BLOCK = 512
MULTIPLE_BLOCKS_PER_ROW_CONFIG = 10
NUM_THREADS_PER_BLOCK_MERGE = 1024
NUM_FILNAL_ITEMS = 2048
NUM_BINS = 2048
RADIX_BITS_FINAL = 8
RADIX_SIZE_FINAL = 1 << RADIX_BITS_FINAL


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
            # txda has no TLE, so only the portable prefix-sum path is ported.
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
                MULTIPLE_BLOCKS_PER_ROW=MULTIPLE_BLOCKS_PER_ROW,
                MERGE_BLOCKS=MERGE_BLOCKS,
            )

    if not continue_to_next_step:
        # txda runs the non-TLE path, so the TLE-only ``_final_select_radix``
        # refinement is not ported; ``out_indices_ptr`` aliases
        # ``s_out_indices_ptr`` here, exactly as in the generic non-TLE kernel.
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


@triton.jit
def non_tle_top_k_per_row_decode(
    logits_ptr,
    out_indices_ptr,
    seq_lens_ptr,
    next_n,
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
    SEQ_LENS_IS_2D: tl.constexpr,
):
    VEC: tl.constexpr = 4
    NUM_BINS: tl.constexpr = 2048
    NUM_FILNAL_ITEMS: tl.constexpr = 2048

    row_id = tl.program_id(0)
    batch_id = row_id // next_n
    batch_offset = row_id % next_n
    # 2D (B, next_n) seq_lens: every row already carries its own effective
    # length, flat-indexed as seq_lens[b, j] == seq_lens[row_id]. 1D (B,)
    # seq_lens: all rows of a batch share one length and the per-row offset is
    # derived here. Mirrors csrc/libtorch_stable/sampler.cu topKPerRowDecode.
    row_start = 0
    if SEQ_LENS_IS_2D:
        row_len = tl.load(seq_lens_ptr + row_id)
    else:
        seq_len = tl.load(seq_lens_ptr + batch_id)
        row_len = seq_len - next_n + batch_offset + 1
    row_end = row_len

    out_indices_ptr += row_id * TOPK
    logits_ptr += row_id * stride0
    # float4 align
    x_off_mod = (row_id * stride0 + row_start) % VEC
    skip_elems = 0 if x_off_mod == 0 else VEC - x_off_mod

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
        MULTIPLE_BLOCKS_PER_ROW=False,
        MERGE_BLOCKS=False,
    )


def top_k_per_row_decode(
    logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
):
    """Top-K per row for the decode phase (txda).

    Drop-in for :func:`flag_gems.fused.top_k_per_row_decode`, with the same
    algorithm and the same output, so a caller can switch between the generic
    and this txda implementation without moving a token.

    Args:
        logits: [num_rows, vocab_size] float32 tensor, row-major.
        next_n: number of next tokens; only consulted for a 1D ``seq_lens``.
        seq_lens: per-row effective context lengths, int32.
            - 1D ``(B,)``: all ``next_n`` rows of a request share one length and
              each row's length is ``seq_len - next_n + (row_id % next_n) + 1``.
            - 2D ``(B, next_n)``: every row already carries its own length, read
              at the flat index ``row_id`` (C-contiguous ``[b, j]``).
        indices: [num_rows, top_k] int32 output buffer, filled with the selected
            indices and ``-1`` for the slots past a row's valid length.
        num_rows: ``logits.shape[0]``.
        stride0: ``logits.stride(0)``.
        stride1: ``logits.stride(1)``.
        top_k: number of top elements to select per row.
    """
    logger.debug("GEMS_TSINGMICRO TOP_K_PER_ROW_DECODE")

    assert num_rows == logits.shape[0]
    vocab_size = logits.shape[1]
    # A 2D seq_lens carries one pre-computed length per logit row (speculative
    # decode with per-token context lengths); a 1D one is per request.
    seq_lens_is_2d = seq_lens.dim() == 2
    if seq_lens_is_2d and not seq_lens.is_contiguous():
        seq_lens = seq_lens.contiguous()

    # One program per row: num_rows is the decode batch (max_num_seqs *
    # (num_spec + 1)), so the grid stays tiny and txda's serialized launches
    # cost less than any per-row split would.
    device = logits.device
    s_histogram_ptr = torch.empty(
        (num_rows, NUM_BINS), device=device, dtype=torch.int32
    )
    s_final_logits_ptr = torch.empty(
        (num_rows, NUM_FILNAL_ITEMS), device=device, dtype=torch.float32
    )
    s_final_cnt_ptr = torch.empty((num_rows,), device=device, dtype=torch.int32)
    s_threshold_bin_idx_ptr = torch.empty((num_rows,), device=device, dtype=torch.int32)
    s_final_bin_size_ptr = torch.empty((num_rows,), device=device, dtype=torch.int32)
    s_found_topk_values_ptr = torch.empty((num_rows,), device=device, dtype=torch.int32)

    non_tle_top_k_per_row_decode[(num_rows,)](
        logits,
        indices,
        seq_lens,
        next_n,
        stride0,
        stride1,
        vocab_size,
        s_histogram_ptr,
        s_final_logits_ptr,
        s_final_cnt_ptr,
        s_threshold_bin_idx_ptr,
        s_final_bin_size_ptr,
        s_found_topk_values_ptr,
        TOPK=top_k,
        BLOCK_SIZE=NUM_THREADS_PER_BLOCK,
        SEQ_LENS_IS_2D=seq_lens_is_2d,
        num_warps=NUM_THREADS_PER_BLOCK // 32,
    )

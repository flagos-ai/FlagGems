"""Triton top_k_per_row_decode for DeepSeek V4 decode-phase token selection.

"""

import logging

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

TLE_FIXED_BLOCK_SIZE = 512
TLE_FIXED_NUM_STAGES = 1
TLE_RADIX_FINAL_SEQ_LEN_THRESHOLD = 12288
HIST_SIZE = 4096
RADIX_BITS_FINAL = 8
RADIX_SIZE_FINAL = 1 << RADIX_BITS_FINAL
CLUSTER_SEQ_LEN_THRESHOLD = 65536
CLUSTER_BLOCK_SIZE = 4096
MAX_CLUSTER_N = 64
SIGN_BIT = tl.constexpr(-(1 << 31))


@triton.jit
def _float_to_sortable(val):
    """Convert IEEE 754 float to order-preserving unsigned integer.

    XOR with sign-dependent mask so that sorted int order == sorted float order.
    """
    bits = val.to(tl.int32, bitcast=True)
    sign_ext = bits >> 31
    mask = sign_ext | tl.full(bits.shape, SIGN_BIT, dtype=tl.int32)
    return bits ^ mask


@triton.jit
def _convert_to_trt_uint32(x):
    bits = x.to(tl.uint32, bitcast=True)
    sign_mask = tl.full(bits.shape, 0x80000000, tl.uint32)
    sign_set = (bits & sign_mask) != 0
    inv = (~bits) & tl.full(bits.shape, 0x7FFFFFFF, tl.uint32)
    return tl.where(sign_set, bits, inv)


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
    x,
    in_range,
    ones,
    step_idx: tl.constexpr,
    logit_pattern,
    hist_base_ptr,
):
    RADIX11_MASK: tl.constexpr = 0x7FF
    RADIX10_MASK: tl.constexpr = 0x3FF
    key = _convert_to_trt_uint32(x)
    if step_idx == 0:
        digit = _convert_to_trt_uint16_hi11(x)
    elif step_idx == 1:
        digit = ((key >> 21) & RADIX11_MASK).to(tl.int32)
    elif step_idx == 2:
        digit = ((key >> 10) & RADIX11_MASK).to(tl.int32)
    else:
        digit = (key & RADIX10_MASK).to(tl.int32)

    if step_idx < 2:
        partial = in_range
    elif step_idx == 2:
        partial = in_range & (((key ^ logit_pattern) >> 21) == 0)
    else:
        partial = in_range & (((key ^ logit_pattern) >> 10) == 0)

    tl.atomic_add(
        hist_base_ptr + digit,
        ones,
        mask=partial,
        sem="relaxed",
        scope="cta",
    )


@triton.jit
def _process_bins(
    x,
    in_range,
    found_ptrs,
    ones,
    offs,
    final_cnt_ptrs,
    step_idx: tl.constexpr,
    logit_pattern,
    threshold_bin_idx,
    write_directly,
    s_out_indices_ptr,
    hist_base_ptr,
    use_final,
    TOPK: tl.constexpr,
):
    FINAL_SORT_ITEMS: tl.constexpr = 2048
    RADIX11_MASK: tl.constexpr = 0x7FF
    RADIX10_MASK: tl.constexpr = 0x3FF

    key = _convert_to_trt_uint32(x)
    if step_idx == 0:
        digit = _convert_to_trt_uint16_hi11(x)
    elif step_idx == 1:
        digit = ((key >> 21) & RADIX11_MASK).to(tl.int32)
    elif step_idx == 2:
        digit = ((key >> 10) & RADIX11_MASK).to(tl.int32)
    else:
        digit = (key & RADIX10_MASK).to(tl.int32)

    if step_idx < 2:
        partial = in_range
    elif step_idx == 2:
        partial = in_range & (((key ^ logit_pattern) >> 21) == 0)
    else:
        partial = in_range & (((key ^ logit_pattern) >> 10) == 0)

    take_lt = partial & (digit < threshold_bin_idx) & write_directly
    out_pos_lt = tl.atomic_add(
        found_ptrs,
        ones,
        mask=take_lt,
        sem="relaxed",
        scope="cta",
    )
    tl.store(
        s_out_indices_ptr + out_pos_lt,
        offs.to(tl.int32),
        mask=take_lt & (out_pos_lt < TOPK),
    )

    if step_idx == 3:
        take_eq = partial & (digit == threshold_bin_idx)
        out_pos_eq = tl.atomic_add(
            hist_base_ptr + digit,
            ones,
            mask=take_eq,
            sem="relaxed",
            scope="cta",
        )
        tl.store(
            s_out_indices_ptr + out_pos_eq,
            offs.to(tl.int32),
            mask=take_eq & (out_pos_eq < TOPK),
        )
    elif use_final:
        take_eq_final = partial & (digit == threshold_bin_idx)
        final_pos = tl.atomic_add(
            final_cnt_ptrs,
            ones,
            mask=take_eq_final,
            sem="relaxed",
            scope="cta",
        )
        tl.store(
            hist_base_ptr + final_pos,
            offs.to(tl.int32),
            mask=take_eq_final & (final_pos < FINAL_SORT_ITEMS),
        )
        tl.store(
            hist_base_ptr + (FINAL_SORT_ITEMS + final_pos),
            x.to(tl.int32, bitcast=True),
            mask=take_eq_final & (final_pos < FINAL_SORT_ITEMS),
        )


@triton.jit
def _processHistogramStep(
    row_ptr,
    stride_xn,
    row_start,
    row_end,
    seq_len,
    step_idx: tl.constexpr,
    logit_pattern,
    s_step_thresholds_ptr,
    found_topk_values,
    hist_base_ptr,
    s_out_indices_ptr,
    s_final_cnt_ptr,
    s_found_topk_values_ptr,
    s_threshold_bin_idx_ptr,
    s_final_bin_size_ptr,
    assume_aligned,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_TLE: tl.constexpr,
):
    VEC: tl.constexpr = 4
    FINAL_SORT_ITEMS: tl.constexpr = 2048
    RADIX11_SIZE: tl.constexpr = 2048
    RADIX11_MASK: tl.constexpr = 0x7FF
    RADIX10_SIZE: tl.constexpr = 1024

    lane = tl.arange(0, BLOCK_SIZE)
    vec = tl.arange(0, VEC)
    ones = tl.full([BLOCK_SIZE], 1, tl.int32)
    ones_vec_2d = tl.full([BLOCK_SIZE, VEC], 1, tl.int32)
    zeros = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    zeros_vec_2d = tl.zeros([BLOCK_SIZE, VEC], dtype=tl.int32)

    for clear_round in tl.range(0, RADIX11_SIZE // BLOCK_SIZE):
        clear_bins = clear_round * BLOCK_SIZE + lane
        tl.store(hist_base_ptr + clear_bins, 0)
    tl.debug_barrier()

    if step_idx == 2:
        step1_threshold = tl.load(s_step_thresholds_ptr + 1)
        logit_pattern = (step1_threshold.to(tl.uint32) & RADIX11_MASK) << 21
    elif step_idx == 3:
        step1_threshold = tl.load(s_step_thresholds_ptr + 1)
        step2_threshold = tl.load(s_step_thresholds_ptr + 2)
        logit_pattern = ((step1_threshold.to(tl.uint32) & RADIX11_MASK) << 21) | (
            (step2_threshold.to(tl.uint32) & RADIX11_MASK) << 10
        )

    n_tiles = tl.cdiv(seq_len, BLOCK_SIZE)
    n_vec_full = seq_len // (BLOCK_SIZE * VEC)
    rem_tiles = (seq_len - n_vec_full * BLOCK_SIZE * VEC) // BLOCK_SIZE

    if assume_aligned:
        for t in tl.range(0, n_vec_full):
            base = t * BLOCK_SIZE * VEC + lane * VEC
            offs = base[:, None] + vec[None, :]
            x_vec = tl.load(row_ptr + offs)
            _distribute_to_bins(
                x_vec,
                True,
                ones_vec_2d,
                step_idx,
                logit_pattern,
                hist_base_ptr,
            )
        for t in tl.range(0, rem_tiles):
            offs = (n_vec_full * VEC + t) * BLOCK_SIZE + lane
            x = tl.load(row_ptr + offs)
            _distribute_to_bins(
                x,
                True,
                ones,
                step_idx,
                logit_pattern,
                hist_base_ptr,
            )
    elif stride_xn == 1:
        row_len = row_end - row_start
        n_vec_full = row_len // (BLOCK_SIZE * VEC)
        rem_tiles = (row_len - n_vec_full * BLOCK_SIZE * VEC) // BLOCK_SIZE
        rem_elems = row_len % BLOCK_SIZE
        for t in tl.range(0, n_vec_full):
            base = t * BLOCK_SIZE * VEC + lane * VEC
            offs = base[:, None] + vec[None, :]
            x_vec = tl.load(row_ptr + row_start + offs)
            _distribute_to_bins(
                x_vec,
                True,
                ones_vec_2d,
                step_idx,
                logit_pattern,
                hist_base_ptr,
            )
        for t in tl.range(0, rem_tiles):
            offs = (n_vec_full * VEC + t) * BLOCK_SIZE + lane
            x = tl.load(row_ptr + row_start + offs)
            _distribute_to_bins(
                x,
                True,
                ones,
                step_idx,
                logit_pattern,
                hist_base_ptr,
            )
        if rem_elems > 0:
            offs = (n_vec_full * VEC + rem_tiles) * BLOCK_SIZE + lane
            in_range = lane < rem_elems
            x = tl.load(row_ptr + row_start + offs, mask=in_range, other=float("-inf"))
            _distribute_to_bins(
                x,
                in_range,
                ones,
                step_idx,
                logit_pattern,
                hist_base_ptr,
            )
    else:
        row_len = row_end - row_start
        n_tiles = tl.cdiv(row_len, BLOCK_SIZE)
        for t in tl.range(0, n_tiles):
            offs = t * BLOCK_SIZE + lane
            in_range = offs < row_len
            x = tl.load(
                row_ptr + row_start + offs * stride_xn,
                mask=in_range,
                other=float("-inf"),
            )
            _distribute_to_bins(
                x,
                in_range,
                ones,
                step_idx,
                logit_pattern,
                hist_base_ptr,
            )
    tl.debug_barrier()

    # TRT-style threshold search with per-round early-exit.
    tl.store(s_threshold_bin_idx_ptr, -1)
    tl.store(s_final_bin_size_ptr, 0)
    tl.debug_barrier()
    threshold_bin_ptrs = s_threshold_bin_idx_ptr + zeros
    final_bin_size_ptrs = s_final_bin_size_ptr + zeros
    last_value = found_topk_values
    threshold_found = False
    threshold_rounds = tl.where(
        step_idx == 3,
        RADIX10_SIZE // BLOCK_SIZE,
        RADIX11_SIZE // BLOCK_SIZE,
    )
    for round_idx in tl.range(0, threshold_rounds):
        if not threshold_found:
            bins = round_idx * BLOCK_SIZE + lane
            counts = tl.load(hist_base_ptr + bins)
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
            tl.store(hist_base_ptr + bins, prefix_sum)
            tl.store(threshold_bin_ptrs, threshold_bin, mask=threshold_mask)
            tl.store(final_bin_size_ptrs, threshold_bin_size, mask=threshold_mask)
            found_round = tl.reduce_or(threshold_mask, axis=0)
            threshold_found = found_round
            last_value = total_sum

    threshold_bin_idx = tl.load(s_threshold_bin_idx_ptr)
    final_bin_size = tl.load(s_final_bin_size_ptr)
    tl.store(s_step_thresholds_ptr + step_idx, threshold_bin_idx)

    use_final = (
        (step_idx < 3) & (threshold_bin_idx >= 0) & (final_bin_size <= FINAL_SORT_ITEMS)
    )
    write_directly = ((step_idx == 0) & (final_bin_size <= FINAL_SORT_ITEMS)) | (
        step_idx >= 1
    )
    if use_final:
        tl.store(s_final_cnt_ptr, 0)
        tl.debug_barrier()

    found_ptrs = s_found_topk_values_ptr + zeros
    final_cnt_ptrs = s_final_cnt_ptr + zeros
    if assume_aligned:
        found_ptrs_vec_2d = s_found_topk_values_ptr + zeros_vec_2d
        final_cnt_ptrs_vec_2d = s_final_cnt_ptr + zeros_vec_2d
        for t in tl.range(0, n_vec_full):
            base = t * BLOCK_SIZE * VEC + lane * VEC
            offs = base[:, None] + vec[None, :]
            x_vec = tl.load(row_ptr + offs)
            _process_bins(
                x_vec,
                True,
                found_ptrs_vec_2d,
                ones_vec_2d,
                offs,
                final_cnt_ptrs_vec_2d,
                step_idx,
                logit_pattern,
                threshold_bin_idx,
                write_directly,
                s_out_indices_ptr,
                hist_base_ptr,
                use_final,
                TOPK=TOPK,
            )
        for t in tl.range(0, rem_tiles):
            offs = (n_vec_full * VEC + t) * BLOCK_SIZE + lane
            x = tl.load(row_ptr + offs)
            _process_bins(
                x,
                True,
                found_ptrs,
                ones,
                offs,
                final_cnt_ptrs,
                step_idx,
                logit_pattern,
                threshold_bin_idx,
                write_directly,
                s_out_indices_ptr,
                hist_base_ptr,
                use_final,
                TOPK=TOPK,
            )
    elif stride_xn == 1:
        row_len = row_end - row_start
        n_vec_full = row_len // (BLOCK_SIZE * VEC)
        rem_tiles = (row_len - n_vec_full * BLOCK_SIZE * VEC) // BLOCK_SIZE
        rem_elems = row_len % BLOCK_SIZE
        found_ptrs_vec_2d = s_found_topk_values_ptr + zeros_vec_2d
        final_cnt_ptrs_vec_2d = s_final_cnt_ptr + zeros_vec_2d
        for t in tl.range(0, n_vec_full):
            base = t * BLOCK_SIZE * VEC + lane * VEC
            offs = base[:, None] + vec[None, :]
            x_vec = tl.load(row_ptr + row_start + offs)
            _process_bins(
                x_vec,
                True,
                found_ptrs_vec_2d,
                ones_vec_2d,
                offs,
                final_cnt_ptrs_vec_2d,
                step_idx,
                logit_pattern,
                threshold_bin_idx,
                write_directly,
                s_out_indices_ptr,
                hist_base_ptr,
                use_final,
                TOPK=TOPK,
            )
        for t in tl.range(0, rem_tiles):
            offs = (n_vec_full * VEC + t) * BLOCK_SIZE + lane
            x = tl.load(row_ptr + row_start + offs)
            _process_bins(
                x,
                True,
                found_ptrs,
                ones,
                offs,
                final_cnt_ptrs,
                step_idx,
                logit_pattern,
                threshold_bin_idx,
                write_directly,
                s_out_indices_ptr,
                hist_base_ptr,
                use_final,
                TOPK=TOPK,
            )
        if rem_elems > 0:
            offs = (n_vec_full * VEC + rem_tiles) * BLOCK_SIZE + lane
            in_range = lane < rem_elems
            x = tl.load(row_ptr + row_start + offs, mask=in_range, other=float("-inf"))
            _process_bins(
                x,
                in_range,
                found_ptrs,
                ones,
                offs,
                final_cnt_ptrs,
                step_idx,
                logit_pattern,
                threshold_bin_idx,
                write_directly,
                s_out_indices_ptr,
                hist_base_ptr,
                use_final,
                TOPK=TOPK,
            )
    else:
        row_len = row_end - row_start
        n_tiles = tl.cdiv(row_len, BLOCK_SIZE)
        for t in tl.range(0, n_tiles):
            offs = t * BLOCK_SIZE + lane
            in_range = offs < row_len
            x = tl.load(
                row_ptr + row_start + offs * stride_xn,
                mask=in_range,
                other=float("-inf"),
            )
            _process_bins(
                x,
                in_range,
                found_ptrs,
                ones,
                offs,
                final_cnt_ptrs,
                step_idx,
                logit_pattern,
                threshold_bin_idx,
                write_directly,
                s_out_indices_ptr,
                hist_base_ptr,
                use_final,
                TOPK=TOPK,
            )
    tl.debug_barrier()
    return final_bin_size > FINAL_SORT_ITEMS


@triton.jit
def _final_select_radix(
    hist_base_ptr,
    s_out_indices_ptr,
    s_final_cnt_ptr,
    s_found_topk_values_ptr,
    s_radix_count_ptr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    FINAL_SORT_ITEMS: tl.constexpr,
    HAS_TLE: tl.constexpr,
):
    RADIX_BITS_FINAL: tl.constexpr = 8
    RADIX_SIZE_FINAL: tl.constexpr = 1 << RADIX_BITS_FINAL
    RADIX_MASK_FINAL: tl.constexpr = RADIX_SIZE_FINAL - 1
    DIGIT_START: tl.constexpr = 32 - RADIX_BITS_FINAL

    lane = tl.arange(0, BLOCK_SIZE)
    ones = tl.full([BLOCK_SIZE], 1, tl.int32)
    zeros = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    bins = tl.arange(0, RADIX_SIZE_FINAL)

    radix_count_vec_ptr = s_radix_count_ptr + bins
    base_idx = tl.load(s_found_topk_values_ptr)
    final_cnt = tl.minimum(tl.load(s_final_cnt_ptr), FINAL_SORT_ITEMS)
    remain = tl.minimum(TOPK - base_idx, final_cnt)
    tl.debug_barrier()

    if remain > 0:
        desired = tl.zeros((), dtype=tl.uint32)
        desired_mask = tl.zeros((), dtype=tl.uint32)
        k_to_find = remain + 1

        for digit_pos in tl.static_range(DIGIT_START, -1, -RADIX_BITS_FINAL):
            tl.store(s_radix_count_ptr + lane, 0, mask=lane < RADIX_SIZE_FINAL)
            tl.debug_barrier()

            cnt_tiles = tl.cdiv(final_cnt, BLOCK_SIZE)
            for t in tl.range(0, cnt_tiles):
                pos = t * BLOCK_SIZE + lane
                valid = pos < final_cnt
                x_bits_i32 = tl.load(
                    hist_base_ptr + (FINAL_SORT_ITEMS + pos),
                    mask=valid,
                    other=0,
                )
                x = x_bits_i32.to(tl.float32, bitcast=True)
                key = _convert_to_trt_uint32(x)
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
            if HAS_TLE:
                prefix_sum, _ = tle.cumsum(counts, axis=0, reverse=False)
            else:
                prefix_sum = tl.sum(counts) - tl.cumsum(counts, axis=0, reverse=True)
            next_prefix_sum = prefix_sum + counts
            threshold_mask = (prefix_sum < k_to_find) & (next_prefix_sum >= k_to_find)
            threshold_init = tl.full((), RADIX_SIZE_FINAL, dtype=tl.int32)
            threshold_bin = tl.min(
                tl.where(threshold_mask, bins, threshold_init), axis=0
            ).to(tl.int32)
            threshold_bin = tl.where(
                threshold_bin == RADIX_SIZE_FINAL, RADIX_SIZE_FINAL - 1, threshold_bin
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
            idx = tl.load(hist_base_ptr + pos, mask=valid, other=0)
            x_bits_i32 = tl.load(
                hist_base_ptr + (FINAL_SORT_ITEMS + pos),
                mask=valid,
                other=0,
            )
            x = x_bits_i32.to(tl.float32, bitcast=True)
            key = _convert_to_trt_uint32(x)
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

        tl.debug_barrier()
        cur = tl.load(s_found_topk_values_ptr)
        if cur < TOPK:
            for t in tl.range(0, cnt_tiles):
                cur = tl.load(s_found_topk_values_ptr)
                if cur < TOPK:
                    pos = t * BLOCK_SIZE + lane
                    valid = pos < final_cnt
                    idx = tl.load(hist_base_ptr + pos, mask=valid, other=0)
                    x_bits_i32 = tl.load(
                        hist_base_ptr + (FINAL_SORT_ITEMS + pos),
                        mask=valid,
                        other=0,
                    )
                    x = x_bits_i32.to(tl.float32, bitcast=True)
                    key = _convert_to_trt_uint32(x)
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

    tl.debug_barrier()
    tl.store(s_found_topk_values_ptr, TOPK)


@triton.jit
def _top_k_per_row_selector(
    row_ptr,
    out_row,
    row_start,
    row_end,
    stride_xn,
    stride_outn,
    vocab_size,
    hist_base_ptr,
    s_final_cnt_ptr,
    s_threshold_bin_idx_ptr,
    s_final_bin_size_ptr,
    s_found_topk_values_ptr,
    s_step_thresholds_ptr,
    s_out_indices_ptr,
    s_radix_count_ptr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_RADIX_FINAL: tl.constexpr,
    HAS_TLE: tl.constexpr,
):
    FINAL_SORT_ITEMS: tl.constexpr = 2048

    assume_aligned = (
        (row_start == 0)
        & (row_end == vocab_size)
        & (stride_xn == 1)
        & (stride_outn == 1)
        & ((vocab_size % BLOCK_SIZE) == 0)
    )
    if assume_aligned:
        tl.assume(row_start == 0)
        tl.assume(row_end == vocab_size)
        tl.assume(stride_xn == 1)
        tl.assume(stride_outn == 1)
        vocab_size = tl.multiple_of(vocab_size, BLOCK_SIZE)
    elif stride_xn == 1:
        tl.assume(stride_xn == 1)

    lane = tl.arange(0, BLOCK_SIZE)
    row_len = row_end - row_start
    if row_len <= TOPK:
        chunks: tl.constexpr = (TOPK + BLOCK_SIZE - 1) // BLOCK_SIZE
        for chunk_idx in tl.range(0, chunks):
            pos = chunk_idx * BLOCK_SIZE + lane
            take_row = pos < row_len
            tl.store(
                out_row + pos * stride_outn,
                (row_start + pos).to(tl.int32),
                mask=take_row,
            )
            take_pad = (pos >= row_len) & (pos < TOPK)
            tl.store(out_row + pos * stride_outn, -1, mask=take_pad)
        return

    tl.store(s_final_cnt_ptr, 0)
    tl.store(s_threshold_bin_idx_ptr, -1)
    tl.store(s_final_bin_size_ptr, 0)
    tl.store(s_found_topk_values_ptr, 0)

    logit_pattern = tl.zeros((), dtype=tl.uint32)
    continue_to_next_step = True
    init_chunks: tl.constexpr = (TOPK + BLOCK_SIZE - 1) // BLOCK_SIZE
    for init_idx in tl.range(0, init_chunks):
        pos = init_idx * BLOCK_SIZE + lane
        tl.store(s_out_indices_ptr + pos, -1, mask=pos < TOPK)

    tl.debug_barrier()
    for step_idx in tl.static_range(0, 4):
        if continue_to_next_step:
            found_topk_values = tl.load(s_found_topk_values_ptr)
            continue_to_next_step = _processHistogramStep(
                row_ptr,
                stride_xn,
                row_start,
                row_end,
                vocab_size,
                step_idx,
                logit_pattern,
                s_step_thresholds_ptr,
                found_topk_values,
                hist_base_ptr,
                s_out_indices_ptr,
                s_final_cnt_ptr,
                s_found_topk_values_ptr,
                s_threshold_bin_idx_ptr,
                s_final_bin_size_ptr,
                assume_aligned=assume_aligned,
                TOPK=TOPK,
                BLOCK_SIZE=BLOCK_SIZE,
                HAS_TLE=HAS_TLE,
            )

    if not continue_to_next_step:
        if USE_RADIX_FINAL:
            _final_select_radix(
                hist_base_ptr,
                s_out_indices_ptr,
                s_final_cnt_ptr,
                s_found_topk_values_ptr,
                s_radix_count_ptr,
                TOPK=TOPK,
                BLOCK_SIZE=BLOCK_SIZE,
                FINAL_SORT_ITEMS=FINAL_SORT_ITEMS,
                HAS_TLE=HAS_TLE,
            )
        else:
            base_idx = tl.load(s_found_topk_values_ptr)
            # Guard against stale/oversized counts to avoid out-of-bounds accesses
            # in the shared-memory final buffers.
            final_cnt = tl.minimum(tl.load(s_final_cnt_ptr), FINAL_SORT_ITEMS)
            sort_chunks = tl.cdiv(final_cnt, BLOCK_SIZE)
            for sort_chunk in tl.range(0, sort_chunks):
                pos = sort_chunk * BLOCK_SIZE + lane
                valid = pos < final_cnt
                logit_i_bits = tl.load(
                    hist_base_ptr + FINAL_SORT_ITEMS + pos,
                    mask=valid,
                    other=0,
                )
                logit_i = logit_i_bits.to(tl.float32, bitcast=True)
                out_rank = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
                for j in tl.range(0, final_cnt):
                    logit_j_bits = tl.load(hist_base_ptr + FINAL_SORT_ITEMS + j)
                    logit_j = logit_j_bits.to(tl.float32, bitcast=True)
                    better = (logit_i < logit_j) | ((logit_i == logit_j) & (pos < j))
                    out_rank = out_rank + (valid & better).to(tl.int32)
                dst_pos = base_idx + out_rank
                take = valid & (dst_pos < TOPK)
                idx_i = tl.load(
                    hist_base_ptr + pos,
                    mask=take,
                    other=0,
                )
                tl.store(s_out_indices_ptr + dst_pos, idx_i, mask=take)
            tl.debug_barrier()
            tl.store(s_found_topk_values_ptr, TOPK)

    flush_chunks: tl.constexpr = (TOPK + BLOCK_SIZE - 1) // BLOCK_SIZE
    for flush_chunk in tl.static_range(flush_chunks):
        pos = flush_chunk * BLOCK_SIZE + lane
        mask = pos < TOPK
        out_vals = tl.load(s_out_indices_ptr + pos, mask=mask, other=-1)
        tl.store(out_row + pos * stride_outn, out_vals, mask=mask)


@triton.jit
def top_k_per_row_selector_wrapper(
    x_ptr,
    out_ptr,
    row_start,
    row_end,
    stride_xm,
    stride_xn,
    stride_outm,
    stride_outn,
    vocab_size,
    hist_base_ptr,
    s_final_cnt_ptr,
    s_threshold_bin_idx_ptr,
    s_final_bin_size_ptr,
    s_found_topk_values_ptr,
    s_step_thresholds_ptr,
    s_out_indices_ptr,
    s_radix_count_ptr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_RADIX_FINAL: tl.constexpr,
    HIST_SIZE: tl.constexpr,
    RADIX_SIZE_FINAL: tl.constexpr,
):
    pid = tl.program_id(0)
    x_ptr += pid * stride_xm
    out_ptr += pid * stride_outm
    hist_base_ptr += pid * HIST_SIZE
    s_final_cnt_ptr += pid
    s_threshold_bin_idx_ptr += pid
    s_final_bin_size_ptr += pid
    s_found_topk_values_ptr += pid
    s_step_thresholds_ptr += pid * 4
    s_out_indices_ptr += pid * TOPK
    if USE_RADIX_FINAL:
        s_radix_count_ptr += pid * RADIX_SIZE_FINAL

    _top_k_per_row_selector(
        x_ptr,
        out_ptr,
        row_start,
        row_end,
        stride_xn,
        stride_outn,
        vocab_size,
        hist_base_ptr,
        s_final_cnt_ptr,
        s_threshold_bin_idx_ptr,
        s_final_bin_size_ptr,
        s_found_topk_values_ptr,
        s_step_thresholds_ptr,
        s_out_indices_ptr,
        s_radix_count_ptr,
        TOPK=TOPK,
        BLOCK_SIZE=BLOCK_SIZE,
        USE_RADIX_FINAL=USE_RADIX_FINAL,
        HAS_TLE=False,
    )


@triton.jit
def top_k_per_row_decode_wrapper(
    x_ptr,
    out_ptr,
    seq_lens_ptr,
    next_n,
    stride_xm,
    stride_xn,
    stride_outm,
    stride_outn,
    vocab_size,
    hist_base_ptr,
    s_final_cnt_ptr,
    s_threshold_bin_idx_ptr,
    s_final_bin_size_ptr,
    s_found_topk_values_ptr,
    s_step_thresholds_ptr,
    s_out_indices_ptr,
    s_radix_count_ptr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_RADIX_FINAL: tl.constexpr,
    HIST_SIZE: tl.constexpr,
    RADIX_SIZE_FINAL: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_id = pid // next_n
    batch_offset = pid % next_n
    seq_len = tl.load(seq_lens_ptr + batch_id)
    row_start = 0
    row_len = seq_len - next_n + batch_offset + 1
    row_end = row_len

    top_k_per_row_selector_wrapper(
        x_ptr,
        out_ptr,
        row_start,
        row_end,
        stride_xm,
        stride_xn,
        stride_outm,
        stride_outn,
        vocab_size,
        hist_base_ptr,
        s_final_cnt_ptr,
        s_threshold_bin_idx_ptr,
        s_final_bin_size_ptr,
        s_found_topk_values_ptr,
        s_step_thresholds_ptr,
        s_out_indices_ptr,
        s_radix_count_ptr,
        TOPK=TOPK,
        BLOCK_SIZE=BLOCK_SIZE,
        USE_RADIX_FINAL=USE_RADIX_FINAL,
        HIST_SIZE=HIST_SIZE,
        RADIX_SIZE_FINAL=RADIX_SIZE_FINAL,
    )


@triton.jit
def top_k_per_row_prefill_wrapper(
    x_ptr,
    out_ptr,
    row_starts_ptr,
    row_ends_ptr,
    stride_xm,
    stride_xn,
    stride_outm,
    stride_outn,
    vocab_size,
    hist_base_ptr,
    s_final_cnt_ptr,
    s_threshold_bin_idx_ptr,
    s_final_bin_size_ptr,
    s_found_topk_values_ptr,
    s_step_thresholds_ptr,
    s_out_indices_ptr,
    s_radix_count_ptr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_RADIX_FINAL: tl.constexpr,
    HIST_SIZE: tl.constexpr,
    RADIX_SIZE_FINAL: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = tl.load(row_starts_ptr + pid)
    row_end = tl.load(row_ends_ptr + pid)

    top_k_per_row_selector_wrapper(
        x_ptr,
        out_ptr,
        row_start,
        row_end,
        stride_xm,
        stride_xn,
        stride_outm,
        stride_outn,
        vocab_size,
        hist_base_ptr,
        s_final_cnt_ptr,
        s_threshold_bin_idx_ptr,
        s_final_bin_size_ptr,
        s_found_topk_values_ptr,
        s_step_thresholds_ptr,
        s_out_indices_ptr,
        s_radix_count_ptr,
        TOPK=TOPK,
        BLOCK_SIZE=BLOCK_SIZE,
        USE_RADIX_FINAL=USE_RADIX_FINAL,
        HIST_SIZE=HIST_SIZE,
        RADIX_SIZE_FINAL=RADIX_SIZE_FINAL,
    )


# alloc smem before call _top_k_per_row_selector
@triton.jit
def tle_top_k_per_row_selector_wrapper(
    x_ptr,
    out_ptr,
    row_start,
    row_end,
    stride_xm,
    stride_xn,
    stride_outm,
    stride_outn,
    vocab_size,
    TOPK: tl.constexpr,
    TOPKP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_RADIX_FINAL: tl.constexpr,
):
    HIST_SIZE: tl.constexpr = 4096
    RADIX_BITS_FINAL: tl.constexpr = 8
    RADIX_SIZE_FINAL: tl.constexpr = 1 << RADIX_BITS_FINAL

    pid = tl.program_id(0)
    x_ptr += pid * stride_xm
    out_ptr += pid * stride_outm

    s_histogram = tle.gpu.alloc(
        [HIST_SIZE],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    # TRT-style union reuse:
    # - [0, FINAL_SORT_ITEMS): final indices (int32)
    # - [FINAL_SORT_ITEMS, 2*FINAL_SORT_ITEMS): final logits bitcast(int32)
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
    s_step_thresholds = tle.gpu.alloc(
        [4],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    hist_base_ptr = tle.gpu.local_ptr(s_histogram, (0,))
    s_final_cnt_ptr = tle.gpu.local_ptr(s_final_cnt, (0,))
    s_threshold_bin_idx_ptr = tle.gpu.local_ptr(s_threshold_bin_idx, (0,))
    s_final_bin_size_ptr = tle.gpu.local_ptr(s_final_bin_size, (0,))
    s_found_topk_values_ptr = tle.gpu.local_ptr(s_found_topk_values, (0,))
    s_step_thresholds_ptr = tle.gpu.local_ptr(s_step_thresholds, (0,))
    s_out_indices_ptr = tle.gpu.local_ptr(s_out_indices, (0,))
    if USE_RADIX_FINAL:
        s_radix_counts = tle.gpu.alloc(
            [RADIX_SIZE_FINAL],
            dtype=tl.int32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        s_radix_count_ptr = tle.gpu.local_ptr(s_radix_counts, (0,))
    else:
        s_radix_count_ptr = None

    _top_k_per_row_selector(
        x_ptr,
        out_ptr,
        row_start,
        row_end,
        stride_xn,
        stride_outn,
        vocab_size,
        hist_base_ptr,
        s_final_cnt_ptr,
        s_threshold_bin_idx_ptr,
        s_final_bin_size_ptr,
        s_found_topk_values_ptr,
        s_step_thresholds_ptr,
        s_out_indices_ptr,
        s_radix_count_ptr,
        TOPK=TOPK,
        BLOCK_SIZE=BLOCK_SIZE,
        USE_RADIX_FINAL=USE_RADIX_FINAL,
        HAS_TLE=True,
    )


@triton.jit
def tle_top_k_per_row_decode_wrapper(
    x_ptr,
    out_ptr,
    seq_lens_ptr,
    next_n,
    stride_xm,
    stride_xn,
    stride_outm,
    stride_outn,
    vocab_size,
    TOPK: tl.constexpr,
    TOPKP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_RADIX_FINAL: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_id = pid // next_n
    batch_offset = pid % next_n
    seq_len = tl.load(seq_lens_ptr + batch_id)
    row_start = 0
    row_len = seq_len - next_n + batch_offset + 1
    row_end = row_len

    tle_top_k_per_row_selector_wrapper(
        x_ptr,
        out_ptr,
        row_start,
        row_end,
        stride_xm,
        stride_xn,
        stride_outm,
        stride_outn,
        vocab_size,
        TOPK=TOPK,
        TOPKP=TOPKP,
        BLOCK_SIZE=BLOCK_SIZE,
        USE_RADIX_FINAL=USE_RADIX_FINAL,
    )


@triton.jit
def tle_top_k_per_row_prefill_wrapper(
    x_ptr,
    out_ptr,
    row_starts_ptr,
    row_ends_ptr,
    stride_xm,
    stride_xn,
    stride_outm,
    stride_outn,
    vocab_size,
    TOPK: tl.constexpr,
    TOPKP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_RADIX_FINAL: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = tl.load(row_starts_ptr + pid)
    row_end = tl.load(row_ends_ptr + pid)

    tle_top_k_per_row_selector_wrapper(
        x_ptr,
        out_ptr,
        row_start,
        row_end,
        stride_xm,
        stride_xn,
        stride_outm,
        stride_outn,
        vocab_size,
        TOPK=TOPK,
        TOPKP=TOPKP,
        BLOCK_SIZE=BLOCK_SIZE,
        USE_RADIX_FINAL=USE_RADIX_FINAL,
    )


@triton.jit
def _top_k_per_row_selector_cluster(
    row_idx,
    block_idx,
    logits_ptr,
    indices_ptr,
    pb_hist_ptr,
    sync_ptr,
    counter_ptr,
    row_start,
    row_end,
    stride0,
    stride1,
    N,
    NUM_BLOCKS,
    BLOCK: tl.constexpr,
    TOP_K: tl.constexpr,
    TOP_KP: tl.constexpr,
):
    HIST_STRIDE: tl.constexpr = 256
    SYNC_STRIDE: tl.constexpr = 4
    CNT_STRIDE: tl.constexpr = 2

    row_len = row_end - row_start
    indices_row = indices_ptr + row_idx * TOP_K
    if row_len <= TOP_K:
        if block_idx * BLOCK < row_end:
            topk_offs = tl.arange(0, TOP_KP)
            topk_mask = (topk_offs >= block_idx * BLOCK) & (
                topk_offs < min(TOP_K, block_idx * BLOCK + BLOCK)
            )
            indices = tl.where(topk_offs < row_len, block_idx * BLOCK + topk_offs, -1)
            tl.store(indices_row + topk_offs, indices, mask=topk_mask)
        return

    offs = block_idx * BLOCK + tl.arange(0, BLOCK)
    valid = offs < row_len
    vals = tl.load(
        logits_ptr + row_idx * stride0 + (row_start + offs) * stride1,
        mask=valid,
        other=float("-inf"),
    )
    sortable = _float_to_sortable(vals)
    s_shifted = sortable ^ tl.full(sortable.shape, SIGN_BIT, dtype=tl.int32)

    pb_row = pb_hist_ptr + row_idx * NUM_BLOCKS * HIST_STRIDE
    sync_row = sync_ptr + row_idx * SYNC_STRIDE
    counter_row = counter_ptr + row_idx * CNT_STRIDE

    bins = tl.arange(0, 256)
    h_base = pb_row + block_idx * HIST_STRIDE

    # ── Iteration 0: byte 3 (MSB) ──────────────────────────
    bucket_0 = (sortable >> 24) & 0xFF
    local_hist = tl.histogram(bucket_0, 256, valid)
    tl.store(h_base + bins, local_hist)

    tl.debug_barrier()
    tl.atomic_add(sync_row, 1)
    while tl.atomic_add(sync_row, 0) < NUM_BLOCKS:
        pass

    counts = tl.zeros([256], dtype=tl.int32)
    for i in tl.static_range(NUM_BLOCKS):
        counts += tl.load(pb_row + i * HIST_STRIDE + bins)

    total = tl.sum(counts)
    ps = tl.cumsum(counts, axis=0)
    ss = total - ps + counts
    pivot_0 = tl.max(tl.where(ss >= TOP_K, bins, -1))
    ca_0 = tl.sum(tl.where(bins > pivot_0, counts, 0))
    remaining_k = TOP_K - ca_0
    match = (bucket_0 == pivot_0) & valid

    # ── Iteration 1: byte 2 ────────────────────────────────
    bucket_1 = (sortable >> 16) & 0xFF
    local_hist = tl.histogram(bucket_1, 256, match)
    tl.store(h_base + bins, local_hist)

    tl.debug_barrier()
    tl.atomic_add(sync_row + 1, 1)
    while tl.atomic_add(sync_row + 1, 0) < NUM_BLOCKS:
        pass

    counts = tl.zeros([256], dtype=tl.int32)
    for i in tl.static_range(NUM_BLOCKS):
        counts += tl.load(pb_row + i * HIST_STRIDE + bins)

    total = tl.sum(counts)
    ps = tl.cumsum(counts, axis=0)
    ss = total - ps + counts
    pivot_1 = tl.max(tl.where(ss >= remaining_k, bins, -1))
    ca_1 = tl.sum(tl.where(bins > pivot_1, counts, 0))
    remaining_k = remaining_k - ca_1
    match = match & (bucket_1 == pivot_1)

    # ── Iteration 2: byte 1 ────────────────────────────────
    bucket_2 = (sortable >> 8) & 0xFF
    local_hist = tl.histogram(bucket_2, 256, match)
    tl.store(h_base + bins, local_hist)

    tl.debug_barrier()
    tl.atomic_add(sync_row + 2, 1)
    while tl.atomic_add(sync_row + 2, 0) < NUM_BLOCKS:
        pass

    counts = tl.zeros([256], dtype=tl.int32)
    for i in tl.static_range(NUM_BLOCKS):
        counts += tl.load(pb_row + i * HIST_STRIDE + bins)

    total = tl.sum(counts)
    ps = tl.cumsum(counts, axis=0)
    ss = total - ps + counts
    pivot_2 = tl.max(tl.where(ss >= remaining_k, bins, -1))
    ca_2 = tl.sum(tl.where(bins > pivot_2, counts, 0))
    remaining_k = remaining_k - ca_2
    match = match & (bucket_2 == pivot_2)

    # ── Iteration 3: byte 0 (LSB) ──────────────────────────
    bucket_3 = sortable & 0xFF
    local_hist = tl.histogram(bucket_3, 256, match)
    tl.store(h_base + bins, local_hist)

    tl.debug_barrier()
    tl.atomic_add(sync_row + 3, 1)
    while tl.atomic_add(sync_row + 3, 0) < NUM_BLOCKS:
        pass

    counts = tl.zeros([256], dtype=tl.int32)
    for i in tl.static_range(NUM_BLOCKS):
        counts += tl.load(pb_row + i * HIST_STRIDE + bins)

    total = tl.sum(counts)
    ps = tl.cumsum(counts, axis=0)
    ss = total - ps + counts
    pivot_3 = tl.max(tl.where(ss >= remaining_k, bins, -1))
    ca_3 = tl.sum(tl.where(bins > pivot_3, counts, 0))
    remaining_k = remaining_k - ca_3

    # Selection phase
    threshold = (pivot_0 << 24) | (pivot_1 << 16) | (pivot_2 << 8) | pivot_3
    above_total = TOP_K - remaining_k

    t_shifted = threshold ^ SIGN_BIT

    above = (s_shifted > t_shifted) & valid
    equal = (sortable == threshold) & valid

    n_above = tl.sum(above.to(tl.int32))
    if n_above > 0:
        pa = tl.cumsum(above.to(tl.int32), axis=0)
        base_a = tl.atomic_add(counter_row, n_above)
        wp = base_a + pa - 1
        tl.store(
            indices_row + wp,
            offs.to(tl.int32),
            mask=above & (wp >= 0) & (wp < TOP_K),
        )

    n_equal = tl.sum(equal.to(tl.int32))
    if n_equal > 0:
        pe = tl.cumsum(equal.to(tl.int32), axis=0)
        base_e = tl.atomic_add(counter_row + 1, n_equal)
        wpe = above_total + base_e + pe - 1
        tl.store(
            indices_row + wpe,
            offs.to(tl.int32),
            mask=equal & ((base_e + pe - 1) < remaining_k) & (wpe >= 0) & (wpe < TOP_K),
        )


@triton.jit
def top_k_per_row_decode_cluster(
    logits_ptr,
    seq_len_ptr,
    indices_ptr,
    pb_hist_ptr,
    sync_ptr,
    counter_ptr,
    next_n,
    stride0,
    stride1,
    N: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
    TOP_K: tl.constexpr,
    TOP_KP: tl.constexpr,
):
    pid = tl.program_id(0)
    row_idx = pid // NUM_BLOCKS
    block_idx = pid % NUM_BLOCKS

    batch_id = row_idx // next_n
    batch_offset = row_idx % next_n
    seq_len = tl.load(seq_len_ptr + batch_id)
    row_end = seq_len - next_n + batch_offset + 1
    row_start = 0

    _top_k_per_row_selector_cluster(
        row_idx,
        block_idx,
        logits_ptr,
        indices_ptr,
        pb_hist_ptr,
        sync_ptr,
        counter_ptr,
        row_start,
        row_end,
        stride0,
        stride1,
        N,
        NUM_BLOCKS,
        BLOCK=BLOCK,
        TOP_K=TOP_K,
        TOP_KP=TOP_KP,
    )


@triton.jit
def top_k_per_row_prefill_cluster(
    logits_ptr,
    row_starts_ptr,
    row_ends_ptr,
    indices_ptr,
    pb_hist_ptr,
    sync_ptr,
    counter_ptr,
    stride0,
    stride1,
    N: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
    TOP_K: tl.constexpr,
    TOP_KP: tl.constexpr,
):
    pid = tl.program_id(0)
    row_idx = pid // NUM_BLOCKS
    block_idx = pid % NUM_BLOCKS

    row_start = tl.load(row_starts_ptr + row_idx)
    row_end = tl.load(row_ends_ptr + row_idx)

    _top_k_per_row_selector_cluster(
        row_idx,
        block_idx,
        logits_ptr,
        indices_ptr,
        pb_hist_ptr,
        sync_ptr,
        counter_ptr,
        row_start,
        row_end,
        stride0,
        stride1,
        N,
        NUM_BLOCKS,
        BLOCK=BLOCK,
        TOP_K=TOP_K,
        TOP_KP=TOP_KP,
    )


def top_k_per_row_decode(
    logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
):
    """Top-K per row for decode phase of DeepSeek V4.

    Selects top_k indices from a single row of logits using radix-based
    selection. Only valid elements within [0, seq_lens[0]) are considered.

    Args:
        logits: [num_rows, vocab_size] float32 tensor.
        next_n: number of next tokens (unused, kept for API compatibility).
        seq_lens: [B,] int32 — valid range [0, seq_lens[0]).
        indices: [num_rows, top_k] int32 — output buffer, filled with selected indices.
        num_rows: must be 1 (decode processes one row at a time).
        stride0: logits.stride(0).
        stride1: logits.stride(1).
        top_k: number of top elements to select.
    """
    logger.debug("GEMS TOP_K_PER_ROW_DECODE")

    assert num_rows == logits.shape[0]
    vocab_size = logits.shape[1]
    use_radix_final = vocab_size >= TLE_RADIX_FINAL_SEQ_LEN_THRESHOLD
    n_blocks = (vocab_size + CLUSTER_BLOCK_SIZE - 1) // CLUSTER_BLOCK_SIZE
    if (
        num_rows == 1
        and vocab_size >= CLUSTER_SEQ_LEN_THRESHOLD
        and n_blocks <= MAX_CLUSTER_N
    ):
        device = logits.device
        pb_size = n_blocks * 256
        pb_hist = torch.empty(num_rows * pb_size, dtype=torch.int32, device=device)
        sync = torch.zeros(num_rows * 4, dtype=torch.int32, device=device)
        counter = torch.zeros(num_rows * 2, dtype=torch.int32, device=device)
        top_k_pad = triton.next_power_of_2(top_k)
        top_k_per_row_decode_cluster[(num_rows * n_blocks,)](
            logits,
            seq_lens,
            indices,
            pb_hist,
            sync,
            counter,
            next_n,
            stride0,
            stride1,
            vocab_size,
            NUM_BLOCKS=n_blocks,
            BLOCK=CLUSTER_BLOCK_SIZE,
            TOP_K=top_k,
            TOP_KP=top_k_pad,
            num_warps=8,
        )
    elif HAS_TLE:
        topkp = triton.next_power_of_2(top_k)
        tle_top_k_per_row_decode_wrapper[(num_rows,)](
            logits,
            indices,
            seq_lens,
            next_n,
            stride0,
            stride1,
            indices.stride(0),
            indices.stride(1),
            vocab_size,
            TOPK=top_k,
            TOPKP=topkp,
            BLOCK_SIZE=TLE_FIXED_BLOCK_SIZE,
            USE_RADIX_FINAL=use_radix_final,
            num_warps=TLE_FIXED_BLOCK_SIZE // 32,
            num_stages=TLE_FIXED_NUM_STAGES,
        )
    else:
        # based on tle version
        device = logits.device
        hist_base_ptr = torch.empty(
            (num_rows, HIST_SIZE), device=device, dtype=torch.int32
        )
        s_final_cnt_ptr = torch.empty((num_rows,), device=device, dtype=torch.int32)
        s_threshold_bin_idx_ptr = torch.empty(
            (num_rows,), device=device, dtype=torch.int32
        )
        s_final_bin_size_ptr = torch.empty(
            (num_rows,), device=device, dtype=torch.int32
        )
        s_found_topk_values_ptr = torch.empty(
            (num_rows,), device=device, dtype=torch.int32
        )
        s_step_thresholds_ptr = torch.empty(
            (num_rows, 4), device=device, dtype=torch.int32
        )
        s_out_indices_ptr = torch.empty(
            (num_rows, top_k), device=device, dtype=torch.int32
        )
        s_radix_count_ptr = (
            torch.empty((num_rows, RADIX_SIZE_FINAL), device=device, dtype=torch.int32)
            if use_radix_final
            else None
        )
        top_k_per_row_decode_wrapper[(num_rows,)](
            logits,
            indices,
            seq_lens,
            next_n,
            stride0,
            stride1,
            indices.stride(0),
            indices.stride(1),
            vocab_size,
            hist_base_ptr,
            s_final_cnt_ptr,
            s_threshold_bin_idx_ptr,
            s_final_bin_size_ptr,
            s_found_topk_values_ptr,
            s_step_thresholds_ptr,
            s_out_indices_ptr,
            s_radix_count_ptr,
            TOPK=top_k,
            BLOCK_SIZE=TLE_FIXED_BLOCK_SIZE,
            USE_RADIX_FINAL=use_radix_final,
            HIST_SIZE=HIST_SIZE,
            RADIX_SIZE_FINAL=RADIX_SIZE_FINAL,
            num_warps=TLE_FIXED_BLOCK_SIZE // 32,
            num_stages=TLE_FIXED_NUM_STAGES,
        )


def top_k_per_row_prefill(
    logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
):
    """Top-K per row for prefill phase of DeepSeek V4 sparse attention.

    Masks invalid ranges in-place, then selects top-K indices per row.
    Output indices are 0-based relative to row_starts[i].

    Args:
        logits: [num_rows, vocab_size] float32 tensor, modified in-place (masked to -inf).
                In DeepSeek V4: vocab_size=129280.
        row_starts: [num_rows] int32 — start of valid range per row (inclusive).
        row_ends: [num_rows] int32 — end of valid range per row (exclusive).
        indices: [num_rows, top_k] int32 — output buffer, filled with 0-based indices
                 relative to row_starts[i]. Caller pre-allocates this.
        num_rows: number of rows (1 for decode, 32/64/2048 for prefill batches).
        stride0: logits.stride(0), typically == vocab_size for contiguous tensor.
        stride1: logits.stride(1), typically == 1 for contiguous tensor.
        top_k: number of top elements per row (1024 in DeepSeek V4).
    """
    vocab_size = logits.shape[1]

    if top_k > vocab_size:
        raise ValueError(f"top_k ({top_k}) must not exceed vocab_size ({vocab_size})")
    assert num_rows == logits.shape[0]

    vocab_size = logits.shape[1]
    use_radix_final = vocab_size >= TLE_RADIX_FINAL_SEQ_LEN_THRESHOLD
    n_blocks = (vocab_size + CLUSTER_BLOCK_SIZE - 1) // CLUSTER_BLOCK_SIZE
    if (
        num_rows == 1
        and vocab_size >= CLUSTER_SEQ_LEN_THRESHOLD
        and n_blocks <= MAX_CLUSTER_N
    ):
        device = logits.device
        pb_size = n_blocks * 256
        pb_hist = torch.empty(num_rows * pb_size, dtype=torch.int32, device=device)
        sync = torch.zeros(num_rows * 4, dtype=torch.int32, device=device)
        counter = torch.zeros(num_rows * 2, dtype=torch.int32, device=device)
        top_k_pad = triton.next_power_of_2(top_k)
        top_k_per_row_prefill_cluster[(num_rows * n_blocks,)](
            logits,
            row_starts,
            row_ends,
            indices,
            pb_hist,
            sync,
            counter,
            stride0,
            stride1,
            vocab_size,
            NUM_BLOCKS=n_blocks,
            BLOCK=CLUSTER_BLOCK_SIZE,
            TOP_K=top_k,
            TOP_KP=top_k_pad,
            num_warps=8,
        )
    elif HAS_TLE:
        topkp = triton.next_power_of_2(top_k)
        tle_top_k_per_row_prefill_wrapper[(num_rows,)](
            logits,
            indices,
            row_starts,
            row_ends,
            stride0,
            stride1,
            indices.stride(0),
            indices.stride(1),
            vocab_size,
            TOPK=top_k,
            TOPKP=topkp,
            BLOCK_SIZE=TLE_FIXED_BLOCK_SIZE,
            USE_RADIX_FINAL=use_radix_final,
            num_warps=TLE_FIXED_BLOCK_SIZE // 32,
            num_stages=TLE_FIXED_NUM_STAGES,
        )
    else:
        # based on tle version
        device = logits.device
        hist_base_ptr = torch.empty(
            (num_rows, HIST_SIZE), device=device, dtype=torch.int32
        )
        s_final_cnt_ptr = torch.empty((num_rows,), device=device, dtype=torch.int32)
        s_threshold_bin_idx_ptr = torch.empty(
            (num_rows,), device=device, dtype=torch.int32
        )
        s_final_bin_size_ptr = torch.empty(
            (num_rows,), device=device, dtype=torch.int32
        )
        s_found_topk_values_ptr = torch.empty(
            (num_rows,), device=device, dtype=torch.int32
        )
        s_step_thresholds_ptr = torch.empty(
            (num_rows, 4), device=device, dtype=torch.int32
        )
        s_out_indices_ptr = torch.empty(
            (num_rows, top_k), device=device, dtype=torch.int32
        )
        s_radix_count_ptr = (
            torch.empty((num_rows, RADIX_SIZE_FINAL), device=device, dtype=torch.int32)
            if use_radix_final
            else None
        )
        top_k_per_row_prefill_wrapper[(num_rows,)](
            logits,
            indices,
            row_starts,
            row_ends,
            stride0,
            stride1,
            indices.stride(0),
            indices.stride(1),
            vocab_size,
            hist_base_ptr,
            s_final_cnt_ptr,
            s_threshold_bin_idx_ptr,
            s_final_bin_size_ptr,
            s_found_topk_values_ptr,
            s_step_thresholds_ptr,
            s_out_indices_ptr,
            s_radix_count_ptr,
            TOPK=top_k,
            BLOCK_SIZE=TLE_FIXED_BLOCK_SIZE,
            USE_RADIX_FINAL=use_radix_final,
            HIST_SIZE=HIST_SIZE,
            RADIX_SIZE_FINAL=RADIX_SIZE_FINAL,
            num_warps=TLE_FIXED_BLOCK_SIZE // 32,
            num_stages=TLE_FIXED_NUM_STAGES,
        )

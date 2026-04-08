import torch
import triton
import triton.language as tl


def _triton_version_at_least(major: int, minor: int, patch: int = 0) -> bool:
    version = str(getattr(triton, "__version__", "0.0.0")).split("+", 1)[0]
    parts = version.split(".")
    parsed = []
    for part in parts[:3]:
        digits = []
        for ch in part:
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        parsed.append(int("".join(digits)) if digits else 0)
    while len(parsed) < 3:
        parsed.append(0)
    return tuple(parsed) >= (major, minor, patch)


if _triton_version_at_least(3, 6, 0):
    try:
        import triton.experimental.tle.language as tle

        HAS_TLE = True
    except ImportError:
        tle = None
        HAS_TLE = False
else:
    tle = None
    HAS_TLE = False


TLE_FIXED_BLOCK_SIZE = 1024
TLE_FIXED_NUM_WARPS = TLE_FIXED_BLOCK_SIZE // 32
TLE_FIXED_NUM_STAGES = 1
TLE_RADIX_FINAL_SEQ_LEN_THRESHOLD = 12288


@triton.jit
def convert_to_uint16(x):
    bits_uint = convert_to_uint32(x)
    return ((bits_uint >> 24) & 0xFF).to(tl.uint16)


@triton.jit
def convert_to_uint32(x):
    bits_uint = x.to(tl.uint32, bitcast=True)
    bits_uint = tl.where(
        x < 0,
        ~bits_uint & tl.full(bits_uint.shape, 0xFFFFFFFF, tl.uint32),
        bits_uint | tl.full(bits_uint.shape, 0x80000000, tl.uint32),
    )
    return bits_uint


@triton.autotune(
    configs=[
        triton.Config({"BS": 32, "BSS": 32}, num_stages=1, num_warps=1),
        triton.Config({"BS": 64, "BSS": 32}, num_stages=1, num_warps=1),
        triton.Config({"BS": 128, "BSS": 32}, num_stages=2, num_warps=1),
        triton.Config({"BS": 256, "BSS": 32}, num_stages=2, num_warps=2),
        triton.Config({"BS": 512, "BSS": 64}, num_stages=2, num_warps=2),
        triton.Config({"BS": 1024, "BSS": 256}, num_stages=2, num_warps=2),
        triton.Config({"BS": 2048, "BSS": 256}, num_stages=2, num_warps=4),
        triton.Config({"BS": 4096, "BSS": 512}, num_stages=3, num_warps=4),
        triton.Config({"BS": 8192, "BSS": 512}, num_stages=3, num_warps=8),
        triton.Config({"BS": 8192, "BSS": 1024}, num_stages=3, num_warps=8),
    ],
    key=["S", "K"],
)
@triton.jit
def kernel_bucket_sort_topk(  # grid(B, BS)
    inputs,  # (B, S) Note: no H because MLA is based on MQA and MHA, not GQA
    indices,  # (B, K) topk index array
    s_input_ids,  # Data indices to be filtered in the next round
    starts,  # for variable length
    ends,  # for variable length
    S: tl.constexpr,  # sequence length
    K: tl.constexpr,  # k of topk
    HISTOGRAM_SIZE: tl.constexpr,
    SMEM_INPUT_SIZE: tl.constexpr,  # to save candidates of next loop
    BS: tl.constexpr,  # block size of S
    BSS: tl.constexpr,  # block size of SMEM_INPUT
):
    # Get thread block id
    i_b = tl.program_id(0)

    # Block base pointer definitions
    s_base = inputs + i_b * S
    indices_base = indices + i_b * K
    s_input_ids_base = s_input_ids + i_b * SMEM_INPUT_SIZE

    # Histogram initialization
    s_histogram = tl.zeros([HISTOGRAM_SIZE], dtype=tl.int32)

    # Support variable length
    l_start_idx = tl.load(starts + i_b).to(tl.int32)
    l_end_idx = tl.load(ends + i_b).to(tl.int32)

    # Record how many positions remain to fill the topk array
    l_new_topk = K

    TS = tl.cdiv(S, BS)
    for s in range(TS):
        input_idx = s * BS + tl.arange(0, BS)
        input_mask = (
            (input_idx < l_end_idx) & (input_idx >= l_start_idx) & (input_idx < S)
        )
        input = tl.load(s_base + input_idx, input_mask, other=float("-inf")).to(
            tl.float32
        )
        inval_int16 = convert_to_uint16(input)
        s_histogram += inval_int16.to(tl.int32).histogram(HISTOGRAM_SIZE)

    s_histogram = s_histogram.cumsum(0, reverse=True)  # Suffix sum

    mv_idx = (
        tl.arange(1, HISTOGRAM_SIZE + 1) % HISTOGRAM_SIZE
    )  # Construct offset index matrix

    cond = (s_histogram > l_new_topk) & (
        (s_histogram.gather(mv_idx, 0) <= l_new_topk) | (mv_idx == 0)
    )
    l_threshold_bin_id = cond.argmax(0)

    l_new_topk -= tl.where(
        tl.arange(0, HISTOGRAM_SIZE) == l_threshold_bin_id + 1, s_histogram, 0
    ).max(0)
    sum = 0
    thre_bin_sum = 0
    for s in range(TS):
        input_idx = s * BS + tl.arange(0, BS)
        input_mask = (
            (input_idx < l_end_idx) & (input_idx >= l_start_idx) & (input_idx < S)
        )
        input = tl.load(s_base + input_idx, input_mask, other=float("-inf")).to(
            tl.float32
        )
        inval_int16 = convert_to_uint16(input)
        # inval_int16 = tl.where(input_mask, inval_int16, 0)
        # This method would slow down the speed, so using other=float("-inf") saves time.

        over_thre = inval_int16.to(tl.int32) > l_threshold_bin_id
        cur_sum = over_thre.to(tl.int32).sum(-1)

        eq_thre = inval_int16.to(tl.int32) == l_threshold_bin_id
        thre_bin_cur_sum = eq_thre.to(tl.int32).sum(-1)

        topk_idx = over_thre.to(tl.int32).cumsum(-1)
        thre_bin_idx = eq_thre.to(tl.int32).cumsum(-1)

        concat_mask = tl.cat(over_thre, eq_thre, True)
        concat_input = tl.cat(input_idx, input_idx, True)
        concat_pointer_matrix = tl.cat(
            indices_base + sum + topk_idx - 1,
            s_input_ids_base + thre_bin_sum + thre_bin_idx - 1,
            True,
        )
        tl.store(concat_pointer_matrix, concat_input, mask=concat_mask)

        thre_bin_sum += thre_bin_cur_sum
        sum += cur_sum

    round = 0
    # print("l_new_topk:", l_new_topk)
    while round < 4 and l_new_topk > 0:
        ss = tl.cdiv(thre_bin_sum, BSS)
        s_histogram = tl.zeros([HISTOGRAM_SIZE], dtype=tl.int32)
        padding_num = 0.0 if round else float("-inf")
        # When round == 0, if the padding value is set to 0.0, the following problem occurs:
        #
        # 0.0 = 0x00000000, inval_int32(0x|00|000000, round=0) = 0x80
        # This causes the padding bucket to be larger than negative candidates,
        #  thus being prioritized and assigned to the next bucket
        #  or even directly into the topk sequence.
        #
        # However, if the padding value is set to "-inf":
        # float("-inf") = 0xFFFFE000, inval_int32(0x|FF|FFE000, round=0) = 0x00
        # This ensures the padding value is placed in the smallest bin,
        #  not affecting the sorting of all normal candidate numbers before it.
        #
        # But when round > 0, if the padding value remains "-inf", the following problem occurs:
        # float("-inf") = 0xFFFFE000, inval_int32(0xFFFFE0|00|, round=3) = 0xFF
        # This causes the padding bucket to be larger than all values,
        # thus preferentially entering the topk sequence and causing errors.
        # Therefore, the padding value should be set to 0.0
        for s in range(ss):
            s_input_idx = s * BSS + tl.arange(0, BSS)
            s_input_idx_mask = s_input_idx < thre_bin_sum
            input_idx = tl.load(
                s_input_ids_base + s_input_idx, s_input_idx_mask, other=-1
            )
            s_input_mask = s_input_idx_mask
            s_input = tl.load(s_base + input_idx, s_input_mask, other=padding_num).to(
                tl.float32
            )
            inval_int32 = (
                convert_to_uint32(s_input) >> (24 - round * 8)
            ) & 0xFF  # Ensure all bits except the last eight are zero
            s_histogram += inval_int32.to(tl.int32).histogram(HISTOGRAM_SIZE)
        s_histogram = s_histogram.cumsum(0, reverse=True)  # Suffix sum
        mv_idx = (
            tl.arange(1, HISTOGRAM_SIZE + 1) % HISTOGRAM_SIZE
        )  # Construct offset index matrix
        cond = (s_histogram > l_new_topk) & (
            (s_histogram.gather(mv_idx, 0) <= l_new_topk) | (mv_idx == 0)
        )
        l_threshold_bin_id = cond.argmax(0)
        l_new_topk -= tl.where(
            tl.arange(0, HISTOGRAM_SIZE) == l_threshold_bin_id + 1, s_histogram, 0
        ).max(0)
        thre_bin_sum, old_thre_bin_sum = 0, thre_bin_sum

        for s in range(ss):
            s_input_idx = s * BSS + tl.arange(0, BSS)
            s_input_idx_mask = s_input_idx < old_thre_bin_sum
            input_idx = tl.load(
                s_input_ids_base + s_input_idx, s_input_idx_mask, other=-1
            )
            s_input_mask = s_input_idx_mask
            s_input = tl.load(s_base + input_idx, s_input_mask, other=padding_num).to(
                tl.float32
            )
            inval_int32 = (convert_to_uint32(s_input) >> (24 - round * 8)) & 0xFF

            over_thre = inval_int32.to(tl.int32) > l_threshold_bin_id
            cur_sum = over_thre.to(tl.int32).sum(-1)
            eq_thre = inval_int32.to(tl.int32) == l_threshold_bin_id
            thre_bin_cur_sum = eq_thre.to(tl.int32).sum(-1)

            topk_idx = over_thre.to(tl.int32).cumsum(-1)
            thre_bin_idx = eq_thre.to(tl.int32).cumsum(-1)

            concat_mask = tl.cat(over_thre, eq_thre, True)
            concat_input = tl.cat(input_idx, input_idx, True)
            concat_pointer_matrix = tl.cat(
                indices_base + sum + topk_idx - 1,
                s_input_ids_base + thre_bin_sum + thre_bin_idx - 1,
                True,
            )

            tl.store(concat_pointer_matrix, concat_input, mask=concat_mask)

            thre_bin_sum += thre_bin_cur_sum
            sum += cur_sum

        round += 1

    if l_new_topk > 0:
        ss = tl.cdiv(l_new_topk, BSS)
        for s in range(ss):
            s_input_idx = s * BSS + tl.arange(0, BSS)
            s_input_idx_mask = s_input_idx < l_new_topk
            input_idx = tl.load(
                s_input_ids_base + s_input_idx, s_input_idx_mask, other=-1
            )
            s_input_mask = s_input_idx_mask
            tl.store(
                indices_base + sum + tl.arange(0, BSS), input_idx, mask=s_input_mask
            )
            sum += BSS


def bucket_sort_topk_triton(inputs, starts, ends, topk):
    B, S = inputs.shape
    K = topk
    HISTOGRAM_SIZE = 256
    SMEM_INPUT_SIZE = 4096
    indices = torch.full((B, topk), -1, dtype=torch.int32, device=inputs.device)
    s_input_idx = torch.zeros(
        B, SMEM_INPUT_SIZE, dtype=torch.int32, device=inputs.device
    )
    grid = (B,)
    kernel_bucket_sort_topk[grid](
        inputs,
        indices,
        s_input_idx,
        starts,
        ends,
        S,
        K,
        HISTOGRAM_SIZE,
        SMEM_INPUT_SIZE,
    )
    return indices


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
def _tle_process_histogram_step(
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
):
    VEC: tl.constexpr = 4
    FINAL_SORT_ITEMS: tl.constexpr = 2048
    RADIX11_SIZE: tl.constexpr = 2048
    RADIX11_MASK: tl.constexpr = 0x7FF
    RADIX10_SIZE: tl.constexpr = 1024
    RADIX10_MASK: tl.constexpr = 0x3FF

    lane = tl.arange(0, BLOCK_SIZE)
    vec = tl.arange(0, VEC)
    ones = tl.full([BLOCK_SIZE], 1, tl.int32)
    ones_vec_2d = tl.full([BLOCK_SIZE, VEC], 1, tl.int32)
    zeros = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    zeros_vec_2d = tl.zeros([BLOCK_SIZE, VEC], dtype=tl.int32)

    clear_rounds = tl.where(
        step_idx == 3,
        RADIX10_SIZE // BLOCK_SIZE,
        RADIX11_SIZE // BLOCK_SIZE,
    )
    for clear_round in tl.range(0, clear_rounds):
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
            key = _convert_to_trt_uint32(x_vec)
            if step_idx == 0:
                digit = _convert_to_trt_uint16_hi11(x_vec)
            elif step_idx == 1:
                digit = ((key >> 21) & RADIX11_MASK).to(tl.int32)
            elif step_idx == 2:
                digit = ((key >> 10) & RADIX11_MASK).to(tl.int32)
            else:
                digit = (key & RADIX10_MASK).to(tl.int32)

            if step_idx < 2:
                partial = tl.full([BLOCK_SIZE, VEC], True, tl.int1)
            elif step_idx == 2:
                partial = ((key ^ logit_pattern) >> 21) == 0
            else:
                partial = ((key ^ logit_pattern) >> 10) == 0

            tl.atomic_add(
                hist_base_ptr + digit,
                ones_vec_2d,
                mask=partial,
                sem="relaxed",
                scope="cta",
            )

        for t in tl.range(0, rem_tiles):
            offs = (n_vec_full * VEC + t) * BLOCK_SIZE + lane
            x = tl.load(row_ptr + offs)
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
                partial = tl.full([BLOCK_SIZE], True, tl.int1)
            elif step_idx == 2:
                partial = ((key ^ logit_pattern) >> 21) == 0
            else:
                partial = ((key ^ logit_pattern) >> 10) == 0

            tl.atomic_add(
                hist_base_ptr + digit,
                ones,
                mask=partial,
                sem="relaxed",
                scope="cta",
            )
    else:
        for t in tl.range(0, n_tiles):
            offs = t * BLOCK_SIZE + lane
            in_range = (offs < seq_len) & (offs >= row_start) & (offs < row_end)
            x = tl.load(row_ptr + offs * stride_xn, mask=in_range, other=float("-inf"))
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
    tl.debug_barrier()

    tl.store(s_threshold_bin_idx_ptr, -1)
    tl.store(s_final_bin_size_ptr, 0)
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
            prefix_sum, counts_total = tle.cumsum(counts, axis=0, reverse=False)
            prefix_sum = prefix_sum + last_value
            total_sum = last_value + counts_total
            next_prefix_sum = prefix_sum + counts
            threshold_mask = (prefix_sum < TOPK) & (next_prefix_sum >= TOPK)
            threshold_bin = bins
            threshold_bin_size = next_prefix_sum - prefix_sum
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
    if use_final:
        tl.store(s_final_cnt_ptr, 0)

    found_ptrs = s_found_topk_values_ptr + zeros
    final_cnt_ptrs = s_final_cnt_ptr + zeros
    if assume_aligned:
        found_ptrs_vec_2d = s_found_topk_values_ptr + zeros_vec_2d
        final_cnt_ptrs_vec_2d = s_final_cnt_ptr + zeros_vec_2d
        for t in tl.range(0, n_vec_full):
            base = t * BLOCK_SIZE * VEC + lane * VEC
            offs = base[:, None] + vec[None, :]
            x_vec = tl.load(row_ptr + offs)
            key = _convert_to_trt_uint32(x_vec)
            if step_idx == 0:
                digit = _convert_to_trt_uint16_hi11(x_vec)
            elif step_idx == 1:
                digit = ((key >> 21) & RADIX11_MASK).to(tl.int32)
            elif step_idx == 2:
                digit = ((key >> 10) & RADIX11_MASK).to(tl.int32)
            else:
                digit = (key & RADIX10_MASK).to(tl.int32)

            if step_idx < 2:
                partial = tl.full([BLOCK_SIZE, VEC], True, tl.int1)
            elif step_idx == 2:
                partial = ((key ^ logit_pattern) >> 21) == 0
            else:
                partial = ((key ^ logit_pattern) >> 10) == 0

            take_lt = partial & (digit < threshold_bin_idx)
            out_pos_lt = tl.atomic_add(
                found_ptrs_vec_2d,
                ones_vec_2d,
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
                    ones_vec_2d,
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
                    final_cnt_ptrs_vec_2d,
                    ones_vec_2d,
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
                    x_vec.to(tl.int32, bitcast=True),
                    mask=take_eq_final & (final_pos < FINAL_SORT_ITEMS),
                )

        for t in tl.range(0, rem_tiles):
            offs = (n_vec_full * VEC + t) * BLOCK_SIZE + lane
            x = tl.load(row_ptr + offs)
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
                partial = tl.full([BLOCK_SIZE], True, tl.int1)
            elif step_idx == 2:
                partial = ((key ^ logit_pattern) >> 21) == 0
            else:
                partial = ((key ^ logit_pattern) >> 10) == 0

            take_lt = partial & (digit < threshold_bin_idx)
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
    else:
        for t in tl.range(0, n_tiles):
            offs = t * BLOCK_SIZE + lane
            in_range = (offs < seq_len) & (offs >= row_start) & (offs < row_end)
            x = tl.load(row_ptr + offs * stride_xn, mask=in_range, other=float("-inf"))
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

            take_lt = partial & (digit < threshold_bin_idx)
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

    if step_idx < 3:
        if use_final:
            need_final_sort = True
            continue_to_next_step = False
        else:
            need_final_sort = False
            continue_to_next_step = True
    else:
        tl.store(s_found_topk_values_ptr, TOPK)
        need_final_sort = False
        continue_to_next_step = False

    tl.debug_barrier()
    return continue_to_next_step, need_final_sort, logit_pattern


@triton.jit
def _tle_final_select_radix(
    hist_base_ptr,
    s_out_indices_ptr,
    s_final_cnt_ptr,
    s_found_topk_values_ptr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    FINAL_SORT_ITEMS: tl.constexpr,
):
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
    radix_count_ptr = tle.gpu.local_ptr(s_radix_counts, (0,))
    radix_count_vec_ptr = tle.gpu.local_ptr(s_radix_counts, (bins,))

    base_idx = tl.load(s_found_topk_values_ptr)
    final_cnt = tl.minimum(tl.load(s_final_cnt_ptr), FINAL_SORT_ITEMS)
    remain = tl.minimum(TOPK - base_idx, final_cnt)
    if remain > 0:
        desired = tl.zeros((), dtype=tl.uint32)
        desired_mask = tl.zeros((), dtype=tl.uint32)
        k_to_find = remain + 1

        for digit_pos in tl.static_range(DIGIT_START, -1, -RADIX_BITS_FINAL):
            tl.store(radix_count_ptr + lane, 0, mask=lane < RADIX_SIZE_FINAL)
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
                    radix_count_ptr + digit,
                    ones,
                    mask=take,
                    sem="relaxed",
                    scope="cta",
                )

            tl.debug_barrier()
            counts = tl.load(radix_count_vec_ptr)
            prefix_sum, _ = tle.cumsum(counts, axis=0, reverse=False)
            next_prefix_sum = prefix_sum + counts
            threshold_mask = (prefix_sum < k_to_find) & (next_prefix_sum >= k_to_find)
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
                tl.where(bins == threshold_bin, prefix_sum, 0),
                axis=0,
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

    tl.store(s_found_topk_values_ptr, TOPK)


@triton.jit
def kernel_tle_bucket_sort_topk(
    x_ptr,
    out_ptr,
    starts_ptr,
    ends_ptr,
    stride_xm,
    stride_xn,
    stride_outm,
    stride_outn,
    seq_len,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_RADIX_FINAL: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = tl.load(starts_ptr + pid).to(tl.int32)
    row_end = tl.load(ends_ptr + pid).to(tl.int32)

    row_ptr = x_ptr + pid * stride_xm
    out_row = out_ptr + pid * stride_outm
    row_len = row_end - row_start

    auto_aligned = (
        (stride_xn == 1)
        & (stride_outn == 1)
        & (row_start == 0)
        & (row_end == seq_len)
        & (seq_len % BLOCK_SIZE == 0)
    )
    assume_aligned = auto_aligned
    if assume_aligned:
        seq_len = tl.multiple_of(seq_len, BLOCK_SIZE)

    lane = tl.arange(0, BLOCK_SIZE)
    if row_len <= K:
        chunks: tl.constexpr = (K + BLOCK_SIZE - 1) // BLOCK_SIZE
        for chunk_idx in tl.range(0, chunks):
            pos = chunk_idx * BLOCK_SIZE + lane
            take_row = pos < row_len
            tl.store(
                out_row + pos * stride_outn,
                (row_start + pos).to(tl.int32),
                mask=take_row,
            )
            take_pad = (pos >= row_len) & (pos < K)
            tl.store(out_row + pos * stride_outn, -1, mask=take_pad)
        return

    FINAL_SORT_ITEMS: tl.constexpr = 2048
    HIST_SIZE: tl.constexpr = 4096

    s_histogram = tle.gpu.alloc(
        [HIST_SIZE],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    hist_base_ptr = tle.gpu.local_ptr(s_histogram, (0,))
    s_out_indices = tle.gpu.alloc(
        [K],
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
    s_final_cnt_ptr = tle.gpu.local_ptr(s_final_cnt, (0,))
    s_threshold_bin_idx_ptr = tle.gpu.local_ptr(s_threshold_bin_idx, (0,))
    s_final_bin_size_ptr = tle.gpu.local_ptr(s_final_bin_size, (0,))
    s_found_topk_values_ptr = tle.gpu.local_ptr(s_found_topk_values, (0,))
    s_step_thresholds_ptr = tle.gpu.local_ptr(s_step_thresholds, (0,))
    s_out_indices_ptr = tle.gpu.local_ptr(s_out_indices, (0,))
    tl.store(s_final_cnt_ptr, 0)
    tl.store(s_threshold_bin_idx_ptr, -1)
    tl.store(s_final_bin_size_ptr, 0)
    tl.store(s_found_topk_values_ptr, 0)

    logit_pattern = tl.zeros((), dtype=tl.uint32)
    continue_to_next_step = True
    need_final_sort = False
    init_chunks: tl.constexpr = (K + BLOCK_SIZE - 1) // BLOCK_SIZE
    for init_idx in tl.range(0, init_chunks):
        pos = init_idx * BLOCK_SIZE + lane
        tl.store(tle.gpu.local_ptr(s_out_indices, (pos,)), -1, mask=pos < K)

    for step_idx in tl.static_range(0, 4):
        if continue_to_next_step:
            found_topk_values = tl.load(s_found_topk_values_ptr)
            (
                continue_to_next_step,
                step_need_final_sort,
                logit_pattern,
            ) = _tle_process_histogram_step(
                row_ptr,
                stride_xn,
                row_start,
                row_end,
                seq_len,
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
                assume_aligned,
                TOPK=K,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            need_final_sort = need_final_sort | step_need_final_sort

    if need_final_sort:
        if USE_RADIX_FINAL:
            _tle_final_select_radix(
                hist_base_ptr,
                s_out_indices_ptr,
                s_final_cnt_ptr,
                s_found_topk_values_ptr,
                TOPK=K,
                BLOCK_SIZE=BLOCK_SIZE,
                FINAL_SORT_ITEMS=FINAL_SORT_ITEMS,
            )
        else:
            base_idx = tl.load(s_found_topk_values_ptr)
            final_cnt = tl.minimum(tl.load(s_final_cnt_ptr), FINAL_SORT_ITEMS)
            sort_chunks = tl.cdiv(final_cnt, BLOCK_SIZE)
            for sort_chunk in tl.range(0, sort_chunks):
                pos = sort_chunk * BLOCK_SIZE + lane
                valid = pos < final_cnt
                logit_i_bits = tl.load(
                    tle.gpu.local_ptr(s_histogram, (FINAL_SORT_ITEMS + pos,)),
                    mask=valid,
                    other=0,
                )
                logit_i = logit_i_bits.to(tl.float32, bitcast=True)
                out_rank = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
                for j in tl.range(0, final_cnt):
                    logit_j_bits = tl.load(
                        tle.gpu.local_ptr(s_histogram, (FINAL_SORT_ITEMS + j,))
                    )
                    logit_j = logit_j_bits.to(tl.float32, bitcast=True)
                    better = (logit_i < logit_j) | ((logit_i == logit_j) & (pos < j))
                    out_rank = out_rank + (valid & better).to(tl.int32)
                dst_pos = base_idx + out_rank
                take = valid & (dst_pos < K)
                idx_i = tl.load(
                    tle.gpu.local_ptr(s_histogram, (pos,)),
                    mask=take,
                    other=0,
                )
                tl.store(tle.gpu.local_ptr(s_out_indices, (dst_pos,)), idx_i, mask=take)
            tl.store(s_found_topk_values_ptr, K)

    flush_chunks: tl.constexpr = (K + BLOCK_SIZE - 1) // BLOCK_SIZE
    for flush_chunk in tl.static_range(flush_chunks):
        pos = flush_chunk * BLOCK_SIZE + lane
        mask = pos < K
        out_vals = tl.load(
            tle.gpu.local_ptr(s_out_indices, (pos,)), mask=mask, other=-1
        )
        tl.store(out_row + pos * stride_outn, out_vals, mask=mask)


def tle_bucket_sort_topk(
    inputs,
    starts,
    ends,
    topk,
):
    if not HAS_TLE:
        raise RuntimeError(
            "TLE is unavailable. bucket_sort_topk TLE kernel requires Triton >= 3.6 with triton.experimental.tle."
        )
    if inputs.ndim != 2:
        raise ValueError("inputs must be a 2D tensor")
    if starts.ndim != 1 or ends.ndim != 1:
        raise ValueError("starts and ends must be 1D tensors")

    x = inputs.float() if inputs.dtype != torch.float32 else inputs
    batch, seq_len = x.shape
    out = torch.full((batch, topk), -1, dtype=torch.int32, device=x.device)
    use_radix_final = seq_len >= TLE_RADIX_FINAL_SEQ_LEN_THRESHOLD

    grid = (batch,)
    kernel_tle_bucket_sort_topk[grid](
        x,
        out,
        starts,
        ends,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        seq_len,
        K=topk,
        BLOCK_SIZE=TLE_FIXED_BLOCK_SIZE,
        USE_RADIX_FINAL=use_radix_final,
        num_warps=TLE_FIXED_NUM_WARPS,
        num_stages=TLE_FIXED_NUM_STAGES,
    )
    return out


def _should_use_tle_bucket_sort_topk(inputs, topk):
    if not HAS_TLE:
        return False
    if not isinstance(inputs, torch.Tensor) or inputs.device.type != "cuda":
        return False
    return True


def bucket_sort_topk(inputs, starts, ends, topk):
    if _should_use_tle_bucket_sort_topk(inputs, topk):
        try:
            return tle_bucket_sort_topk(inputs, starts, ends, topk)
        except Exception:
            # Fallback to legacy implementation when TLE path is unsupported at runtime.
            return bucket_sort_topk_triton(inputs, starts, ends, topk)
    return bucket_sort_topk_triton(inputs, starts, ends, topk)

import torch
import triton
import triton.language as tl


@triton.jit
def convert_to_uint16(x):
    hval = x.cast(dtype=tl.float16)
    bits_uint = hval.cast(dtype=tl.uint16, bitcast=True)  # Equivalent to reinterpret
    bits_uint = tl.where(x < 0, ~bits_uint & (0xFFFF), bits_uint | (0x8000))
    return bits_uint >> 8


@triton.jit
def convert_to_uint32(x):
    bits_uint = x.cast(dtype=tl.uint32, bitcast=True)
    bits_uint = tl.where(
        x < 0,
        ~bits_uint & tl.cast((0xFFFFFFFF), tl.uint32, bitcast=True),
        bits_uint | tl.cast((0x80000000), tl.uint32, bitcast=True),
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


def bucket_sort_topk(inputs, starts, ends, topk):
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

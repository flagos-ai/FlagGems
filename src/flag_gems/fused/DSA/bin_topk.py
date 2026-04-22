import torch
import triton
import triton.language as tl

from flag_gems.utils.triton_version_utils import HAS_TLE

if HAS_TLE:
    import triton.experimental.tle.language as tle_gpu
else:
    tle_gpu = None


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


if HAS_TLE:

    @triton.jit
    def tle_topk_selector_kernel(
        x_ptr,
        out_ptr,
        starts_ptr,
        ends_ptr,
        stride_xm,
        stride_xn,
        stride_outm,
        stride_outn,
        seq_len,
        RADIX: tl.constexpr,
        HIST_SIZE: tl.constexpr,
        ASSUME_ALIGNED: tl.constexpr,
        TOPK: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        N_TILES: tl.constexpr,
        SMEM_INPUT: tl.constexpr,
        NUM_INPUT_TILES: tl.constexpr,
    ):
        pid = tl.program_id(0)
        row_start = tl.load(starts_ptr + pid).to(tl.int32)
        row_end = tl.load(ends_ptr + pid).to(tl.int32)

        row_ptr = x_ptr + pid * stride_xm
        out_row = out_ptr + pid * stride_outm

        if ASSUME_ALIGNED:
            tl.assume(row_start == 0)
            tl.assume(row_end == seq_len)
            tl.assume(stride_xn == 1)
            tl.assume(stride_outn == 1)
            seq_len = tl.multiple_of(seq_len, BLOCK_SIZE)

        lane = tl.arange(0, BLOCK_SIZE)
        ones = tl.full([BLOCK_SIZE], 1, tl.int32)

        s_histogram = tle_gpu.gpu.alloc(
            [HIST_SIZE],
            dtype=tl.int32,
            layout=None,
            scope=tle_gpu.gpu.smem,
            nv_mma_shared_layout=False,
        )
        s_num_input = tle_gpu.gpu.alloc(
            [2],
            dtype=tl.int32,
            layout=None,
            scope=tle_gpu.gpu.smem,
            nv_mma_shared_layout=False,
        )
        s_input_idx = tle_gpu.gpu.alloc(
            [2, SMEM_INPUT],
            dtype=tl.int32,
            layout=None,
            scope=tle_gpu.gpu.smem,
            nv_mma_shared_layout=False,
        )

        hist_idx = tl.arange(0, RADIX)
        hist_last = tl.full([1], RADIX, tl.int32)

        hist_ptrs = tle_gpu.gpu.local_ptr(s_histogram, (hist_idx,))
        hist_last_ptrs = tle_gpu.gpu.local_ptr(s_histogram, (hist_last,))
        tl.store(hist_ptrs, 0)
        tl.store(hist_last_ptrs, 0)
        tl.store(tle_gpu.gpu.local_ptr(s_num_input, (tl.arange(0, 2),)), 0)
        tl.debug_barrier()

        l_new_topk = tl.full((), TOPK, tl.int32)

        # Stage 1: coarse 8-bit histogram with uint16 conversion
        for t in tl.static_range(N_TILES):
            offs = t * BLOCK_SIZE + lane
            in_range = (offs < seq_len) & (offs >= row_start) & (offs < row_end)
            x = tl.load(row_ptr + offs * stride_xn, mask=in_range, other=0.0)
            bin_u16 = convert_to_uint16(x)
            bin_i32 = bin_u16.to(tl.int32)
            hist_bin_ptrs = tle_gpu.gpu.local_ptr(s_histogram, (bin_i32,))
            tl.atomic_add(hist_bin_ptrs, ones, mask=in_range)

        rev_idx = (RADIX - 1) - hist_idx
        hist_rev = tl.load(tle_gpu.gpu.local_ptr(s_histogram, (rev_idx,)))
        hist_cum_rev = tl.cumsum(hist_rev, axis=0)
        tl.store(tle_gpu.gpu.local_ptr(s_histogram, (rev_idx,)), hist_cum_rev)
        tl.debug_barrier()

        hist_cum = tl.load(hist_ptrs)
        hist_cum_next = tl.load(
            tle_gpu.gpu.local_ptr(s_histogram, (hist_idx + 1,)),
            mask=hist_idx + 1 < RADIX,
            other=0,
        )
        cond = (hist_cum > l_new_topk) & (hist_cum_next <= l_new_topk)
        cand = tl.where(cond, hist_idx.to(tl.int32), -1)
        threshold = tl.max(cand, axis=0)
        hist_next = tl.max(tl.where(hist_idx == threshold + 1, hist_cum, 0), axis=0)
        l_new_topk = tl.maximum(l_new_topk - hist_next, 0)

        num_ptrs = tle_gpu.gpu.local_ptr(
            s_num_input, (tl.zeros([BLOCK_SIZE], tl.int32),)
        )

        # Compact: > threshold → direct output, == threshold → s_input_idx
        for t in tl.static_range(N_TILES):
            offs = t * BLOCK_SIZE + lane
            in_range = (offs < seq_len) & (offs >= row_start) & (offs < row_end)
            x = tl.load(row_ptr + offs * stride_xn, mask=in_range, other=0.0)
            bin_u16 = convert_to_uint16(x)
            bin_i32 = bin_u16.to(tl.int32)
            gt_thr = bin_i32 > threshold
            eq_thr = bin_i32 == threshold

            pos = tl.atomic_add(
                tle_gpu.gpu.local_ptr(s_histogram, (bin_i32 + 1,)),
                ones,
                mask=in_range & gt_thr,
            )
            pos = tl.where(in_range & gt_thr, pos, 0)
            tl.store(
                out_row + pos * stride_outn,
                offs.to(tl.int32),
                mask=in_range & gt_thr & (pos < TOPK),
            )

            pos_eq = tl.atomic_add(
                num_ptrs, ones, mask=in_range & eq_thr & (l_new_topk > 0)
            )
            pos_eq = tl.where(in_range & eq_thr, pos_eq, 0)
            tl.store(
                tle_gpu.gpu.local_ptr(
                    s_input_idx, (tl.zeros([BLOCK_SIZE], tl.int32), pos_eq)
                ),
                offs.to(tl.int32),
                mask=in_range & eq_thr & (pos_eq < SMEM_INPUT) & (l_new_topk > 0),
            )

        # Stage 2: fine 32-bit refinement (4 rounds × 8 bits)
        for round_id in tl.static_range(4):
            r_idx = round_id & 1
            next_idx = r_idx ^ 1
            start_pos = TOPK - l_new_topk

            tl.store(hist_ptrs, 0)
            tl.store(hist_last_ptrs, 0)
            num_ptrs_next = tle_gpu.gpu.local_ptr(
                s_num_input, (tl.full([BLOCK_SIZE], next_idx, tl.int32),)
            )
            tl.store(num_ptrs_next, 0, mask=lane == 0)
            tl.debug_barrier()

            num_ptrs_r = tle_gpu.gpu.local_ptr(
                s_num_input, (tl.full([BLOCK_SIZE], r_idx, tl.int32),)
            )
            l_num_input = tl.max(tl.load(num_ptrs_r), axis=0).to(tl.int32)
            max_input = tl.full((), SMEM_INPUT, tl.int32)
            l_num_input = tl.minimum(l_num_input, max_input)
            active = l_new_topk > 0

            shift = 24 - round_id * 8
            for t in tl.static_range(NUM_INPUT_TILES):
                offs = t * BLOCK_SIZE + lane
                valid = offs < l_num_input
                cand_idx = tl.load(
                    tle_gpu.gpu.local_ptr(
                        s_input_idx,
                        (tl.full([BLOCK_SIZE], r_idx, tl.int32), offs),
                    ),
                    mask=valid,
                    other=0,
                )
                x = tl.load(row_ptr + cand_idx * stride_xn, mask=valid, other=0.0)
                bin_u32 = convert_to_uint32(x)
                bin_i32 = ((bin_u32 >> shift) & 0xFF).to(tl.int32)
                tl.atomic_add(
                    tle_gpu.gpu.local_ptr(s_histogram, (bin_i32,)),
                    ones,
                    mask=valid & active,
                )

            rev_idx = (RADIX - 1) - hist_idx
            hist_rev = tl.load(tle_gpu.gpu.local_ptr(s_histogram, (rev_idx,)))
            hist_cum_rev = tl.cumsum(hist_rev, axis=0)
            tl.store(tle_gpu.gpu.local_ptr(s_histogram, (rev_idx,)), hist_cum_rev)
            tl.debug_barrier()

            hist_cum = tl.load(hist_ptrs)
            hist_cum_next = tl.load(
                tle_gpu.gpu.local_ptr(s_histogram, (hist_idx + 1,)),
                mask=hist_idx + 1 < RADIX,
                other=0,
            )
            cond = (hist_cum > l_new_topk) & (hist_cum_next <= l_new_topk)
            cand = tl.where(cond, hist_idx.to(tl.int32), -1)
            threshold = tl.max(cand, axis=0)
            hist_next = tl.max(tl.where(hist_idx == threshold + 1, hist_cum, 0), axis=0)
            l_new_topk = tl.maximum(l_new_topk - hist_next, 0)

            for t in tl.static_range(NUM_INPUT_TILES):
                offs = t * BLOCK_SIZE + lane
                valid = offs < l_num_input
                cand_idx = tl.load(
                    tle_gpu.gpu.local_ptr(
                        s_input_idx,
                        (tl.full([BLOCK_SIZE], r_idx, tl.int32), offs),
                    ),
                    mask=valid,
                    other=0,
                )
                x = tl.load(row_ptr + cand_idx * stride_xn, mask=valid, other=0.0)
                bin_u32 = convert_to_uint32(x)
                bin_i32 = ((bin_u32 >> shift) & 0xFF).to(tl.int32)

                gt_thr = bin_i32 > threshold
                eq_thr = bin_i32 == threshold

                pos = tl.atomic_add(
                    tle_gpu.gpu.local_ptr(s_histogram, (bin_i32 + 1,)),
                    ones,
                    mask=valid & gt_thr & active,
                )
                pos = tl.where(valid & gt_thr & active, pos, 0)
                out_pos = pos + start_pos
                tl.store(
                    out_row + out_pos * stride_outn,
                    cand_idx,
                    mask=valid & gt_thr & active & (out_pos < TOPK),
                )

                if round_id == 3:
                    pos_eq = tl.atomic_add(
                        tle_gpu.gpu.local_ptr(s_histogram, (bin_i32 + 1,)),
                        ones,
                        mask=valid & eq_thr & active & (l_new_topk > 0),
                    )
                    pos_eq = tl.where(valid & eq_thr & active, pos_eq, 0)
                    out_pos = pos_eq + start_pos
                    tl.store(
                        out_row + out_pos * stride_outn,
                        cand_idx,
                        mask=valid
                        & eq_thr
                        & active
                        & (out_pos < TOPK)
                        & (l_new_topk > 0),
                    )
                else:
                    num_ptrs = tle_gpu.gpu.local_ptr(
                        s_num_input,
                        (tl.full([BLOCK_SIZE], next_idx, tl.int32),),
                    )
                    pos_eq = tl.atomic_add(
                        num_ptrs,
                        ones,
                        mask=valid & eq_thr & active & (l_new_topk > 0),
                    )
                    pos_eq = tl.where(valid & eq_thr & active, pos_eq, 0)
                    tl.store(
                        tle_gpu.gpu.local_ptr(
                            s_input_idx,
                            (
                                tl.full([BLOCK_SIZE], next_idx, tl.int32),
                                pos_eq,
                            ),
                        ),
                        cand_idx,
                        mask=valid
                        & eq_thr
                        & active
                        & (pos_eq < SMEM_INPUT)
                        & (l_new_topk > 0),
                    )


def bucket_sort_topk(inputs, starts, ends, topk):
    B, S = inputs.shape
    K = topk
    HISTOGRAM_SIZE = 256
    SMEM_INPUT_SIZE = 4096
    indices = torch.full((B, topk), -1, dtype=torch.int32, device=inputs.device)

    if HAS_TLE and inputs.is_cuda:
        block_size = 1024
        n_tiles = triton.cdiv(S, block_size)
        num_input_tiles = triton.cdiv(SMEM_INPUT_SIZE, block_size)
        hist_size = HISTOGRAM_SIZE * 2
        x = inputs.float() if inputs.dtype != torch.float32 else inputs
        assume_aligned = (
            x.is_contiguous()
            and indices.is_contiguous()
            and S % block_size == 0
            and torch.all(starts == 0).item()
            and torch.all(ends == S).item()
        )
        tle_topk_selector_kernel[(B,)](
            x,
            indices,
            starts,
            ends,
            x.stride(0),
            x.stride(1),
            indices.stride(0),
            indices.stride(1),
            S,
            RADIX=HISTOGRAM_SIZE,
            HIST_SIZE=hist_size,
            ASSUME_ALIGNED=assume_aligned,
            TOPK=topk,
            BLOCK_SIZE=block_size,
            N_TILES=n_tiles,
            SMEM_INPUT=SMEM_INPUT_SIZE,
            NUM_INPUT_TILES=num_input_tiles,
            num_warps=32,
        )
    else:
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

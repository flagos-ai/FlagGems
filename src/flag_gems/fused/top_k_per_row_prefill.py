"""
top_k_per_row_prefill: Triton implementation for vLLM's top_k_per_row_prefill kernel.

Uses histogram-based radix select (O(n) algorithm) for efficient top-k selection.
Based on top_k_per_row_decode implementation.
"""

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn


@triton.jit
def convert_to_uint32(x):
    """Convert float32 to uint32 for histogram binning, preserving order."""
    bits_uint = x.cast(dtype=tl.uint32, bitcast=True)
    bits_uint = tl.where(
        x < 0,
        ~bits_uint & tl.cast((0xFFFFFFFF), tl.uint32, bitcast=True),
        bits_uint | tl.cast((0x80000000), tl.uint32, bitcast=True),
    )
    return bits_uint


@triton.jit
def kernel_topk_prefill_histogram(
    inputs,  # (num_rows, vocab_size) logits
    indices,  # (num_rows, K) output topk indices
    s_input_ids,  # Temp buffer for candidates
    row_starts,  # (num_rows,) start indices
    row_ends,  # (num_rows,) end indices
    stride0,  # logits.stride(0)
    stride1,  # logits.stride(1)
    S: tl.constexpr,  # vocab_size
    K: tl.constexpr,  # top_k
    HISTOGRAM_SIZE: tl.constexpr,
    SMEM_INPUT_SIZE: tl.constexpr,
    BS: tl.constexpr,  # block size for vocab iteration
    BSS: tl.constexpr,  # block size for candidate iteration
):
    """
    Histogram-based radix select for top-k selection (prefill version).
    Uses O(n) algorithm with 256-bin histogram and multi-round refinement.
    """
    row_idx = tl.program_id(0)

    # Load row boundaries directly from arrays
    l_start_idx = tl.load(row_starts + row_idx)
    l_end_idx = tl.load(row_ends + row_idx)
    l_end_idx = tl.minimum(l_end_idx, S)
    l_end_idx = tl.maximum(l_end_idx, 0)
    l_start_idx = tl.maximum(l_start_idx, 0)
    l_start_idx = tl.minimum(l_start_idx, l_end_idx)

    # Calculate valid element count and actual k to select
    valid_count = l_end_idx - l_start_idx
    actual_k = tl.minimum(K, valid_count)

    # Block base pointer definitions
    s_base = inputs + row_idx * stride0
    indices_base = indices + row_idx * K
    s_input_ids_base = s_input_ids + row_idx * SMEM_INPUT_SIZE

    # Early exit if no valid elements
    if valid_count <= 0:
        return

    # Histogram initialization
    s_histogram = tl.zeros([HISTOGRAM_SIZE], dtype=tl.int32)

    # Record how many positions remain to fill the topk array
    l_new_topk = actual_k

    # Round 0: Build histogram using uint32 high 8 bits
    TS = tl.cdiv(S, BS)
    for s in range(TS):
        input_idx = s * BS + tl.arange(0, BS)
        input_mask = (
            (input_idx < l_end_idx) & (input_idx >= l_start_idx) & (input_idx < S)
        )
        if stride1 == 1:
            input_val = tl.load(s_base + input_idx, input_mask, other=float("-inf")).to(
                tl.float32
            )
        else:
            input_val = tl.load(
                s_base + input_idx * stride1, input_mask, other=float("-inf")
            ).to(tl.float32)
        inval_uint8 = (convert_to_uint32(input_val) >> 24) & 0xFF
        s_histogram += inval_uint8.to(tl.int32).histogram(HISTOGRAM_SIZE)

    # Find threshold bin using suffix sum
    s_histogram = s_histogram.cumsum(0, reverse=True)
    mv_idx = tl.arange(1, HISTOGRAM_SIZE + 1) % HISTOGRAM_SIZE
    cond = (s_histogram > l_new_topk) & (
        (s_histogram.gather(mv_idx, 0) <= l_new_topk) | (mv_idx == 0)
    )
    l_threshold_bin_id = cond.argmax(0)
    l_new_topk -= tl.where(
        tl.arange(0, HISTOGRAM_SIZE) == l_threshold_bin_id + 1, s_histogram, 0
    ).max(0)

    # Partition elements: above threshold -> output, equal threshold -> candidates
    sum_val = 0
    thre_bin_sum = 0
    for s in range(TS):
        input_idx = s * BS + tl.arange(0, BS)
        input_mask = (
            (input_idx < l_end_idx) & (input_idx >= l_start_idx) & (input_idx < S)
        )
        if stride1 == 1:
            input_val = tl.load(s_base + input_idx, input_mask, other=float("-inf")).to(
                tl.float32
            )
        else:
            input_val = tl.load(
                s_base + input_idx * stride1, input_mask, other=float("-inf")
            ).to(tl.float32)
        inval_uint8 = (convert_to_uint32(input_val) >> 24) & 0xFF

        # Must AND with input_mask to exclude invalid positions
        over_thre = (inval_uint8.to(tl.int32) > l_threshold_bin_id) & input_mask
        cur_sum = over_thre.to(tl.int32).sum(-1)
        eq_thre = (inval_uint8.to(tl.int32) == l_threshold_bin_id) & input_mask
        thre_bin_cur_sum = eq_thre.to(tl.int32).sum(-1)

        topk_idx = over_thre.to(tl.int32).cumsum(-1)
        thre_bin_idx = eq_thre.to(tl.int32).cumsum(-1)

        # Bound check for candidate storage to avoid buffer overflow
        candidate_write_pos = thre_bin_sum + thre_bin_idx - 1
        eq_thre_bounded = eq_thre & (candidate_write_pos < SMEM_INPUT_SIZE)

        concat_mask = tl.cat(over_thre, eq_thre_bounded, True)
        concat_input = tl.cat(input_idx, input_idx, True)
        concat_pointer_matrix = tl.cat(
            indices_base + sum_val + topk_idx - 1,
            s_input_ids_base + candidate_write_pos,
            True,
        )
        tl.store(concat_pointer_matrix, concat_input, mask=concat_mask)

        # Track actual candidates stored (bounded by SMEM_INPUT_SIZE)
        thre_bin_sum = tl.minimum(thre_bin_sum + thre_bin_cur_sum, SMEM_INPUT_SIZE)
        sum_val += cur_sum

    # Subsequent rounds: refine using more bits (23-16, 15-8, 7-0)
    # Use fixed max iterations to avoid dynamic loop bound issues in Triton JIT
    MAX_CAND_ITERS: tl.constexpr = (SMEM_INPUT_SIZE + BSS - 1) // BSS

    round_num = 1
    while round_num < 4 and l_new_topk > 0:
        ss = tl.cdiv(thre_bin_sum, BSS)
        s_histogram = tl.zeros([HISTOGRAM_SIZE], dtype=tl.int32)
        padding_num = 0.0

        for s in range(MAX_CAND_ITERS):
            if s < ss:
                s_input_idx = s * BSS + tl.arange(0, BSS)
                s_input_idx_mask = s_input_idx < thre_bin_sum
                input_idx = tl.load(
                    s_input_ids_base + s_input_idx, s_input_idx_mask, other=-1
                )
                if stride1 == 1:
                    s_input = tl.load(
                        s_base + input_idx, s_input_idx_mask, other=padding_num
                    ).to(tl.float32)
                else:
                    s_input = tl.load(
                        s_base + input_idx * stride1,
                        s_input_idx_mask,
                        other=padding_num,
                    ).to(tl.float32)
                inval_int32 = (
                    convert_to_uint32(s_input) >> (24 - round_num * 8)
                ) & 0xFF
                s_histogram += inval_int32.to(tl.int32).histogram(HISTOGRAM_SIZE)

        s_histogram = s_histogram.cumsum(0, reverse=True)
        mv_idx = tl.arange(1, HISTOGRAM_SIZE + 1) % HISTOGRAM_SIZE
        cond = (s_histogram > l_new_topk) & (
            (s_histogram.gather(mv_idx, 0) <= l_new_topk) | (mv_idx == 0)
        )
        l_threshold_bin_id = cond.argmax(0)
        l_new_topk -= tl.where(
            tl.arange(0, HISTOGRAM_SIZE) == l_threshold_bin_id + 1, s_histogram, 0
        ).max(0)
        thre_bin_sum, old_thre_bin_sum = 0, thre_bin_sum
        old_ss = ss

        for s in range(MAX_CAND_ITERS):
            if s < old_ss:
                s_input_idx = s * BSS + tl.arange(0, BSS)
                s_input_idx_mask = s_input_idx < old_thre_bin_sum
                input_idx = tl.load(
                    s_input_ids_base + s_input_idx, s_input_idx_mask, other=-1
                )
                if stride1 == 1:
                    s_input = tl.load(
                        s_base + input_idx, s_input_idx_mask, other=padding_num
                    ).to(tl.float32)
                else:
                    s_input = tl.load(
                        s_base + input_idx * stride1,
                        s_input_idx_mask,
                        other=padding_num,
                    ).to(tl.float32)
                inval_int32 = (
                    convert_to_uint32(s_input) >> (24 - round_num * 8)
                ) & 0xFF

                # Must AND with s_input_idx_mask to exclude invalid positions (padding values)
                over_thre = (
                    inval_int32.to(tl.int32) > l_threshold_bin_id
                ) & s_input_idx_mask
                cur_sum = over_thre.to(tl.int32).sum(-1)
                eq_thre = (
                    inval_int32.to(tl.int32) == l_threshold_bin_id
                ) & s_input_idx_mask
                thre_bin_cur_sum = eq_thre.to(tl.int32).sum(-1)

                topk_idx = over_thre.to(tl.int32).cumsum(-1)
                thre_bin_idx = eq_thre.to(tl.int32).cumsum(-1)

                # Bound check for candidate storage to avoid buffer overflow
                candidate_write_pos = thre_bin_sum + thre_bin_idx - 1
                eq_thre_bounded = eq_thre & (candidate_write_pos < SMEM_INPUT_SIZE)

                concat_mask = tl.cat(over_thre, eq_thre_bounded, True)
                concat_input = tl.cat(input_idx, input_idx, True)
                concat_pointer_matrix = tl.cat(
                    indices_base + sum_val + topk_idx - 1,
                    s_input_ids_base + candidate_write_pos,
                    True,
                )
                tl.store(concat_pointer_matrix, concat_input, mask=concat_mask)

                # Track actual candidates stored (bounded by SMEM_INPUT_SIZE)
                thre_bin_sum = tl.minimum(
                    thre_bin_sum + thre_bin_cur_sum, SMEM_INPUT_SIZE
                )
                sum_val += cur_sum

        round_num += 1

    # Copy remaining candidates if needed
    # Copy min(l_new_topk, thre_bin_sum) elements - we may have fewer candidates than needed
    copy_count = tl.minimum(l_new_topk, thre_bin_sum)
    if copy_count > 0:
        ss = tl.cdiv(copy_count, BSS)
        MAX_COPY_ITERS: tl.constexpr = (K + BSS - 1) // BSS
        for s in range(MAX_COPY_ITERS):
            if s < ss:
                s_input_idx = s * BSS + tl.arange(0, BSS)
                s_input_idx_mask = s_input_idx < copy_count
                input_idx = tl.load(
                    s_input_ids_base + s_input_idx, s_input_idx_mask, other=-1
                )
                tl.store(
                    indices_base + sum_val + tl.arange(0, BSS),
                    input_idx,
                    mask=s_input_idx_mask,
                )
                sum_val += BSS


def top_k_per_row_prefill(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    top_k: int,
) -> None:
    """
    Select top-k indices from each row of logits for prefill phase.

    Uses histogram-based radix select (O(n) algorithm) for efficient selection.

    Args:
        logits: Input logits tensor [num_rows, vocab_size], float32
        row_starts: Start indices for each row [num_rows], int32
        row_ends: End indices for each row [num_rows], int32
        indices: Output indices tensor [num_rows, top_k], int32 (modified in-place)
        num_rows: Number of rows
        stride0: logits.stride(0)
        stride1: logits.stride(1)
        top_k: Number of top elements to select
    """
    assert logits.dtype == torch.float32, "logits must be float32"
    assert row_starts.dtype == torch.int32, "row_starts must be int32"
    assert row_ends.dtype == torch.int32, "row_ends must be int32"
    assert indices.dtype == torch.int32, "indices must be int32"

    vocab_size = logits.shape[1]
    device = logits.device

    # Fixed block sizes to avoid autotune LLVM issues
    BS = 1024  # Block size for vocab iteration
    BSS = 256  # Block size for candidate iteration
    HISTOGRAM_SIZE = 256
    # For worst case (e.g., 10LSBits data where all values have same high bits),
    # all elements could become candidates. Use vocab_size as upper bound.
    SMEM_INPUT_SIZE = min(vocab_size, 65536)  # Cap at 64K to limit memory usage

    # Allocate temp buffer for candidates
    s_input_ids = torch.zeros(
        num_rows, SMEM_INPUT_SIZE, dtype=torch.int32, device=device
    )

    # Initialize indices to -1 (invalid marker)
    indices.fill_(-1)

    grid = (num_rows,)
    with torch_device_fn.device(device):
        kernel_topk_prefill_histogram[grid](
            logits,
            indices,
            s_input_ids,
            row_starts,
            row_ends,
            stride0,
            stride1,
            vocab_size,
            top_k,
            HISTOGRAM_SIZE,
            SMEM_INPUT_SIZE,
            BS,
            BSS,
        )

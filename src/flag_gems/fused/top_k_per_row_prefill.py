"""Triton top_k_per_row_prefill for DeepSeek V4 sparse attention.

Replaces vLLM's persistent_topk CUDA kernel with a Triton implementation.

Background:
    In DeepSeek V4 prefill, each token computes attention logits over a subset of
    the vocabulary [row_starts[i], row_ends[i]) and selects the top-K indices.
    Typical config: vocab_size=129280, top_k=1024, num_rows=1 (decode) or 32+ (prefill).

Strategy:
    1. In-place masking kernel: set logits outside [row_starts, row_ends) to -inf.
       Early exit when the row uses full vocab (start==0, end>=vocab_size), which is
       the common case during inference and avoids unnecessary memory writes.
    2. Adaptive top-K selection:
       - num_rows=1: torch.argsort (backed by CUB radix sort, O(N) for single row,
         ~2x faster than torch.topk for large vocab on a single row)
       - num_rows>1: torch.topk with sorted=False (heap-based O(N log k), better
         parallelism across rows than argsort)
    3. Fused postprocess kernel: single Triton kernel performs slice + cast + subtract
       in one pass, converting absolute vocab indices to 0-based indices relative to
       row_starts[i]. Saves one kernel launch vs separate slice/subtract ops.

Performance (DeepSeek V4 config, vocab=129280, top_k=1024):
    - num_rows=1:  0.89x vs vLLM CUDA (competitive, bounded by argsort)
    - num_rows=32: 0.38x vs vLLM CUDA (bounded by torch.topk on large vocab)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _mask_invalid_kernel(
    logits_ptr,
    row_starts_ptr,
    row_ends_ptr,
    stride0,  # logits row stride (= vocab_size for contiguous tensor)
    BLOCK_SIZE: tl.constexpr,  # 8192: tuned for 129280 vocab (16 blocks/row)
    VOCAB_SIZE: tl.constexpr,  # total vocabulary size (e.g. 129280)
):
    """Mask logits outside [row_starts[i], row_ends[i]) to -inf, in-place.

    Grid: (num_rows * num_blocks_per_row,) — 1D flat grid.
    Each program handles one BLOCK_SIZE chunk of one row.
    Early exits when the row uses full vocab to avoid unnecessary stores.
    """
    pid = tl.program_id(0)
    num_blocks_per_row = tl.cdiv(VOCAB_SIZE, BLOCK_SIZE)
    row_id = pid // num_blocks_per_row
    block_id = pid % num_blocks_per_row

    start = tl.load(row_starts_ptr + row_id)
    end = tl.load(row_ends_ptr + row_id)

    # Early exit: most rows in inference use full vocab (start=0, end=vocab_size).
    # Skipping these avoids ~90% of memory writes in typical workloads.
    if start == 0 and end >= VOCAB_SIZE:
        return

    # Compute which positions in this block are outside the valid range
    offs = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_of_range = (offs < start) | (offs >= end)
    # Only write to positions that are both within vocab bounds AND out of valid range
    mask = (offs < VOCAB_SIZE) & out_of_range

    tl.store(logits_ptr + row_id * stride0 + offs, float("-inf"), mask=mask)


@triton.jit
def _fused_postprocess_kernel(
    src_ptr,  # source indices (from argsort or topk)
    dst_ptr,  # destination: output indices buffer [num_rows, top_k]
    row_starts_ptr,  # per-row start offsets for index adjustment
    num_rows: tl.constexpr,
    top_k: tl.constexpr,  # 1024 in DeepSeek V4
    src_stride0: tl.constexpr,  # row stride of src (vocab_size for argsort, top_k for topk)
    BLOCK_SIZE: tl.constexpr,  # next_power_of_2(top_k), e.g. 1024
):
    """Fused slice + cast + subtract: convert absolute indices to row-relative.

    For each row i, computes: dst[i, :top_k] = src[i, :top_k] - row_starts[i]
    This converts absolute vocab indices to 0-based indices within the valid range.
    Grid: (num_rows,) — one program per row.
    """
    row_id = tl.program_id(0)
    if row_id >= num_rows:
        return

    row_start = tl.load(row_starts_ptr + row_id)

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < top_k

    src_idx = row_id * src_stride0 + offs
    src_vals = tl.load(src_ptr + src_idx, mask=mask, other=0)

    # Subtract row_start to get 0-based index within [row_start, row_end)
    dst_vals = (src_vals - row_start).to(tl.int32)

    dst_idx = row_id * top_k + offs
    tl.store(dst_ptr + dst_idx, dst_vals, mask=mask)


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
        raise ValueError(
            f"top_k ({top_k}) must not exceed vocab_size ({vocab_size})"
        )

    # --- Phase 1: Mask invalid ranges to -inf ---
    # BLOCK_SIZE=8192 chosen to balance occupancy vs. grid size:
    # For vocab=129280, this gives ceil(129280/8192)=16 blocks per row.
    # num_warps=2 is sufficient since masking is memory-bound (simple store).
    MASK_BS = 8192
    num_mask_blocks = (vocab_size + MASK_BS - 1) // MASK_BS
    _mask_invalid_kernel[(num_rows * num_mask_blocks,)](
        logits,
        row_starts,
        row_ends,
        stride0,
        BLOCK_SIZE=MASK_BS,
        VOCAB_SIZE=vocab_size,
        num_warps=2,
    )

    # --- Phase 2: Select top-K indices ---
    # POSTPROC_BLOCK must be power-of-2 >= top_k for tl.arange.
    # For top_k=1024, this is exactly 1024 (no waste).
    POSTPROC_BLOCK = triton.next_power_of_2(top_k)

    if num_rows == 1:
        # Single row path: torch.argsort uses CUB radix sort under the hood.
        # For large vocab (129280) with a single row, radix sort O(N) is ~2x faster
        # than torch.topk's heap-based O(N log k) because it fully utilizes GPU
        # parallelism without the sequential heap maintenance bottleneck.
        sorted_idx = torch.argsort(logits, dim=1, descending=True, stable=False)
        # src_stride0=vocab_size because argsort returns full-width sorted indices
        _fused_postprocess_kernel[(1,)](
            sorted_idx,
            indices,
            row_starts,
            num_rows=1,
            top_k=top_k,
            src_stride0=vocab_size,
            BLOCK_SIZE=POSTPROC_BLOCK,
            num_warps=4,
        )
    else:
        # Multi-row path: torch.topk with sorted=False.
        # For batched rows, topk's heap approach has better parallelism across rows
        # than argsort (which serializes the full sort per row).
        # sorted=False avoids an unnecessary final sort pass.
        _, top_idx = torch.topk(logits, top_k, dim=1, largest=True, sorted=False)
        # src_stride0=top_k because topk output shape is [num_rows, top_k]
        _fused_postprocess_kernel[(num_rows,)](
            top_idx,
            indices,
            row_starts,
            num_rows=num_rows,
            top_k=top_k,
            src_stride0=top_k,
            BLOCK_SIZE=POSTPROC_BLOCK,
            num_warps=4,
        )

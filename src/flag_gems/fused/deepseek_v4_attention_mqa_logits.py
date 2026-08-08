
"""BF16 non-paged MQA logits kernel (Triton).

Computes sparse-attention logits for DeepSeek-V4 style MLA prefill:
    logits[m, n] = sum_h(relu(sum_d(q[m,h,d] * kv[n,d])) * weights[m,h])
for n in [cu_start[m], cu_end[m]).

This is the bf16 analog of the fp8_fp4_mqa_logits DeepGEMM kernel, operating
directly on bf16 Q and KV without quantization.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# BF16 KV Gather: paged cache -> contiguous buffer
# ---------------------------------------------------------------------------


@triton.jit
def _gather_bf16_kv_kernel(
    # Inputs
    kv_cache_ptr,  # [num_blocks, block_size, head_dim] bf16
    block_table_ptr,  # [num_reqs, max_blocks_per_req] int32
    cu_seq_lens_ptr,  # [num_reqs + 1] int32
    token_to_seq_ptr,  # [total_seq_lens] int32 — request index per token
    # Output
    dst_ptr,  # [total_seq_lens, head_dim] bf16
    # Dims
    total_seq_lens,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_blocks_per_req,
    cache_stride_block: tl.int64,  # kv_cache.stride(0) in elements
    cache_stride_token: tl.int64,  # kv_cache.stride(1) in elements
    dst_stride_token: tl.int64,  # dst.stride(0) in elements
    # Block size for parallelism
    BLOCK_D: tl.constexpr,
):
    """Gather KV from paged cache into a contiguous buffer.

    Grid: (total_seq_lens,)
    Each program copies one token's head_dim elements from page to dst.
    """
    token_idx = tl.program_id(0)
    if token_idx >= total_seq_lens:
        return

    d_offset = tl.arange(0, BLOCK_D)
    d_mask = d_offset < head_dim

    # Look up which request this token belongs to
    req_idx = tl.load(token_to_seq_ptr + token_idx)
    seq_start = tl.load(cu_seq_lens_ptr + req_idx)
    in_seq_idx = token_idx - seq_start

    # Map to page
    block_idx_in_seq = in_seq_idx // block_size
    pos_in_block = in_seq_idx % block_size

    # Look up physical block
    phys_block = tl.load(
        block_table_ptr + req_idx * max_blocks_per_req + block_idx_in_seq
    )

    # Source offset in kv_cache
    src_offset = (
        phys_block * cache_stride_block
        + pos_in_block * cache_stride_token
        + d_offset
    )

    # Load from cache
    vals = tl.load(kv_cache_ptr + src_offset, mask=d_mask, other=0.0)

    # Store to dst
    dst_offset = token_idx * dst_stride_token + d_offset
    tl.store(dst_ptr + dst_offset, vals, mask=d_mask)


def gather_bf16_kv_from_pages(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    token_to_seq: torch.Tensor,
    total_seq_lens: int,
    dst: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather BF16 KV from paged cache into a contiguous [total_seq_lens, D] buffer.

    Args:
        kv_cache: [num_blocks, block_size, head_dim], bf16
        block_table: [num_reqs, max_blocks_per_req], int32
        cu_seq_lens: [num_reqs + 1], int32 (cumulative seq lengths)
        token_to_seq: [total_seq_lens], int32 — maps each token to its request
        total_seq_lens: total number of tokens to gather
        dst: optional pre-allocated output [total_seq_lens, head_dim], bf16

    Returns:
        Contiguous bf16 tensor [total_seq_lens, head_dim].
    """
    _, block_size, head_dim = kv_cache.shape
    max_blocks_per_req = block_table.shape[1]

    if dst is None:
        dst = torch.empty(
            (total_seq_lens, head_dim),
            dtype=torch.bfloat16,
            device=kv_cache.device,
        )

    if total_seq_lens == 0:
        return dst

    BLOCK_D = triton.next_power_of_2(head_dim)
    grid = (total_seq_lens,)

    _gather_bf16_kv_kernel[grid](
        kv_cache_ptr=kv_cache,
        block_table_ptr=block_table,
        cu_seq_lens_ptr=cu_seq_lens,
        token_to_seq_ptr=token_to_seq,
        dst_ptr=dst,
        total_seq_lens=total_seq_lens,
        head_dim=head_dim,
        block_size=block_size,
        max_blocks_per_req=max_blocks_per_req,
        cache_stride_block=kv_cache.stride(0),
        cache_stride_token=kv_cache.stride(1),
        dst_stride_token=dst.stride(0),
        BLOCK_D=BLOCK_D,
    )

    return dst


@triton.jit
def _bf16_mqa_logits_kernel(
    Q_ptr,  # bf16 [seq_len, H, D]
    KV_ptr,  # bf16 [seq_len_kv, D]
    weights_ptr,  # fp32 [seq_len, H]
    cu_start_ptr,  # int32 [seq_len]
    cu_end_ptr,  # int32 [seq_len]
    logits_ptr,  # fp32 [seq_len, stride_logits_s]
    seq_len_kv,
    # strides
    stride_q_s: tl.int64,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_kv_s: tl.int64,
    stride_kv_d: tl.constexpr,
    stride_w_s: tl.int64,
    stride_w_h: tl.constexpr,
    stride_logits_s: tl.int64,
    stride_logits_k: tl.int64,
    # compile-time constants
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """One program handles one query row (m)."""
    row_id = tl.program_id(0)
    # Process rows from end to start to reduce tail effect
    row_id = tl.num_programs(0) - row_id - 1
    tl.assume(row_id >= 0)
    tl.assume(stride_q_s > 0)
    tl.assume(stride_q_h > 0)
    tl.assume(stride_q_d > 0)
    tl.assume(stride_kv_s > 0)
    tl.assume(stride_kv_d > 0)
    tl.assume(stride_w_s > 0)
    tl.assume(stride_w_h > 0)

    logits_row_ptrs = logits_ptr + row_id * stride_logits_s

    h_inds = tl.arange(0, NUM_HEADS)[:, None]
    d_inds = tl.arange(0, HEAD_SIZE)

    # Load Q[row_id, :, :] -> [NUM_HEADS, HEAD_SIZE]
    q_ptrs = (
        Q_ptr
        + row_id * stride_q_s
        + h_inds * stride_q_h
        + d_inds[None, :] * stride_q_d
    )
    q_block = tl.load(q_ptrs, cache_modifier=".cg")  # [NUM_HEADS, HEAD_SIZE]

    # Load weights[row_id, :] -> [NUM_HEADS, 1]
    w_ptrs = weights_ptr + row_id * stride_w_s + h_inds * stride_w_h
    w_block = tl.load(w_ptrs, cache_modifier=".cg").to(tl.float32)

    # Load start/end indices for this row
    start_ind = tl.load(cu_start_ptr + row_id)
    end_ind = tl.load(cu_end_ptr + row_id)

    start_ind = tl.maximum(start_ind, 0)
    end_ind = tl.minimum(end_ind, seq_len_kv)
    shifted_end = end_ind - start_ind
    shifted_unmasked_end = shifted_end // BLOCK_KV * BLOCK_KV

    kv_col_offsets = tl.arange(0, BLOCK_KV) + start_ind
    kv_ptrs = (
        KV_ptr
        + kv_col_offsets[None, :] * stride_kv_s
        + d_inds[:, None] * stride_kv_d
    )

    logits_ptrs = logits_row_ptrs + kv_col_offsets * stride_logits_k

    # Loop over full KV tiles (no masking needed)
    for _ in tl.range(0, shifted_unmasked_end, BLOCK_KV):
        # Load KV block: [HEAD_SIZE, BLOCK_KV]
        kv_block = tl.load(kv_ptrs)

        # scores: [NUM_HEADS, BLOCK_KV] = [NUM_HEADS, HEAD_SIZE] x [HEAD_SIZE, BLOCK_KV]
        scores = tl.dot(q_block, kv_block, input_precision="ieee")
        # ReLU
        scores = tl.maximum(scores, 0.0)
        # Weight and reduce across heads: [NUM_HEADS, BLOCK_KV] -> [BLOCK_KV]
        scores = scores * w_block
        scores = tl.sum(scores, axis=0)
        tl.store(logits_ptrs, scores)

        kv_ptrs += BLOCK_KV * stride_kv_s
        kv_col_offsets += BLOCK_KV
        logits_ptrs += BLOCK_KV * stride_logits_k

    # Handle the last (possibly partial) tile with masking
    kv_col_mask = kv_col_offsets < end_ind
    kv_block = tl.load(kv_ptrs, mask=kv_col_mask[None, :], other=0.0)

    scores = tl.dot(q_block, kv_block, input_precision="ieee")
    scores = tl.maximum(scores, 0.0)
    scores = scores * w_block
    scores = tl.sum(scores, axis=0)

    in_window = (kv_col_offsets >= start_ind) & (kv_col_offsets < end_ind)
    tl.store(logits_ptrs, scores, mask=in_window)


def bf16_mqa_logits(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    cu_seq_len_k_start: torch.Tensor,
    cu_seq_len_k_end: torch.Tensor,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Compute non-paged MQA logits with bf16 Q and KV.

    Args:
        q: Query tensor [seq_len, num_heads, head_dim], dtype bfloat16.
        kv: Key-value tensor [seq_len_kv, head_dim], dtype bfloat16.
        weights: Weight tensor [seq_len, num_heads], dtype float32.
        cu_seq_len_k_start: Start indices [seq_len], dtype int32.
        cu_seq_len_k_end: End indices [seq_len], dtype int32.
        clean_logits: If True, fill positions outside [start, end) with -inf.

    Returns:
        logits: [seq_len, seq_len_kv], dtype float32.
    """
    assert q.dtype == torch.bfloat16
    assert kv.dtype == torch.bfloat16
    assert weights.dtype == torch.float32
    assert cu_seq_len_k_start.dtype == torch.int32
    assert cu_seq_len_k_end.dtype == torch.int32

    seq_len, num_heads, head_dim = q.shape
    seq_len_kv = kv.shape[0]

    assert num_heads & (num_heads - 1) == 0, "num_heads must be power of 2"
    assert head_dim & (head_dim - 1) == 0, "head_dim must be power of 2"
    assert q.is_contiguous()
    assert kv.is_contiguous()

    BLOCK_KV = 128

    # Initialize output with -inf for clean_logits semantics
    if clean_logits:
        logits = torch.full(
            (seq_len, seq_len_kv),
            fill_value=-float("inf"),
            dtype=torch.float32,
            device=q.device,
        )
    else:
        logits = torch.empty(
            (seq_len, seq_len_kv),
            dtype=torch.float32,
            device=q.device,
        )

    stride_q_s, stride_q_h, stride_q_d = q.stride()
    stride_kv_s, stride_kv_d = kv.stride()
    stride_w_s, stride_w_h = weights.stride()
    stride_logits_s, stride_logits_k = logits.stride()

    _bf16_mqa_logits_kernel[(seq_len,)](
        Q_ptr=q,
        KV_ptr=kv,
        weights_ptr=weights,
        cu_start_ptr=cu_seq_len_k_start,
        cu_end_ptr=cu_seq_len_k_end,
        logits_ptr=logits,
        seq_len_kv=seq_len_kv,
        stride_q_s=stride_q_s,
        stride_q_h=stride_q_h,
        stride_q_d=stride_q_d,
        stride_kv_s=stride_kv_s,
        stride_kv_d=stride_kv_d,
        stride_w_s=stride_w_s,
        stride_w_h=stride_w_h,
        stride_logits_s=stride_logits_s,
        stride_logits_k=stride_logits_k,
        NUM_HEADS=num_heads,
        HEAD_SIZE=head_dim,
        BLOCK_KV=BLOCK_KV,
        num_warps=4,
        num_stages=2,
    )

    return logits

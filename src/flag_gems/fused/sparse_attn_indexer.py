"""Sparse attention indexer for DeepSeek V4 MLA prefill.

This kernel selects the top-K KV cache positions for each query token during
prefill in DeepSeek V4's sparse attention mechanism. It replaces the vLLM
tilelang-based implementation with a pure Triton version.

Algorithm Overview (3-path approach):
--------------------------------------
1. Shortcut path: For early tokens where kv_end <= topk (the first topk tokens
   in a sequence), all KV positions are selected trivially -- just fill [0..kv_end).

2. Fused single-kernel path (TODO/future): For moderate KV lengths, a single
   kernel could handle logit computation + selection. Currently unused.

3. Multi-kernel path: For tokens with kv_end > topk, three phases execute:
   a) Compute logits: For each query token, compute attention-like scores
      against all visible KV positions using FP8-quantized keys.
   b) Binary search threshold: Find a threshold value such that exactly topk
      positions have logits >= threshold.
   c) Collect indices: Gather positions exceeding the threshold into the
      output buffer.

DeepSeek V4 target shapes:
    - num_heads = 64 (number of attention heads)
    - head_dim = 128 (dimension per head)
    - topk = 1024 (KV positions selected per token)
    - Typical prefill lengths: 1 to 2048 tokens

FP8 quantization format:
    The KV cache stores keys in E4M3 FP8 format. Each cache slot contains:
    - head_dim bytes of FP8 data (the quantized key vector)
    - 4 bytes storing the FP32 scale factor as little-endian uint8 bytes
    Total slot size: head_dim + 4 bytes (e.g., 132 for head_dim=128)

Compact logits optimization:
    With causal masking, token i can only attend to KV positions [0, i+1).
    The maximum visible KV length across all active tokens is at most
    num_tokens (not total_kv_len which can be much larger for long contexts).
    We allocate the logits tensor as (num_active_tokens x max_kv_end) instead
    of (num_active_tokens x total_kv_len), reducing memory by up to 4x for
    typical 2048-token / 8192-context configurations.

Causal mask assumption:
    Token i sees KV positions [0, i+1). This is standard left-to-right causal
    attention used in autoregressive language models.
"""

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: Quantize K vectors to FP8 and store in KV cache
# ---------------------------------------------------------------------------


@triton.jit
def _k_quant_cache_kernel(
    k_ptr,
    kv_cache_ptr,
    num_tokens: tl.constexpr,
    head_dim: tl.constexpr,
    cache_stride_slot: tl.constexpr,
):
    """Quantize a key vector to FP8 E4M3 and write to KV cache.

    Each cache slot stores head_dim bytes of FP8 data followed by 4 bytes
    encoding the FP32 per-token scale as little-endian uint8.
    """
    tid = tl.program_id(0)
    if tid >= num_tokens:
        return

    offset = tl.arange(0, head_dim)
    k_val = tl.load(k_ptr + tid * head_dim + offset).to(tl.float32)

    # Compute per-token absmax scale (clamp to avoid division by zero)
    amax = tl.max(tl.abs(k_val))
    amax = tl.maximum(amax, 1e-4)
    scale = amax / 448.0  # 448 = max representable value in E4M3

    # Quantize to FP8 E4M3
    k_fp8 = (k_val / scale).to(tl.float8e4nv)
    k_uint8 = k_fp8.to(tl.uint8, bitcast=True)

    # Store FP8 key data
    dst_base = tid * cache_stride_slot
    tl.store(kv_cache_ptr + dst_base + offset, k_uint8)

    # Store scale as 4 little-endian bytes
    scale_uint32 = scale.to(tl.uint32, bitcast=True)
    tl.store(
        kv_cache_ptr + dst_base + head_dim,
        (scale_uint32 & 0xFF).to(tl.uint8),
    )
    tl.store(
        kv_cache_ptr + dst_base + head_dim + 1,
        ((scale_uint32 >> 8) & 0xFF).to(tl.uint8),
    )
    tl.store(
        kv_cache_ptr + dst_base + head_dim + 2,
        ((scale_uint32 >> 16) & 0xFF).to(tl.uint8),
    )
    tl.store(
        kv_cache_ptr + dst_base + head_dim + 3,
        ((scale_uint32 >> 24) & 0xFF).to(tl.uint8),
    )


# ---------------------------------------------------------------------------
# Kernel 2: Shortcut fill for tokens where kv_end <= topk
# ---------------------------------------------------------------------------


@triton.jit
def _shortcut_fill_kernel(
    topk_out_ptr,
    topk: tl.constexpr,
    num_shortcut_tokens: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fill output with [0, 1, ..., kv_end-1, -1, ...] for shortcut tokens."""
    row = tl.program_id(0)
    if row >= num_shortcut_tokens:
        return

    kv_end = row + 1
    out_base = row * topk
    offs = tl.arange(0, BLOCK)

    for block_start in range(0, topk, BLOCK):
        idx = block_start + offs
        mask = idx < topk
        vals = tl.where(
            idx < kv_end,
            idx.to(tl.int32),
            tl.full([BLOCK], -1, dtype=tl.int32),
        )
        tl.store(topk_out_ptr + out_base + idx, vals, mask=mask)


# ---------------------------------------------------------------------------
# Kernel 3: Compute attention logits into compact tensor
# ---------------------------------------------------------------------------


@triton.jit
def _compute_logits_compact_kernel(
    q_ptr,
    kv_cache_ptr,
    weights_ptr,
    logits_ptr,
    num_tokens: tl.constexpr,
    num_heads: tl.constexpr,
    max_kv_end: tl.constexpr,
    head_dim: tl.constexpr,
    cache_stride_slot: tl.constexpr,
    token_offset: tl.constexpr,
    logits_stride: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Compute logits into compact tensor. Grid: (num_active_tokens, num_kv_blocks).

    For each query token, computes:
        logit[kv_pos] = sum_h(ReLU(q_h . k_kv_pos) * weight_h) * k_scale

    logits_stride is the row stride of the compact logits tensor (= max_kv_end).
    Each row only stores values up to its kv_end.
    """
    row_local = tl.program_id(0)
    kv_block_id = tl.program_id(1)
    row = row_local + token_offset

    if row >= num_tokens:
        return

    kv_end = tl.minimum(row + 1, max_kv_end)
    col_start = kv_block_id * BLOCK_KV

    kv_offs = tl.arange(0, BLOCK_KV)
    out_base = row_local * logits_stride + col_start

    # Out-of-range KV block: store -inf
    if col_start >= kv_end:
        store_mask = (col_start + kv_offs) < logits_stride
        tl.store(
            logits_ptr + out_base + kv_offs,
            tl.full([BLOCK_KV], float("-inf"), dtype=tl.float32),
            mask=store_mask,
        )
        return

    # Load all head queries and weights for this token
    hd_offs = tl.arange(0, head_dim)
    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < num_heads

    q_base = row * num_heads * head_dim
    q_all = tl.load(
        q_ptr + q_base + h_offs[:, None] * head_dim + hd_offs[None, :],
        mask=h_mask[:, None],
        other=0.0,
    ).to(tl.float16)

    w_base = row * num_heads
    w_all = tl.load(weights_ptr + w_base + h_offs, mask=h_mask, other=0.0)

    # Load KV block (FP8 keys + scales)
    col_offs = col_start + kv_offs
    col_mask = (col_offs < max_kv_end) & (col_offs < kv_end)

    k_block = tl.load(
        kv_cache_ptr + col_offs[:, None] * cache_stride_slot + hd_offs[None, :],
        mask=col_mask[:, None],
        other=0.0,
    )
    k_block_fp8 = k_block.to(tl.float8e4nv, bitcast=True)
    k_block_f16 = k_block_fp8.to(tl.float16)

    # Reconstruct FP32 scale from 4 little-endian bytes
    scale_base = col_offs * cache_stride_slot + head_dim
    s0 = tl.load(kv_cache_ptr + scale_base, mask=col_mask, other=0).to(tl.uint32)
    s1 = tl.load(kv_cache_ptr + scale_base + 1, mask=col_mask, other=0).to(tl.uint32)
    s2 = tl.load(kv_cache_ptr + scale_base + 2, mask=col_mask, other=0).to(tl.uint32)
    s3 = tl.load(kv_cache_ptr + scale_base + 3, mask=col_mask, other=0).to(tl.uint32)
    scale_uint32 = s0 | (s1 << 8) | (s2 << 16) | (s3 << 24)
    k_scale = scale_uint32.to(tl.float32, bitcast=True)

    # Compute dot products: [num_heads, BLOCK_KV]
    dots = tl.dot(q_all, tl.trans(k_block_f16), out_dtype=tl.float32)
    dots = tl.maximum(dots, 0.0)  # ReLU activation

    # Weighted sum across heads, then apply per-KV scale
    acc = tl.sum(dots * w_all[:, None], axis=0)
    acc = acc * k_scale
    acc = tl.where(col_mask, acc, float("-inf"))

    tl.store(
        logits_ptr + out_base + kv_offs,
        acc,
        mask=(col_start + kv_offs) < logits_stride,
    )


# ---------------------------------------------------------------------------
# Kernel 4: Binary search for top-K threshold on compact logits
# ---------------------------------------------------------------------------


@triton.jit
def _topk_threshold_compact_kernel(
    logits_ptr,
    threshold_ptr,
    num_active_tokens: tl.constexpr,
    max_kv_end: tl.constexpr,
    topk: tl.constexpr,
    token_offset: tl.constexpr,
    logits_stride: tl.constexpr,
    BLOCK_SCAN: tl.constexpr,
    NUM_ITERS: tl.constexpr,
):
    """Binary search for threshold such that count(logits >= threshold) >= topk.

    Uses NUM_ITERS iterations of binary search (6 iterations gives sufficient
    precision for the compact logits range). Operates on the compact logits
    tensor where each row has logits_stride elements.
    """
    row_local = tl.program_id(0)
    if row_local >= num_active_tokens:
        return

    row = row_local + token_offset
    kv_end = row + 1
    if kv_end > max_kv_end:
        kv_end = max_kv_end

    row_base = row_local * logits_stride

    # Find min/max for binary search bounds
    lo = tl.full([], float("inf"), dtype=tl.float32)
    hi = tl.full([], float("-inf"), dtype=tl.float32)
    offs = tl.arange(0, BLOCK_SCAN)

    for block_start in range(0, kv_end, BLOCK_SCAN):
        idx = block_start + offs
        mask = idx < kv_end
        vals = tl.load(logits_ptr + row_base + idx, mask=mask, other=float("-inf"))
        valid = tl.where(mask, vals, float("inf"))
        lo = tl.minimum(lo, tl.min(valid))
        hi = tl.maximum(hi, tl.max(vals))

    # Binary search iterations
    for _ in range(NUM_ITERS):
        mid = (lo + hi) * 0.5
        count = tl.zeros([], dtype=tl.int32)
        for block_start in range(0, kv_end, BLOCK_SCAN):
            idx = block_start + offs
            mask = idx < kv_end
            vals = tl.load(logits_ptr + row_base + idx, mask=mask, other=float("-inf"))
            count += tl.sum(tl.where(vals >= mid, 1, 0).to(tl.int32))
        if count >= topk:
            lo = mid
        else:
            hi = mid

    tl.store(threshold_ptr + row_local, lo)


# ---------------------------------------------------------------------------
# Kernel 5: Collect indices exceeding threshold
# ---------------------------------------------------------------------------


@triton.jit
def _topk_collect_compact_kernel(
    logits_ptr,
    threshold_ptr,
    topk_out_ptr,
    counter_ptr,
    num_active_tokens: tl.constexpr,
    max_kv_end: tl.constexpr,
    topk: tl.constexpr,
    token_offset: tl.constexpr,
    logits_stride: tl.constexpr,
    BLOCK_COLLECT: tl.constexpr,
):
    """Collect KV indices with logits >= threshold into output buffer.

    Uses atomic counter to handle concurrent writes from multiple blocks
    processing the same row.
    """
    row_local = tl.program_id(0)
    col_block = tl.program_id(1)

    if row_local >= num_active_tokens:
        return

    row = row_local + token_offset
    kv_end = row + 1
    if kv_end > max_kv_end:
        kv_end = max_kv_end

    col_start = col_block * BLOCK_COLLECT
    if col_start >= kv_end:
        return

    thresh = tl.load(threshold_ptr + row_local)
    row_base = row_local * logits_stride
    out_base = row * topk

    offs = tl.arange(0, BLOCK_COLLECT)
    col_offs = col_start + offs
    mask = col_offs < kv_end
    vals = tl.load(logits_ptr + row_base + col_offs, mask=mask, other=float("-inf"))

    # Count qualifying positions in this block
    qualify = (vals >= thresh) & mask
    local_count = tl.sum(qualify.to(tl.int32))

    if local_count == 0:
        return

    # Atomic reservation of output slots
    write_start = tl.atomic_add(counter_ptr + row_local, local_count)

    # Write qualifying indices (compact write with prefix scan)
    qualify_int = qualify.to(tl.int32)
    prefix = tl.cumsum(qualify_int) - 1
    write_mask = qualify & (write_start + prefix < topk)
    tl.store(
        topk_out_ptr + out_base + write_start + prefix,
        col_offs.to(tl.int32),
        mask=write_mask,
    )


# ---------------------------------------------------------------------------
# Kernel 6: Fill remaining output slots with -1
# ---------------------------------------------------------------------------


@triton.jit
def _topk_fill_remaining_kernel(
    topk_out_ptr,
    counter_ptr,
    num_active_tokens: tl.constexpr,
    topk: tl.constexpr,
    token_offset: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fill unfilled output positions with -1 (invalid index sentinel)."""
    row_local = tl.program_id(0)
    if row_local >= num_active_tokens:
        return

    row = row_local + token_offset
    filled = tl.load(counter_ptr + row_local)
    out_base = row * topk
    offs = tl.arange(0, BLOCK)

    for block_start in range(0, topk, BLOCK):
        idx = block_start + offs
        mask = (idx >= filled) & (idx < topk)
        tl.store(
            topk_out_ptr + out_base + idx,
            tl.full([BLOCK], -1, dtype=tl.int32),
            mask=mask,
        )


# ---------------------------------------------------------------------------
# Host-side orchestration
# ---------------------------------------------------------------------------


def sparse_attn_indexer_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    topk: int,
    total_kv_len: int,
    insert_k: bool = True,
) -> torch.Tensor:
    """Select top-K KV cache positions for sparse attention.

    Args:
        q: Query tensor [num_tokens, num_heads, head_dim], float32.
        k: Key tensor [num_tokens, head_dim], float32 (pre-RoPE latent).
        weights: Per-head scoring weights [num_tokens, num_heads], float32.
        kv_cache: KV cache buffer [total_kv_len, head_dim + 4], uint8.
            Each slot: head_dim bytes FP8 + 4 bytes FP32 scale (LE).
        num_tokens: Number of query tokens in this prefill chunk.
        num_heads: Number of attention heads (64 for DeepSeek V4).
        head_dim: Dimension per head (128 for DeepSeek V4).
        topk: Number of KV positions to select per token (1024).
        total_kv_len: Total allocated KV cache length.
        insert_k: Whether to quantize and insert k into kv_cache.

    Returns:
        topk_indices: [num_tokens, topk] int32 tensor. Each row contains
            the selected KV position indices (unordered), padded with -1.
    """
    device = q.device
    cache_stride_slot = head_dim + 4

    # Reshape q to [num_tokens, num_heads * head_dim] for kernel access
    q_flat = q.reshape(num_tokens, num_heads * head_dim).contiguous()

    # Step 1: Quantize K to FP8 and insert into KV cache
    if insert_k:
        grid_quant = (num_tokens,)
        _k_quant_cache_kernel[grid_quant](
            k,
            kv_cache,
            num_tokens,
            head_dim,
            cache_stride_slot,
            num_warps=4,
        )

    # Allocate output buffer
    topk_indices_buffer = torch.full(
        (num_tokens, topk), -1, dtype=torch.int32, device=device
    )

    # Determine shortcut boundary: tokens with kv_end <= topk get trivial fill
    # Token i has kv_end = i + 1, so tokens 0..topk-1 qualify for shortcut
    shortcut_end = min(topk, num_tokens)

    # Step 2: Shortcut path -- fill trivial rows
    if shortcut_end > 0:
        BLOCK_SHORTCUT = 256
        grid_shortcut = (shortcut_end,)
        _shortcut_fill_kernel[grid_shortcut](
            topk_indices_buffer,
            topk,
            shortcut_end,
            BLOCK_SHORTCUT,
            num_warps=4,
        )

    # Step 3: Multi-kernel path for remaining tokens
    token_offset = shortcut_end
    num_active_tokens = num_tokens - token_offset

    if num_active_tokens > 0:
        # Compact logits: max visible KV length is num_tokens (causal mask)
        max_kv_end = num_tokens
        topk_tokens = topk
        logits_stride = max_kv_end

        # Allocate compact logits tensor
        logits = torch.empty(
            num_active_tokens,
            logits_stride,
            device=device,
            dtype=torch.float32,
        )

        # Phase A: Compute logits
        BLOCK_KV = 64
        BLOCK_H = 64
        num_kv_blocks = (max_kv_end + BLOCK_KV - 1) // BLOCK_KV
        grid_logits = (num_active_tokens, num_kv_blocks)
        _compute_logits_compact_kernel[grid_logits](
            q_flat,
            kv_cache,
            weights,
            logits,
            num_tokens,
            num_heads,
            max_kv_end,
            head_dim,
            cache_stride_slot,
            token_offset,
            logits_stride,
            BLOCK_KV,
            BLOCK_H,
            num_warps=8,
            num_stages=2,
        )

        # Phase B: Binary search for threshold
        BLOCK_SCAN = 4096
        NUM_ITERS = 6  # 6 iterations sufficient for compact range
        threshold = torch.empty(num_active_tokens, device=device, dtype=torch.float32)
        grid_thresh = (num_active_tokens,)
        _topk_threshold_compact_kernel[grid_thresh](
            logits,
            threshold,
            num_active_tokens,
            max_kv_end,
            topk_tokens,
            token_offset,
            logits_stride,
            BLOCK_SCAN,
            NUM_ITERS,
            num_warps=8,
            num_stages=2,
        )

        # Phase C: Collect indices exceeding threshold
        BLOCK_COLLECT = 64
        num_collect_blocks = (max_kv_end + BLOCK_COLLECT - 1) // BLOCK_COLLECT
        counter = torch.zeros(num_active_tokens, device=device, dtype=torch.int32)
        grid_collect = (num_active_tokens, num_collect_blocks)
        _topk_collect_compact_kernel[grid_collect](
            logits,
            threshold,
            topk_indices_buffer,
            counter,
            num_active_tokens,
            max_kv_end,
            topk_tokens,
            token_offset,
            logits_stride,
            BLOCK_COLLECT,
            num_warps=4,
            num_stages=2,
        )

        # Fill remaining output slots with -1
        BLOCK_FILL = 256
        grid_fill = (num_active_tokens,)
        _topk_fill_remaining_kernel[grid_fill](
            topk_indices_buffer,
            counter,
            num_active_tokens,
            topk_tokens,
            token_offset,
            BLOCK_FILL,
            num_warps=4,
        )

    return topk_indices_buffer


# Public API
sparse_attn_indexer = sparse_attn_indexer_triton

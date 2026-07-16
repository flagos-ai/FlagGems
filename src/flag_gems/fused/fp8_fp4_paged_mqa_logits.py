"""Triton kernel for FP8/FP4 Paged Multi-Query Attention Logits.

Computes weighted attention logits from FP8-quantized queries against a paged
KV cache in FP8/FP4 format. Used in DeepSeek-V4 decode-phase inference.

The kernel uses tensor-core MMA (dot product) and an adaptive BLOCK_KV
selection based on context length to balance SM utilization across
different workload distributions.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _mqa_logits_kernel(
    Q_ptr,  # [total_rows, H * D] uint8 (FP8 bitcast)
    KV_data_ptr,  # [num_phys_blocks * BLOCK_SIZE, D] uint8 (FP8, flat paged)
    KV_scales_ptr,  # [num_phys_blocks * BLOCK_SIZE] float32
    Weights_ptr,  # [total_rows, H] float32
    Block_tables_ptr,  # [total_rows, max_blocks_per_seq] int32
    Output_ptr,  # [total_rows, max_model_len] float32
    Ctx_lens_ptr,  # [total_rows] int32
    total_rows,
    max_ctx,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    max_model_len,
    block_size: tl.constexpr,
    max_blocks_per_seq,
    num_phys_blocks,
    stride_q_row,
    stride_kv_flat,
    stride_bt_row,
    stride_out_row,
    stride_w_row,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    """Per-tile kernel: each program processes one BLOCK_KV tile for one row."""
    kv_block = tl.program_id(0)
    row_idx = tl.program_id(1)

    if row_idx >= total_rows:
        return

    ctx_len = tl.load(Ctx_lens_ptr + row_idx)
    kv_start = kv_block * BLOCK_KV
    if kv_start >= ctx_len:
        return

    q_row_base = Q_ptr + row_idx * stride_q_row
    w_row_base = Weights_ptr + row_idx * stride_w_row
    bt_row_base = Block_tables_ptr + row_idx * stride_bt_row
    out_row_base = Output_ptr + row_idx * stride_out_row

    h_ids = tl.arange(0, num_heads)
    d_ids = tl.arange(0, BLOCK_D)

    # Pre-load Q as FP8: [num_heads, head_dim]
    q_offsets = h_ids[:, None] * head_dim + d_ids[None, :]
    q_u8 = tl.load(q_row_base + q_offsets)
    q_fp8 = q_u8.to(tl.float8e4nv, bitcast=True)

    # Pre-load weights: [num_heads] float32
    w_all = tl.load(w_row_base + tl.arange(0, num_heads))

    end_pos = tl.minimum(kv_start + BLOCK_KV, ctx_len)
    first_lb = kv_start // block_size

    p_ids = tl.arange(0, block_size)

    for blk_idx in range(NUM_BLOCKS):
        lb = first_lb + blk_idx
        logical_base = lb * block_size
        if logical_base < end_pos:
            # Block-table lookup for physical block index
            phys_block = tl.load(bt_row_base + lb)
            phys_block = tl.maximum(phys_block, 0)
            phys_block = tl.minimum(phys_block, num_phys_blocks - 1)
            flat_base = phys_block * block_size

            # Coalesced KV load: [block_size, head_dim] as FP8
            kv_offsets = (flat_base + p_ids[:, None]) * stride_kv_flat + d_ids[None, :]
            kv_u8 = tl.load(KV_data_ptr + kv_offsets)
            kv_fp8 = kv_u8.to(tl.float8e4nv, bitcast=True)

            # Tensor-core MMA: Q[H, D] @ KV[block_size, D]^T -> [H, block_size]
            dots = tl.dot(q_fp8, tl.trans(kv_fp8))

            # Coalesced scale load: [block_size] float32
            scale_tile = tl.load(KV_scales_ptr + flat_base + p_ids)

            # Fused scale, relu, weight, reduce over heads
            scores = tl.maximum(dots * scale_tile[None, :], 0.0)
            weighted = scores * w_all[:, None]
            output_tile = tl.sum(weighted, axis=0)

            pos_ids = logical_base + p_ids
            valid_mask = pos_ids < end_pos
            tl.store(out_row_base + pos_ids, output_tile, mask=valid_mask)


def _preprocess_kv_cache(kv_cache, block_tables, context_lens, total_rows, next_n_val):
    """Reshape paged KV cache from [num_blocks, block_size, 1, D+4] uint8
    into flat data [flat_size, D] and scales [flat_size] arrays.
    """
    num_phys_blocks = kv_cache.shape[0]
    block_size = kv_cache.shape[1]
    D = kv_cache.shape[3] - 4

    flat_size = num_phys_blocks * block_size
    block_stride = block_size * (D + 4)

    kv_flat = kv_cache.reshape(num_phys_blocks, block_stride)

    kv_data = kv_flat[:, : block_size * D].reshape(num_phys_blocks, block_size, D)
    kv_data = kv_data.reshape(flat_size, D).contiguous()

    # Extract per-token FP32 scales from the trailing 4 bytes per token
    scale_bytes = kv_flat[:, block_size * D :].reshape(num_phys_blocks, block_size, 4)
    kv_scales = (
        scale_bytes.contiguous()
        .reshape(flat_size, 4)
        .view(torch.float32)
        .reshape(flat_size)
        .contiguous()
    )

    if block_tables.dim() == 2:
        B = block_tables.shape[0]
        block_tables_expanded = (
            block_tables.unsqueeze(1)
            .expand(B, next_n_val, -1)
            .reshape(total_rows, -1)
            .contiguous()
            .to(torch.int32)
        )
    else:
        block_tables_expanded = block_tables.contiguous().to(torch.int32)

    max_ctx = int(context_lens.max().item())

    return kv_data, kv_scales, block_tables_expanded, max_ctx


def _select_block_kv(max_ctx, block_size):
    """Adaptive BLOCK_KV selection based on context length.

    Uses 4 levels to balance SM utilization across production context-length
    distribution (1k through 64k).
    """
    if max_ctx <= 2048:
        block_kv = 256
    elif max_ctx <= 4096:
        block_kv = 512
    elif max_ctx <= 8192:
        block_kv = 1024
    else:
        block_kv = 2048
    num_blocks = block_kv // block_size
    return block_kv, num_blocks


def fp8_fp4_paged_mqa_logits(
    q,
    kv_cache,
    weights,
    context_lens,
    block_tables,
    schedule_metadata,
    max_model_len,
    clean_logits=False,
):
    """Compute paged MQA logits from FP8 queries against FP8/FP4 KV cache.

    Args:
        q: Tuple of (q_values [B, next_n, H, D] float8_e4m3fn, q_scale).
        kv_cache: [num_blocks, block_size, 1, D+4] uint8 paged KV cache.
        weights: [B*next_n, H] float32 per-head weights.
        context_lens: [B] or [B, next_n] int32 context lengths.
        block_tables: [B, max_blocks] int32 block table mapping.
        schedule_metadata: Metadata from get_paged_mqa_logits_metadata (unused
            by Triton kernel, kept for API compatibility with vLLM).
        max_model_len: Maximum model sequence length.
        clean_logits: If True, initialize output with -inf instead of 0.

    Returns:
        Logits tensor [total_rows, max_model_len] float32.
    """
    logger.debug("GEMS FP8_FP4_PAGED_MQA_LOGITS")

    q_values, q_scale = q

    if q_values.dim() == 3:
        q_values = q_values.unsqueeze(1)

    B, next_n_val, H, D = q_values.shape
    total_rows = B * next_n_val

    block_size = kv_cache.shape[1]
    head_dim = kv_cache.shape[3] - 4
    assert head_dim == D

    if context_lens.dim() == 2:
        ctx_lens_flat = (
            context_lens.reshape(-1)[:total_rows].contiguous().to(torch.int32)
        )
    else:
        ctx_lens_flat = (
            context_lens.repeat_interleave(next_n_val).contiguous().to(torch.int32)
        )

    kv_data, kv_scales, block_tables_expanded, max_ctx = _preprocess_kv_cache(
        kv_cache, block_tables, ctx_lens_flat, total_rows, next_n_val
    )

    q_flat = q_values.reshape(total_rows, H, D).contiguous()
    q_u8 = q_flat.view(torch.uint8).reshape(total_rows, H * D)

    logits = torch.full(
        (total_rows, max_model_len),
        float("-inf") if clean_logits else 0.0,
        device=q_values.device,
        dtype=torch.float32,
    )

    BLOCK_D = 128
    BLOCK_KV, NUM_BLOCKS = _select_block_kv(max_ctx, block_size)

    num_phys_blocks = kv_cache.shape[0]
    max_blocks_per_seq = block_tables_expanded.shape[1]

    grid = (triton.cdiv(max_ctx, BLOCK_KV), total_rows)
    _mqa_logits_kernel[grid](
        q_u8,
        kv_data,
        kv_scales,
        weights,
        block_tables_expanded,
        logits,
        ctx_lens_flat,
        total_rows=total_rows,
        max_ctx=max_ctx,
        num_heads=H,
        head_dim=D,
        max_model_len=max_model_len,
        block_size=block_size,
        max_blocks_per_seq=max_blocks_per_seq,
        num_phys_blocks=num_phys_blocks,
        stride_q_row=H * D,
        stride_kv_flat=D,
        stride_bt_row=max_blocks_per_seq,
        stride_out_row=max_model_len,
        stride_w_row=H,
        BLOCK_KV=BLOCK_KV,
        BLOCK_D=BLOCK_D,
        NUM_BLOCKS=NUM_BLOCKS,
    )

    return logits

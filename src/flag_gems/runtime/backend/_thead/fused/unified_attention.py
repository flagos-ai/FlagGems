"""Optimized Triton paged attention kernel (kgen_v2).

Based on kgen's efficient kernel design:
1. 3D grid with direct sequence indexing (no binary search)
2. GQA batching: multiple query heads share one KV load
3. K loaded transposed for natural dot-product layout
4. Separate decode/prefill tile configs
5. Early causal exit to skip fully-masked KV tiles

Extended with:
- Sliding window support
- 3D parallel KV segment decode (for long sequences)
- reduce_segments kernel for segment reduction
- Full unified_attention wrapper matching vLLM interface
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_attn_fwd(
    # Pointers
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    # Segment output pointers (used in 3D mode)
    segm_output_ptr, segm_max_ptr, segm_expsum_ptr,
    # Block table
    block_table_ptr,
    # Sequence info
    cu_seqlens_q_ptr, seqused_k_ptr,
    # Dimensions
    num_query_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    head_size: tl.constexpr,
    kv_block_size: tl.constexpr,
    max_blocks_per_seq: tl.constexpr,
    # Scalars
    softmax_scale,
    # Constexpr controls
    USE_SOFTCAP: tl.constexpr,
    softcap,
    causal: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    # Strides for Q: [total_tokens, num_query_heads, head_size]
    stride_qt, stride_qh, stride_qd,
    # Strides for K: [num_blocks, block_size, num_kv_heads, head_size]
    stride_kb, stride_ks, stride_kh, stride_kd,
    # Strides for V: [num_blocks, block_size, num_kv_heads, head_size]
    stride_vb, stride_vs, stride_vh, stride_vd,
    # Strides for Out: [total_tokens, num_query_heads, head_size]
    stride_ot, stride_oh, stride_od,
    # Stride for block_table: [num_seqs, max_blocks_per_seq]
    stride_bt_seq, stride_bt_blk,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_Q_POS: tl.constexpr,
    TILE_KV: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    # 3D parallel segments
    IS_3D: tl.constexpr,
    NUM_SEGMENTS: tl.constexpr,
):
    """GQA-batched paged attention with sliding window and 3D segment support."""
    # Program IDs
    pid_q_block = tl.program_id(0)
    pid_kv_head = tl.program_id(1)
    pid_composite = tl.program_id(2)

    # Decompose composite ID into seq and segment
    if IS_3D:
        pid_seq = pid_composite // NUM_SEGMENTS
        segment_id = pid_composite % NUM_SEGMENTS
    else:
        pid_seq = pid_composite
        segment_id = 0

    # Load sequence boundaries
    q_start = tl.load(cu_seqlens_q_ptr + pid_seq)
    q_end = tl.load(cu_seqlens_q_ptr + pid_seq + 1)
    query_len = q_end - q_start
    seq_len = tl.load(seqused_k_ptr + pid_seq)
    context_len = seq_len - query_len

    # Early exit if this Q block is out of range
    q_block_start_pos = pid_q_block * BLOCK_Q_POS
    if q_block_start_pos >= query_len:
        return

    # Build BLOCK_M-dimensional index packing q_positions and heads together
    offs_m = tl.arange(0, BLOCK_M)
    query_pos_local = offs_m // num_queries_per_kv
    head_in_group = offs_m % num_queries_per_kv

    query_pos = q_block_start_pos + query_pos_local
    query_head_idx = pid_kv_head * num_queries_per_kv + head_in_group

    # Mask for valid queries
    q_mask = query_pos < query_len

    # Absolute positions for causal masking
    query_abs_pos = context_len + query_pos

    # Compute KV tile range for this segment (3D mode splits KV across segments)
    total_kv_blocks = (seq_len + kv_block_size - 1) // kv_block_size
    total_kv_tiles = (seq_len + TILE_KV - 1) // TILE_KV
    if IS_3D:
        tiles_per_segment = (total_kv_tiles + NUM_SEGMENTS - 1) // NUM_SEGMENTS
        kv_tile_start = segment_id * tiles_per_segment
        kv_tile_end = tl.minimum((segment_id + 1) * tiles_per_segment, total_kv_tiles)
    else:
        kv_tile_start = 0
        kv_tile_end = total_kv_tiles

    # Load Q: [BLOCK_M, HEAD_SIZE_PADDED]
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    d_mask = offs_d < head_size
    q_ptrs = Q_ptr + (q_start + query_pos)[:, None] * stride_qt +              query_head_idx[:, None] * stride_qh +              offs_d[None, :] * stride_qd
    Q = tl.load(q_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0)
    Q = (Q * softmax_scale).to(Q_ptr.dtype.element_ty)

    # Initialize accumulators
    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 0.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # Block table for this sequence
    bt_base = pid_seq * stride_bt_seq

    # Compute effective loop bounds (causal early exit + sliding window)
    if causal:
        max_q_abs = context_len + tl.minimum(q_block_start_pos + BLOCK_Q_POS - 1, query_len - 1)
        kv_upper = max_q_abs + 1
        causal_tiles = (kv_upper + TILE_KV - 1) // TILE_KV
        kv_tile_end = tl.minimum(kv_tile_end, causal_tiles)

    if SLIDING_WINDOW > 0:
        min_q_abs = context_len + q_block_start_pos
        sw_lower = min_q_abs - SLIDING_WINDOW + 1
        if sw_lower > 0:
            sw_tile_start = sw_lower // TILE_KV
            kv_tile_start = tl.maximum(kv_tile_start, sw_tile_start)

    # Iterate over KV tiles
    for tile_idx in range(kv_tile_start, kv_tile_end):
        kv_start = tile_idx * TILE_KV
        kv_offsets = kv_start + tl.arange(0, TILE_KV)

        # Mask for valid KV positions
        tile_mask = kv_offsets < seq_len

        # Map KV indices to physical blocks
        block_ids = kv_offsets // kv_block_size
        slot_offsets = kv_offsets % kv_block_size
        phys_blocks = tl.load(block_table_ptr + bt_base + block_ids * stride_bt_blk,
                              mask=tile_mask, other=0)

        # Load K transposed: [HEAD_SIZE_PADDED, TILE_KV]
        k_ptrs = K_ptr + phys_blocks[None, :] * stride_kb +                  slot_offsets[None, :] * stride_ks +                  pid_kv_head * stride_kh +                  offs_d[:, None] * stride_kd
        K_T = tl.load(k_ptrs, mask=tile_mask[None, :] & d_mask[:, None], other=0.0)

        # Compute attention scores: [BLOCK_M, TILE_KV]
        S = tl.dot(Q, K_T)

        # Apply softcap
        if USE_SOFTCAP:
            t = S / softcap
            S = softcap * (2.0 / (1.0 + tl.exp(-2.0 * t)) - 1.0)

        # Apply causal mask
        if causal:
            causal_mask = kv_offsets[None, :] <= query_abs_pos[:, None]
            S = tl.where(causal_mask, S, float("-inf"))

        # Apply sliding window per-element mask
        if SLIDING_WINDOW > 0:
            sw_elem_mask = kv_offsets[None, :] >= (query_abs_pos[:, None] - SLIDING_WINDOW + 1)
            S = tl.where(sw_elem_mask, S, float("-inf"))

        # Mask invalid KV and Q positions
        S = tl.where(tile_mask[None, :], S, float("-inf"))
        S = tl.where(q_mask[:, None], S, float("-inf"))

        # Online softmax
        m_j = tl.max(S, axis=1)
        m_new = tl.maximum(M, m_j)
        m_safe = tl.where(m_new > float("-inf"), m_new, 0.0)
        alpha = tl.exp(M - m_safe)
        P = tl.exp(S - m_safe[:, None])
        L_new = alpha * L + tl.sum(P, axis=1)

        # Load V: [TILE_KV, HEAD_SIZE_PADDED]
        v_ptrs = V_ptr + phys_blocks[:, None] * stride_vb +                  slot_offsets[:, None] * stride_vs +                  pid_kv_head * stride_vh +                  offs_d[None, :] * stride_vd
        V = tl.load(v_ptrs, mask=tile_mask[:, None] & d_mask[None, :], other=0.0)

        # Update accumulator
        acc = acc * alpha[:, None] + tl.dot(P.to(V.dtype), V)

        M = m_safe
        L = L_new

    # Final normalization
    L_safe = tl.where(L == 0.0, 1.0, L)
    acc = acc / L_safe[:, None]

    if IS_3D:
        # Store per-segment results for later reduction
        # segm_output: [total_tokens, num_query_heads, NUM_SEGMENTS, HEAD_SIZE_PADDED]
        # segm_max/expsum: [total_tokens, num_query_heads, NUM_SEGMENTS]
        token_indices = q_start + query_pos
        segm_out_base = token_indices[:, None] * (num_query_heads * NUM_SEGMENTS * HEAD_SIZE_PADDED) +                         query_head_idx[:, None] * (NUM_SEGMENTS * HEAD_SIZE_PADDED) +                         segment_id * HEAD_SIZE_PADDED +                         offs_d[None, :]
        tl.store(segm_output_ptr + segm_out_base, acc,
                 mask=q_mask[:, None] & d_mask[None, :])

        segm_scalar_base = token_indices * (num_query_heads * NUM_SEGMENTS) +                            query_head_idx * NUM_SEGMENTS + segment_id
        tl.store(segm_max_ptr + segm_scalar_base, M, mask=q_mask)
        tl.store(segm_expsum_ptr + segm_scalar_base, L, mask=q_mask)
    else:
        # Write output directly
        out_ptrs = Out_ptr + (q_start + query_pos)[:, None] * stride_ot +                    query_head_idx[:, None] * stride_oh +                    offs_d[None, :] * stride_od
        tl.store(out_ptrs, acc.to(Out_ptr.dtype.element_ty),
                 mask=q_mask[:, None] & d_mask[None, :])


@triton.jit
def _reduce_segments(
    output_ptr,
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
    seq_lens_ptr,
    num_seqs,
    num_query_heads: tl.constexpr,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    query_start_len_ptr,
    NUM_SEGMENTS: tl.constexpr,
):
    """Reduce partial segment results into final output."""
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    # Compute number of active segments for this sequence
    # Find which sequence this token belongs to (simple scan since decode has 1 token/seq)
    seq_len = tl.load(seq_lens_ptr + query_token_idx)  # works for decode (1 token per seq)
    total_kv_tiles = (seq_len + TILE_SIZE - 1) // TILE_SIZE
    tiles_per_segment = (total_kv_tiles + NUM_SEGMENTS - 1) // NUM_SEGMENTS
    act_num_segments = (total_kv_tiles + tiles_per_segment - 1) // tiles_per_segment

    segm_mask = tl.arange(0, NUM_SEGMENTS) < tl.full(
        [NUM_SEGMENTS], act_num_segments, dtype=tl.int32
    )
    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)

    # Load segment max values
    segm_offset = (
        query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS)
        + query_head_idx * NUM_SEGMENTS
        + tl.arange(0, NUM_SEGMENTS)
    )
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    overall_max = tl.max(segm_max)

    # Load and rescale expsums
    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    # Load segment outputs and reduce
    segm_output_offset = (
        query_token_idx.to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS * HEAD_SIZE_PADDED)
        + query_head_idx * (NUM_SEGMENTS * HEAD_SIZE_PADDED)
        + tl.arange(0, NUM_SEGMENTS)[:, None] * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    segm_output = tl.load(
        segm_output_ptr + segm_output_offset,
        mask=segm_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    segm_output *= tl.exp(segm_max - overall_max)[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    # Store final output
    output_offset = (
        query_token_idx * output_stride_0
        + query_head_idx * output_stride_1
        + tl.arange(0, HEAD_SIZE_PADDED)
    )
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)


# ---- Python wrapper ----


def unified_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    seq_threshold_3D=None,
    num_par_softmax_segments=None,
    softmax_segm_output=None,
    softmax_segm_max=None,
    softmax_segm_expsum=None,
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    sinks=None,
    mm_prefix_range=None,
    use_alibi_sqrt=False,
    kv_quant_mode=0,
    k_scale_cache=None,
    v_scale_cache=None,
    chunk_lookback=-1,
):
    """Full-featured paged attention wrapper with kgen's efficient kernel.

    Supports: GQA, causal, softcap, sliding window, 3D parallel segments.
    Not supported (asserts): alibi, FP8, sinks, qq_bias, mm_prefix, kv_quant.
    """
    assert causal, "Only causal attention is supported"
    assert q_descale is None, "Q scales not supported"
    assert alibi_slopes is None, "Alibi slopes not supported in kgen_v2"
    assert output_scale is None, "FP8 output scale not supported in kgen_v2"
    assert qq_bias is None, "QQ bias not supported in kgen_v2"
    assert sinks is None, "Sinks not supported in kgen_v2"
    assert mm_prefix_range is None, "MM prefix not supported in kgen_v2"
    assert kv_quant_mode == 0, "KV quantization not supported in kgen_v2"
    assert k_scale_cache is None, "KV scale cache not supported in kgen_v2"
    assert v_scale_cache is None, "KV scale cache not supported in kgen_v2"

    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    head_size = q.shape[2]
    num_kv_heads = k.shape[2]
    kv_block_size = k.shape[1]
    max_blocks_per_seq = block_table.shape[1]
    num_queries_per_kv = num_query_heads // num_kv_heads

    HEAD_SIZE_PADDED = triton.next_power_of_2(head_size)

    # Sliding window
    sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0

    # Softcap
    USE_SOFTCAP = softcap is not None and softcap > 0.0
    softcap_val = float(softcap) if USE_SOFTCAP else 0.0

    # Determine 3D mode
    use_3d = not (
        seq_threshold_3D is None
        or num_par_softmax_segments is None
        or softmax_segm_output is None
        or softmax_segm_max is None
        or softmax_segm_expsum is None
        or max_seqlen_q > 1
        or num_seqs > seq_threshold_3D
    )

    num_segments = num_par_softmax_segments if use_3d else 1

    # ---------- Tile size selection ----------
    if max_seqlen_q == 1:
        # === Decode path ===
        if num_queries_per_kv <= 16:
            BLOCK_M = 16
        else:
            BLOCK_M = triton.next_power_of_2(num_queries_per_kv)
        BLOCK_Q_POS = BLOCK_M // num_queries_per_kv
        TILE_KV = 64
        num_warps = 4
        num_stages = 2
    else:
        # === Prefill path ===
        if head_size <= 128:
            BLOCK_M = 64
            BLOCK_Q_POS = BLOCK_M // num_queries_per_kv
            TILE_KV = 64
            num_warps = 4
            num_stages = 2
        else:
            if num_queries_per_kv <= 16:
                BLOCK_M = 16
            else:
                BLOCK_M = triton.next_power_of_2(num_queries_per_kv)
            BLOCK_Q_POS = BLOCK_M // num_queries_per_kv
            TILE_KV = 32
            num_warps = 4
            num_stages = 2

    # Grid: (max_q_blocks_per_seq, num_kv_heads, num_seqs * num_segments)
    max_q_blocks = (max_seqlen_q + BLOCK_Q_POS - 1) // BLOCK_Q_POS
    grid = (max_q_blocks, num_kv_heads, num_seqs * num_segments)

    # Segment buffers
    segm_output_ptr = softmax_segm_output if use_3d else out
    segm_max_ptr = softmax_segm_max if use_3d else out
    segm_expsum_ptr = softmax_segm_expsum if use_3d else out

    _paged_attn_fwd[grid](
        q, k, v, out,
        segm_output_ptr, segm_max_ptr, segm_expsum_ptr,
        block_table,
        cu_seqlens_q, seqused_k,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        num_queries_per_kv=num_queries_per_kv,
        head_size=head_size,
        kv_block_size=kv_block_size,
        max_blocks_per_seq=max_blocks_per_seq,
        softmax_scale=softmax_scale,
        USE_SOFTCAP=USE_SOFTCAP,
        softcap=softcap_val,
        causal=causal,
        SLIDING_WINDOW=sliding_window_val,
        stride_qt=q.stride(0), stride_qh=q.stride(1), stride_qd=q.stride(2),
        stride_kb=k.stride(0), stride_ks=k.stride(1), stride_kh=k.stride(2), stride_kd=k.stride(3),
        stride_vb=v.stride(0), stride_vs=v.stride(1), stride_vh=v.stride(2), stride_vd=v.stride(3),
        stride_ot=out.stride(0), stride_oh=out.stride(1), stride_od=out.stride(2),
        stride_bt_seq=block_table.stride(0), stride_bt_blk=block_table.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_Q_POS=BLOCK_Q_POS,
        TILE_KV=TILE_KV,
        HEAD_SIZE_PADDED=HEAD_SIZE_PADDED,
        IS_3D=use_3d,
        NUM_SEGMENTS=num_segments,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    # Reduce segments for 3D mode
    if use_3d:
        _reduce_segments[(q.shape[0], num_query_heads)](
            output_ptr=out,
            segm_output_ptr=softmax_segm_output,
            segm_max_ptr=softmax_segm_max,
            segm_expsum_ptr=softmax_segm_expsum,
            seq_lens_ptr=seqused_k,
            num_seqs=num_seqs,
            num_query_heads=num_query_heads,
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            TILE_SIZE=TILE_KV,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=HEAD_SIZE_PADDED,
            query_start_len_ptr=cu_seqlens_q,
            NUM_SEGMENTS=num_par_softmax_segments,
        )

    return out

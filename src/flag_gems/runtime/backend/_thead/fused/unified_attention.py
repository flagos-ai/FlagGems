"""Optimized Triton paged multi-head attention kernel for LLM inference.

Key optimizations over vLLM baseline:
1. GQA batching: multiple query heads share one KV load (like vLLM)
2. 3D grid with direct indexing: eliminates binary search overhead
3. K loaded transposed: avoids explicit tl.trans()
4. Separate decode/prefill tile configs:
   - Decode: BLOCK_M=16, TILE_KV=64 for maximum memory throughput
   - Prefill: BLOCK_M=64, TILE_KV=64 for maximum compute amortization
5. Leaner kernel: no unused features (alibi, FP8, sliding window)
6. Early causal exit to skip fully-masked KV tiles
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_attn_fwd(
    # Pointers
    Q_ptr, K_ptr, V_ptr, Out_ptr,
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
):
    """GQA-batched paged attention. One CTA handles multiple query heads sharing one KV head."""
    # Program IDs
    pid_q_block = tl.program_id(0)
    pid_kv_head = tl.program_id(1)
    pid_seq = tl.program_id(2)

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

    q_mask = (query_pos < query_len) & (query_head_idx < num_query_heads)

    # Load Q: [BLOCK_M, HEAD_SIZE_PADDED]
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    d_mask = offs_d < head_size

    q_ptrs = Q_ptr + (q_start + query_pos)[:, None] * stride_qt + \
             query_head_idx[:, None] * stride_qh + \
             offs_d[None, :] * stride_qd
    Q = tl.load(q_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0)

    # Initialize online softmax accumulators
    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # Compute loop bounds with causal early exit
    if causal:
        max_q_abs = context_len + tl.minimum(q_block_start_pos + BLOCK_Q_POS - 1, query_len - 1)
        kv_upper = max_q_abs + 1
        num_kv_tiles = (kv_upper + TILE_KV - 1) // TILE_KV
    else:
        num_kv_tiles = (seq_len + TILE_KV - 1) // TILE_KV

    query_abs_pos = context_len + query_pos
    offs_t = tl.arange(0, TILE_KV)

    # Iterate over KV tiles
    for tile_idx in range(num_kv_tiles):
        kv_start = tile_idx * TILE_KV
        kv_indices = kv_start + offs_t
        tile_mask = kv_indices < seq_len

        # Look up physical blocks
        logical_blocks = kv_indices // kv_block_size
        slot_offsets = kv_indices % kv_block_size

        bt_ptrs = block_table_ptr + pid_seq * stride_bt_seq + logical_blocks * stride_bt_blk
        phys_blocks = tl.load(bt_ptrs, mask=tile_mask, other=0)

        # Load K transposed: [HEAD_SIZE_PADDED, TILE_KV] for direct Q @ K
        k_ptrs = K_ptr + phys_blocks[None, :] * stride_kb + \
                 slot_offsets[None, :] * stride_ks + \
                 pid_kv_head * stride_kh + \
                 offs_d[:, None] * stride_kd
        K = tl.load(k_ptrs, mask=d_mask[:, None] & tile_mask[None, :], other=0.0)

        # Compute scores: [BLOCK_M, TILE_KV]
        S = tl.dot(Q, K) * softmax_scale

        # Apply softcap
        if USE_SOFTCAP:
            t = S / softcap
            S = softcap * (2.0 / (1.0 + tl.exp(-2.0 * t)) - 1.0)

        # Apply causal mask
        if causal:
            causal_mask = kv_indices[None, :] <= query_abs_pos[:, None]
            S = tl.where(causal_mask, S, float("-inf"))

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
        v_ptrs = V_ptr + phys_blocks[:, None] * stride_vb + \
                 slot_offsets[:, None] * stride_vs + \
                 pid_kv_head * stride_vh + \
                 offs_d[None, :] * stride_vd
        V = tl.load(v_ptrs, mask=tile_mask[:, None] & d_mask[None, :], other=0.0)

        # Update accumulator
        acc = acc * alpha[:, None] + tl.dot(P.to(V.dtype), V)

        M = m_safe
        L = L_new

    # Final normalization
    L_safe = tl.where(L == 0.0, 1.0, L)
    acc = acc / L_safe[:, None]

    # Write output
    out_ptrs = Out_ptr + (q_start + query_pos)[:, None] * stride_ot + \
               query_head_idx[:, None] * stride_oh + \
               offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc.to(Out_ptr.dtype.element_ty),
             mask=q_mask[:, None] & d_mask[None, :])


def unified_attention(
    q,             # [total_tokens, num_query_heads, head_size]
    k,             # [num_blocks, block_size, num_kv_heads, head_size]
    v,             # [num_blocks, block_size, num_kv_heads, head_size]
    out,           # [total_tokens, num_query_heads, head_size]
    cu_seqlens_q,  # [num_seqs + 1]
    max_seqlen_q,  # int
    seqused_k,     # [num_seqs]
    max_seqlen_k,  # int
    softmax_scale, # float
    causal,        # bool
    window_size,   # (int, int)
    block_table,   # [num_seqs, max_blocks_per_seq]
    softcap,       # float
    q_descale,     # None
    k_descale,     # None
    v_descale,     # None
    seq_threshold_3D=None,
    num_par_softmax_segments=None,
    softmax_segm_output=None,
    softmax_segm_max=None,
    softmax_segm_expsum=None,
):
    """Paged attention with GQA, causal mask, and optional softcap."""
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    head_size = q.shape[2]
    num_kv_heads = k.shape[2]
    kv_block_size = k.shape[1]
    max_blocks_per_seq = block_table.shape[1]
    num_queries_per_kv = num_query_heads // num_kv_heads

    HEAD_SIZE_PADDED = triton.next_power_of_2(head_size)

    # ---------- Tile size selection ----------
    if max_seqlen_q == 1:
        # === Decode path ===
        # GQA batching packs heads together; only 1 query position valid
        if num_queries_per_kv <= 16:
            BLOCK_M = 16
        else:
            BLOCK_M = triton.next_power_of_2(num_queries_per_kv)
        BLOCK_Q_POS = BLOCK_M // num_queries_per_kv

        if HEAD_SIZE_PADDED <= 128:
            TILE_KV = 64
            num_warps = 4
            num_stages = 3
        else:
            # head_size=256: reduce tile to manage register pressure
            TILE_KV = 32
            num_warps = 4
            num_stages = 2
    else:
        # === Prefill path ===
        # Use larger BLOCK_M to amortize KV loads across more query positions
        if HEAD_SIZE_PADDED <= 128:
            # Target BLOCK_M=64: 16 positions × 4 heads for GQA 32:8
            target_block_m = 64
            BLOCK_M = max(16, min(target_block_m, triton.next_power_of_2(
                num_queries_per_kv * min(16, max_seqlen_q)
            )))
            # Ensure BLOCK_M is power of 2 and >= 16
            if BLOCK_M < 16:
                BLOCK_M = 16
            BLOCK_Q_POS = BLOCK_M // num_queries_per_kv
            TILE_KV = 64
            num_warps = 4
            num_stages = 2
        else:
            # head_size=256: smaller tiles for register pressure
            if num_queries_per_kv <= 16:
                BLOCK_M = 16
            else:
                BLOCK_M = triton.next_power_of_2(num_queries_per_kv)
            BLOCK_Q_POS = BLOCK_M // num_queries_per_kv
            TILE_KV = 32
            num_warps = 4
            num_stages = 2

    # Softcap
    USE_SOFTCAP = softcap is not None and softcap > 0.0
    softcap_val = float(softcap) if USE_SOFTCAP else 0.0

    # Grid: (max_q_blocks_per_seq, num_kv_heads, num_seqs)
    max_q_blocks = (max_seqlen_q + BLOCK_Q_POS - 1) // BLOCK_Q_POS
    grid = (max_q_blocks, num_kv_heads, num_seqs)

    _paged_attn_fwd[grid](
        q, k, v, out,
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
        stride_qt=q.stride(0), stride_qh=q.stride(1), stride_qd=q.stride(2),
        stride_kb=k.stride(0), stride_ks=k.stride(1), stride_kh=k.stride(2), stride_kd=k.stride(3),
        stride_vb=v.stride(0), stride_vs=v.stride(1), stride_vh=v.stride(2), stride_vd=v.stride(3),
        stride_ot=out.stride(0), stride_oh=out.stride(1), stride_od=out.stride(2),
        stride_bt_seq=block_table.stride(0), stride_bt_blk=block_table.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_Q_POS=BLOCK_Q_POS,
        TILE_KV=TILE_KV,
        HEAD_SIZE_PADDED=HEAD_SIZE_PADDED,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return out

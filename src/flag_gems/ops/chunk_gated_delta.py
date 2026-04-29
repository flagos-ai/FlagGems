import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_C': 64, 'BLOCK_SIZE_D': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_C': 128, 'BLOCK_SIZE_D': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE_C': 64, 'BLOCK_SIZE_D': 128}, num_warps=8, num_stages=3),
    ],
    key=['SEQ_LEN', 'DIM'],
)
@triton.jit
def extreme_chunk_gated_delta_fwd_kernel(
    q_ptr, k_ptr, v_ptr, beta_ptr, out_ptr, state_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_betab, stride_betah, stride_betas,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_stateb, stride_stateh, stride_stated1, stride_stated2,
    BATCH, HEADS, SEQ_LEN, DIM, NUM_CHUNKS,
    BLOCK_SIZE_C: tl.constexpr, 
    BLOCK_SIZE_D: tl.constexpr, 
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    q_head_ptr = q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    k_head_ptr = k_ptr + batch_idx * stride_kb + head_idx * stride_kh
    v_head_ptr = v_ptr + batch_idx * stride_vb + head_idx * stride_vh
    beta_head_ptr = beta_ptr + batch_idx * stride_betab + head_idx * stride_betah
    o_head_ptr = out_ptr + batch_idx * stride_ob + head_idx * stride_oh
    state_head_ptr = state_ptr + batch_idx * stride_stateb + head_idx * stride_stateh

    offs_d1 = tl.arange(0, BLOCK_SIZE_D)
    offs_d2 = tl.arange(0, BLOCK_SIZE_D)
    state = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_D), dtype=tl.float32)

    offs_c = tl.arange(0, BLOCK_SIZE_C)
    causal_mask = offs_c[:, None] >= offs_c[None, :]


    for chunk_idx in range(NUM_CHUNKS):
        seq_start = chunk_idx * BLOCK_SIZE_C
        seq_offs = seq_start + offs_c
        
        seq_mask = seq_offs < SEQ_LEN
        dim_mask1 = offs_d1 < DIM
        dim_mask2 = offs_d2 < DIM

        q_ptrs = q_head_ptr + seq_offs[:, None] * stride_qs + offs_d1[None, :] * stride_qd
        k_ptrs = k_head_ptr + seq_offs[:, None] * stride_ks + offs_d2[None, :] * stride_kd
        v_ptrs = v_head_ptr + seq_offs[:, None] * stride_vs + offs_d2[None, :] * stride_vd
        beta_ptrs = beta_head_ptr + seq_offs * stride_betas
        o_ptrs = o_head_ptr + seq_offs[:, None] * stride_os + offs_d2[None, :] * stride_od


        q = tl.load(q_ptrs, mask=seq_mask[:, None] & dim_mask1[None, :], other=0.0)
        k = tl.load(k_ptrs, mask=seq_mask[:, None] & dim_mask2[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=seq_mask[:, None] & dim_mask2[None, :], other=0.0)
        beta = tl.load(beta_ptrs, mask=seq_mask, other=0.0).to(tl.float32)

        log_beta = tl.math.log(beta + 1e-6)
        cum_log_beta = tl.cumsum(log_beta, axis=0)
        decay_matrix = tl.math.exp(cum_log_beta[:, None] - cum_log_beta[None, :])
        decay_masked = tl.where(causal_mask, decay_matrix, 0.0)

        qk = tl.dot(q, tl.trans(k))
        attn_weights = (qk * decay_masked).to(q.dtype)
        out_intra = tl.dot(attn_weights, v)

        out_inter = tl.dot(q, state.to(q.dtype))

        out = out_intra + out_inter
        tl.store(o_ptrs, out.to(q_ptr.dtype.element_ty), mask=seq_mask[:, None] & dim_mask2[None, :])
        decay_to_end = tl.math.exp(cum_log_beta[-1] - cum_log_beta)
        k_decayed = (k * decay_to_end[:, None]).to(k.dtype)
        
        state = state * tl.math.exp(cum_log_beta[-1]) + tl.dot(tl.trans(k_decayed), v)

    state_ptrs = state_head_ptr + offs_d1[:, None] * stride_stated1 + offs_d2[None, :] * stride_stated2
    tl.store(state_ptrs, state.to(state_ptr.dtype.element_ty), mask=dim_mask1[:, None] & dim_mask2[None, :])

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_C': 64, 'BLOCK_SIZE_D': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_C': 128, 'BLOCK_SIZE_D': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE_C': 64, 'BLOCK_SIZE_D': 128}, num_warps=8, num_stages=3),
    ],
    key=['SEQ_LEN', 'DIM'],
)
@triton.jit
def extreme_chunk_gated_delta_bwd_kernel(
    q_ptr, k_ptr, v_ptr, beta_ptr, do_ptr, fwd_state_ptr,
    dq_ptr, dk_ptr, dv_ptr, dbeta_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_betab, stride_betah, stride_betas,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_fwd_stateb, stride_fwd_stateh, stride_fwd_statec, stride_fwd_stated1, stride_fwd_stated2,
    BATCH, HEADS, SEQ_LEN, DIM, NUM_CHUNKS,
    BLOCK_SIZE_C: tl.constexpr, 
    BLOCK_SIZE_D: tl.constexpr, 
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    offset_qkv = batch_idx * stride_qb + head_idx * stride_qh
    offset_beta = batch_idx * stride_betab + head_idx * stride_betah
    offset_do = batch_idx * stride_ob + head_idx * stride_oh
    offset_fwd_state = batch_idx * stride_fwd_stateb + head_idx * stride_fwd_stateh

    q_head_ptr = q_ptr + offset_qkv
    k_head_ptr = k_ptr + offset_qkv
    v_head_ptr = v_ptr + offset_qkv
    beta_head_ptr = beta_ptr + offset_beta
    do_head_ptr = do_ptr + offset_do

    dq_head_ptr = dq_ptr + offset_qkv
    dk_head_ptr = dk_ptr + offset_qkv
    dv_head_ptr = dv_ptr + offset_qkv
    dbeta_head_ptr = dbeta_ptr + offset_beta
    fwd_state_head_ptr = fwd_state_ptr + offset_fwd_state

    offs_d1 = tl.arange(0, BLOCK_SIZE_D)
    offs_d2 = tl.arange(0, BLOCK_SIZE_D)
    d_state = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_D), dtype=tl.float32)
    offs_c = tl.arange(0, BLOCK_SIZE_C)
    causal_mask = offs_c[:, None] >= offs_c[None, :]


    for chunk_idx in range(NUM_CHUNKS - 1, -1, -1):
        seq_start = chunk_idx * BLOCK_SIZE_C
        seq_offs = seq_start + offs_c
        
        seq_mask = seq_offs < SEQ_LEN
        dim_mask1 = offs_d1 < DIM
        dim_mask2 = offs_d2 < DIM

        q_ptrs = q_head_ptr + seq_offs[:, None] * stride_qs + offs_d1[None, :] * stride_qd
        k_ptrs = k_head_ptr + seq_offs[:, None] * stride_ks + offs_d2[None, :] * stride_kd
        v_ptrs = v_head_ptr + seq_offs[:, None] * stride_vs + offs_d2[None, :] * stride_vd
        do_ptrs = do_head_ptr + seq_offs[:, None] * stride_os + offs_d1[None, :] * stride_od
        beta_ptrs = beta_head_ptr + seq_offs * stride_betas

        q = tl.load(q_ptrs, mask=seq_mask[:, None] & dim_mask1[None, :], other=0.0)
        k = tl.load(k_ptrs, mask=seq_mask[:, None] & dim_mask2[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=seq_mask[:, None] & dim_mask2[None, :], other=0.0)
        do = tl.load(do_ptrs, mask=seq_mask[:, None] & dim_mask1[None, :], other=0.0)
        beta = tl.load(beta_ptrs, mask=seq_mask, other=0.0).to(tl.float32)

        log_beta = tl.math.log(beta + 1e-6)
        cum_log_beta = tl.cumsum(log_beta, axis=0)
        decay_matrix = tl.math.exp(cum_log_beta[:, None] - cum_log_beta[None, :])
        decay_masked = tl.where(causal_mask, decay_matrix, 0.0)
        decay_to_end = tl.math.exp(cum_log_beta[-1] - cum_log_beta)

        qk_t = tl.dot(q, tl.trans(k))
        attn_weights = (qk_t * decay_masked).to(q.dtype)


        attn_trans = tl.trans(attn_weights)
        dv_intra = tl.dot(attn_trans, do)
        
        k_decayed = (k * decay_to_end[:, None]).to(k.dtype)
        dv_inter = tl.dot(k_decayed, d_state.to(k.dtype))
        
        dv = dv_intra + dv_inter
        
        dv_store_ptrs = dv_head_ptr + seq_offs[:, None] * stride_vs + offs_d2[None, :] * stride_vd
        tl.store(dv_store_ptrs, dv.to(v_ptr.dtype.element_ty), mask=seq_mask[:, None] & dim_mask2[None, :])


        do_v_t = tl.dot(do, tl.trans(v))
        dq_attn = tl.where(causal_mask, do_v_t * decay_matrix, 0.0).to(q.dtype)
        dq_intra = tl.dot(dq_attn, k)
        
        fwd_state_ptrs = fwd_state_head_ptr + chunk_idx * stride_fwd_statec + offs_d1[:, None] * stride_fwd_stated1 + offs_d2[None, :] * stride_fwd_stated2
        fwd_state = tl.load(fwd_state_ptrs, mask=dim_mask1[:, None] & dim_mask2[None, :], other=0.0).to(do.dtype)
        
        dq_inter = tl.dot(do, tl.trans(fwd_state))
        
        dq = dq_intra + dq_inter
        
        dq_store_ptrs = dq_head_ptr + seq_offs[:, None] * stride_qs + offs_d1[None, :] * stride_qd
        tl.store(dq_store_ptrs, dq.to(q_ptr.dtype.element_ty), mask=seq_mask[:, None] & dim_mask1[None, :])


        dk_attn = tl.trans(dq_attn)
        dk_intra = tl.dot(dk_attn, q)
        
        v_decayed = (v * decay_to_end[:, None]).to(v.dtype)
        dk_inter = tl.dot(v_decayed, tl.trans(d_state).to(v.dtype))
        
        dk = dk_intra + dk_inter
        
        dk_store_ptrs = dk_head_ptr + seq_offs[:, None] * stride_ks + offs_d2[None, :] * stride_kd
        tl.store(dk_store_ptrs, dk.to(k_ptr.dtype.element_ty), mask=seq_mask[:, None] & dim_mask2[None, :])


        dbeta_inner = tl.sum(dq_attn * qk_t, axis=1) 
        dbeta = dbeta_inner * beta 
        
        dbeta_store_ptrs = dbeta_head_ptr + seq_offs * stride_betas
        tl.store(dbeta_store_ptrs, dbeta.to(beta_ptr.dtype.element_ty), mask=seq_mask)

        q_decayed = (q * decay_to_end[:, None]).to(q.dtype)
        d_state = d_state * tl.math.exp(cum_log_beta[-1]) + tl.dot(tl.trans(q_decayed), do)


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, q, k, v, beta):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        beta = beta.contiguous()

        BATCH, HEADS, SEQ_LEN, DIM = q.shape
        
        BLOCK_SIZE_C = 64  
        def next_power_of_2(n):
            n -= 1
            n |= n >> 1
            n |= n >> 2
            n |= n >> 4
            n |= n >> 8
            n |= n >> 16
            n += 1
            return max(n, 16)
            
        BLOCK_SIZE_D = next_power_of_2(DIM)
        NUM_CHUNKS = triton.cdiv(SEQ_LEN, BLOCK_SIZE_C)

        out = torch.empty_like(q)
        
        fwd_states = torch.zeros((BATCH, HEADS, NUM_CHUNKS, DIM, DIM), dtype=torch.float32, device=q.device)

        grid = (BATCH, HEADS)

        extreme_chunk_gated_delta_fwd_kernel[grid](
            q, k, v, beta, out, fwd_states,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            beta.stride(0), beta.stride(1), beta.stride(2),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            fwd_states.stride(0), fwd_states.stride(1), fwd_states.stride(3), fwd_states.stride(4),
            BATCH, HEADS, SEQ_LEN, DIM, NUM_CHUNKS,
            BLOCK_SIZE_C=BLOCK_SIZE_C,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
        )

        ctx.save_for_backward(q, k, v, beta, fwd_states)
        ctx.dims = (BATCH, HEADS, SEQ_LEN, DIM, NUM_CHUNKS)
        ctx.blocks = (BLOCK_SIZE_C, BLOCK_SIZE_D)

        return out

    @staticmethod
    def backward(ctx, do):
        do = do.contiguous()
        q, k, v, beta, fwd_states = ctx.saved_tensors
        BATCH, HEADS, SEQ_LEN, DIM, NUM_CHUNKS = ctx.dims
        BLOCK_SIZE_C, BLOCK_SIZE_D = ctx.blocks
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dbeta = torch.empty_like(beta)

        grid = (BATCH, HEADS)

        extreme_chunk_gated_delta_bwd_kernel[grid](
            q, k, v, beta, do, fwd_states,
            dq, dk, dv, dbeta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            beta.stride(0), beta.stride(1), beta.stride(2),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            fwd_states.stride(0), fwd_states.stride(1), fwd_states.stride(2), fwd_states.stride(3), fwd_states.stride(4),
            BATCH, HEADS, SEQ_LEN, DIM, NUM_CHUNKS,
            BLOCK_SIZE_C=BLOCK_SIZE_C,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
        )

        return dq, dk, dv, dbeta

class FlagOS_ChunkGatedDelta(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, beta):
        """
        ግብዓቶች:
            q:    (Batch, Heads, Sequence_Length, Dimension)
            k:    (Batch, Heads, Sequence_Length, Dimension)
            v:    (Batch, Heads, Sequence_Length, Dimension)
            beta: (Batch, Heads, Sequence_Length) - The decay gate
        ውጤት:
            out:  (Batch, Heads, Sequence_Length, Dimension)
        """
        return ChunkGatedDeltaRuleFunction.apply(q, k, v, beta)
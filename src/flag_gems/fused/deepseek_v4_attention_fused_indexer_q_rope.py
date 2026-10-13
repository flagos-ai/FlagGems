# Adapt from vllm 0.20.2 fused_indexer_q_rope_quant
import torch
import triton
import triton.language as tl

@triton.jit
def _get_cos_sin(
    cos_sin_cache_ptr,
    cos_sin_cache_stride,
    pos,
    HALF_ROT_DIM: tl.constexpr,
):
    block = tl.arange(0, HALF_ROT_DIM)
    cos = tl.load(cos_sin_cache_ptr + pos * cos_sin_cache_stride + block)
    cos = cos.to(tl.float32)
    sin = tl.load(cos_sin_cache_ptr + pos * cos_sin_cache_stride + block + HALF_ROT_DIM)
    sin = sin.to(tl.float32)
    return cos, sin

@triton.jit
def _fused_indexer_q_rope_kernel(
    pos_ptr,
    # Index Q RoPE
    index_q_ptr,
    index_q_stride0,
    index_q_stride1,
    index_q_cos_sin_ptr,
    index_q_cos_sin_stride,
    INDEX_Q_HALF_ROT_DIM: tl.constexpr,
    INDEX_Q_HEAD_DIM: tl.constexpr,
    # Index weights
    index_weights_ptr,
    index_weights_stride,
    index_weights_softmax_scale,
    index_weights_head_scale,
    index_weights_out_ptr,
    index_weights_out_stride,
):
    # Layout matches the unfused reference (DeepseekV4ScalingRotaryEmbedding
    # ): GPT-J interleaved RoPE applied to the
    # LAST rope_dim dims of each head; the leading [0, NOPE_DIM) is passed
    # through unchanged.
    INDEX_Q_ROT_DIM: tl.constexpr = 2 * INDEX_Q_HALF_ROT_DIM
    INDEX_Q_NOPE_DIM: tl.constexpr = INDEX_Q_HEAD_DIM - INDEX_Q_ROT_DIM
    tl.static_assert(INDEX_Q_NOPE_DIM >= 0)

    tok_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    pos = tl.load(pos_ptr + tok_idx)
    cos, sin = _get_cos_sin(
        index_q_cos_sin_ptr,
        index_q_cos_sin_stride,
        pos,
        INDEX_Q_HALF_ROT_DIM,
    )
    half_offset = tl.arange(0, INDEX_Q_HALF_ROT_DIM)
    base_ptr = index_q_ptr + tok_idx * index_q_stride0 + head_idx * index_q_stride1

    # Interleaved (GPT-J) RoPE on dims [NOPE_DIM, HEAD_DIM):
    #   even = q[NOPE_DIM + 2*i],  odd = q[NOPE_DIM + 2*i + 1]
    rot_base = base_ptr + INDEX_Q_NOPE_DIM
    x_even = tl.load(rot_base + half_offset * 2).to(tl.float32)
    x_odd = tl.load(rot_base + half_offset * 2 + 1).to(tl.float32)
    r_even = x_even * cos - x_odd * sin
    r_odd = x_odd * cos + x_even * sin

    # Match reference numerics: fp32 → bf16 → fp32 before the ue8m0 absmax.
    # Same pattern as the K-side compressor kernel (fused_compress_quant_cache.py).
    r_even = r_even.to(tl.bfloat16).to(tl.float32)
    r_odd = r_odd.to(tl.bfloat16).to(tl.float32)

    if INDEX_Q_NOPE_DIM > 0:
        nope_offset = tl.arange(0, INDEX_Q_NOPE_DIM)
        x_nope = tl.load(base_ptr + nope_offset).to(tl.float32)
        tl.store(
            base_ptr + nope_offset,
            x_nope.to(tl.bfloat16),
        )
    rot_base = base_ptr + INDEX_Q_NOPE_DIM
    tl.store(
        rot_base + half_offset * 2, r_even.to(tl.bfloat16)
    )
    tl.store(
        rot_base + half_offset * 2 + 1,
        r_odd.to(tl.bfloat16),
    )

    # BF16 weight-fold contract:
    #   index_weights_out = index_weights * q_scale * softmax_scale * head_scale
    # The per-token-per-head q_scale (fp32) IS folded into the output weights
    # here because FP8 Q is stored WITHOUT a companion scale tensor — the
    # downstream fp8_fp4_mqa_logits/fp8_fp4_paged_mqa_logits kernels use `weights` to
    # apply per-token Q scale inline. See the MXFP4 kernel below for the
    # contrasting convention (scales live with the Q values, weights are NOT
    # q-scaled).
    index_weights = tl.load(
        index_weights_ptr + tok_idx * index_weights_stride + head_idx
    )
    index_weights = index_weights.to(tl.float32)
    index_weights *= index_weights_softmax_scale
    index_weights *= index_weights_head_scale
    tl.store(
        index_weights_out_ptr + tok_idx * index_weights_out_stride + head_idx,
        index_weights,
    )

def fused_indexer_q_rope(
    positions: torch.Tensor,
    index_q: torch.Tensor,
    index_q_cos_sin_cache: torch.Tensor,
    # Index weights
    index_weights: torch.Tensor,
    index_weights_softmax_scale: float,
    index_weights_head_scale: float,
) -> tuple[
    torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    torch.Tensor,
]:
    """Fused RoPE for the sparse indexer.
    Weight-fold semantics (important — the two paths differ):
    BF16 path:
        q_bf16     : (T, H, HEAD_DIM) bf16, per-token-per-head
                     scalar scale (NOT stored — folded into weights below)
        weights_out = weights * softmax_scale * head_scale
        logits kernel would otherwise multiply in. Folding it into `weights`
        avoids emitting a separate tensor and is free for the logits kernel.
    Returns (q, weights_out) where q is either a Tensor (BF16). 
    This matches the union type accepted by `SparseAttnIndexer.forward_*`.
    """
    assert positions.ndim == 1
    assert index_q.ndim == 3
    assert index_q_cos_sin_cache.ndim == 2

    num_tokens = positions.shape[0]
    num_index_q_heads = index_q.shape[1]
    index_q_head_dim = index_q.shape[2]

    index_weights_out = torch.empty_like(index_weights, dtype=torch.float32)

    index_q_ = torch.empty_like(index_q, dtype=torch.bfloat16)
    _fused_indexer_q_rope_kernel[(num_tokens, num_index_q_heads)](
        positions,
        index_q,
        index_q.stride(0),
        index_q.stride(1),
        index_q_cos_sin_cache,
        index_q_cos_sin_cache.stride(0),
        index_q_cos_sin_cache.shape[-1] // 2,
        index_q_head_dim,
        index_weights,
        index_weights.stride(0),
        index_weights_softmax_scale,
        index_weights_head_scale,
        index_weights_out,
        index_weights_out.stride(0),
        num_warps=1,  # TODO: Tune this
    )
    return index_q, index_weights_out


#ifndef FLASH_ATTN_VARLEN_CUH
#define FLASH_ATTN_VARLEN_CUH
#include "attention_utils.cuh"
#include <torch/extension.h>

// 变长FlashAttention（标准MHA，非MLA，prefill+decode通用varlen）
// q: [total_q, nheads, head_dim]
// k,v: [total_k, nheads, head_dim]
// cum_seqlens_q / cum_seqlens_k: int32 cumulative sequence length
// 推理专用，无反向
template<typename T, int BLOCK_Q, int BLOCK_K, int HEAD_DIM>
void launch_flash_attn_varlen_causal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor cum_seqlens_q,
    at::Tensor cum_seqlens_k,
    at::Tensor out,
    float softmax_scale,
    int num_heads,
    int max_seqlen_q,
    int max_seqlen_k
);

// 动态入口，自动分支decode（max_seqlen_q==1）/prefill
at::Tensor flash_attn_varlen_fun_launcher(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor cum_seqlens_q,
    at::Tensor cum_seqlens_k,
    float softmax_scale,
    int num_heads,
    int max_seqlen_q,
    int max_seqlen_k,
    bool causal = true)
{
    auto out = at::empty_like(q);
    if (!causal) {
        TORCH_CHECK(false, "current only support causal=true for LLM inference");
    }
    // 固定特化模板，和现有MLA保持一致 BLOCK_Q, BLOCK_K
    if (q.scalar_type() == at::kBFloat16) {
        launch_flash_attn_varlen_causal<at::BFloat16, 128, 128, 128>(
            q, k, v, cum_seqlens_q, cum_seqlens_k, out,
            softmax_scale, num_heads, max_seqlen_q, max_seqlen_k);
    } else {
        TORCH_CHECK(false, "only bf16 supported");
    }
    return out;
}

#endif

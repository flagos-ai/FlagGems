#pragma once
#include "attention_utils.cuh"

template<typename Dtype, int BLOCK_Q, int BLOCK_K, int HEAD_DIM, int LATENT_DIM>
__global__ void flash_mla_prefill_alibi_causal_kernel(
    const Dtype* q,
    const Dtype* latent_k,
    const Dtype* latent_v,
    const int* cum_seqlens,
    Dtype* out_latent,
    float softmax_scale,
    float alibi_slope,
    int num_heads,
    int max_seqlen
) {
    int tid = threadIdx.x;
    int head = blockIdx.z;
    int block_q = blockIdx.x;
    int block_k = blockIdx.y;

    __shared__ Dtype sh_q[BLOCK_Q * LATENT_DIM];
    __shared__ Dtype sh_lk[BLOCK_K * LATENT_DIM];
    __shared__ Dtype sh_lv[BLOCK_K * LATENT_DIM];

    int seq_start = cum_seqlens[head];
    int seq_end   = cum_seqlens[head + 1];
    int seq_len   = seq_end - seq_start;

    int q_base = seq_start * num_heads * HEAD_DIM + head * HEAD_DIM;
    int k_base = seq_start * num_heads * LATENT_DIM + head * LATENT_DIM;

    for(int i = tid; i < BLOCK_Q * LATENT_DIM; i += blockDim.x) {
        int q_idx_in_tile = i / LATENT_DIM;
        int d = i % LATENT_DIM;
        int pos_q = block_q * BLOCK_Q + q_idx_in_tile;
        sh_q[i] = (pos_q >= seq_len) ? Dtype(0.0f) : q[q_base + pos_q * num_heads * HEAD_DIM + d];
    }
    __syncthreads();

    for(int i = tid; i < BLOCK_K * LATENT_DIM; i += blockDim.x) {
        int k_idx_in_tile = i / LATENT_DIM;
        int d = i % LATENT_DIM;
        int pos_k = block_k * BLOCK_K + k_idx_in_tile;
        sh_lk[i] = (pos_k >= seq_len) ? Dtype(0.0f) : latent_k[k_base + pos_k * num_heads * LATENT_DIM + d];
    }
    __syncthreads();

    for(int i = tid; i < BLOCK_K * LATENT_DIM; i += blockDim.x) {
        int k_idx_in_tile = i / LATENT_DIM;
        int d = i % LATENT_DIM;
        int pos_k = block_k * BLOCK_K + k_idx_in_tile;
        sh_lv[i] = (pos_k >= seq_len) ? Dtype(0.0f) : latent_v[k_base + pos_k * num_heads * LATENT_DIM + d];
    }
    __syncthreads();

    Dtype acc[BLOCK_Q][LATENT_DIM] = {Dtype(0.0f)};
    Dtype row_max[BLOCK_Q] = {Dtype(-1e20f)};
    Dtype row_sumexp[BLOCK_Q] = {Dtype(0.0f)};

    for(int qi = 0; qi < BLOCK_Q; qi++) {
        int pos_q = block_q * BLOCK_Q + qi;
        if(pos_q >= seq_len) continue;

        for(int ki = 0; ki < BLOCK_K; ki++) {
            int pos_k = block_k * BLOCK_K + ki;
            if(pos_k >= seq_len) continue;
            if(pos_k > pos_q) continue; // causal mask

            Dtype s = Dtype(0.0f);
            for(int d = 0; d < LATENT_DIM; d++) {
                s += sh_q[qi * LATENT_DIM + d] * sh_lk[ki * LATENT_DIM + d];
            }
            s *= softmax_scale;
            s += Dtype(alibi_slope * float(pos_k - pos_q));

            Dtype old_max = row_max[qi];
            row_max[qi] = max(row_max[qi], s);
            row_sumexp[qi] = row_sumexp[qi] * exp(old_max - row_max[qi]) + exp(s - row_max[qi]);
            Dtype prob = exp(s - row_max[qi]) / row_sumexp[qi];

            for(int d = 0; d < LATENT_DIM; d++) {
                acc[qi][d] += prob * sh_lv[ki * LATENT_DIM + d];
            }
        }
    }
    __syncthreads();

    int out_base = seq_start * num_heads * LATENT_DIM + head * LATENT_DIM;
    for(int qi = 0; qi < BLOCK_Q; qi++) {
        int pos_q = block_q * BLOCK_Q + qi;
        if(pos_q >= seq_len) continue;
        for(int d = tid; d < LATENT_DIM; d += blockDim.x) {
            out_latent[out_base + pos_q * num_heads * LATENT_DIM + d] = acc[qi][d];
        }
    }
}

template<typename Dtype, int BLOCK_Q, int BLOCK_K, int HEAD_DIM, int LATENT_DIM>
void launch_flash_mla_prefill_alibi_causal(
    const torch::Tensor& q,
    const torch::Tensor& latent_k,
    const torch::Tensor& latent_v,
    const torch::Tensor& cum_seqlens,
    torch::Tensor& out_latent,
    float softmax_scale,
    float alibi_slope,
    int num_heads,
    int max_seqlen
) {
    auto stream = get_flag_stream();
    dim3 grid(
        (max_seqlen + BLOCK_Q - 1)/BLOCK_Q,
        (max_seqlen + BLOCK_K - 1)/BLOCK_K,
        cum_seqlens.numel() - 1
    );
    dim3 block(128);
    flash_mla_prefill_alibi_causal_kernel<Dtype, BLOCK_Q, BLOCK_K, HEAD_DIM, LATENT_DIM>
    <<<grid, block, 0, stream>>>(
        q.data_ptr<Dtype>(),
        latent_k.data_ptr<Dtype>(),
        latent_v.data_ptr<Dtype>(),
        cum_seqlens.data_ptr<int>(),
        out_latent.data_ptr<Dtype>(),
        softmax_scale, alibi_slope, num_heads, max_seqlen
    );
}

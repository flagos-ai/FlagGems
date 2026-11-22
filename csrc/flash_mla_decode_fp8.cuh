#pragma once
#include "attention_utils.cuh"

template<typename ComputeDtype, int HEAD_DIM, int LATENT_DIM, int PAGE_SIZE>
__global__ void flash_mla_paged_decode_fp8_kernel(
    const __nv_fp8_e4m3* q_fp8,
    const __nv_fp8_e4m3* latent_k_cache_fp8,
    const __nv_fp8_e4m3* latent_v_cache_fp8,
    const float q_amax,
    const float kv_amax,
    const int* page_table,
    const int* seq_lens,
    ComputeDtype* out,
    float softmax_scale,
    int batch_size,
    int num_heads,
    int max_pages_per_seq
) {
    int tid = threadIdx.x;
    int block_seq = blockIdx.x;
    int head = blockIdx.y;
    __shared__ ComputeDtype sh_q[HEAD_DIM];
    __shared__ ComputeDtype sh_lk[PAGE_SIZE * LATENT_DIM];
    __shared__ ComputeDtype sh_lv[PAGE_SIZE * LATENT_DIM];

    int q_off = block_seq * num_heads * HEAD_DIM + head * HEAD_DIM;
    for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
        sh_q[d] = float(q_fp8[q_off + d]) * q_amax;
    }
    __syncthreads();

    int cur_seq_len = seq_lens[block_seq];
    ComputeDtype acc_out[HEAD_DIM] = {ComputeDtype(0.0f)};
    ComputeDtype max_score = ComputeDtype(-1e20f);
    ComputeDtype sum_exp = ComputeDtype(0.0f);
    int total_pages = (cur_seq_len + PAGE_SIZE - 1) / PAGE_SIZE;

    for (int p = 0; p < total_pages; p++) {
        int page_id = page_table[block_seq * max_pages_per_seq + p];
        int real_k_in_page = min(PAGE_SIZE, cur_seq_len - p * PAGE_SIZE);
        int cache_base = page_id * PAGE_SIZE * num_heads * LATENT_DIM + head * LATENT_DIM;
        for (int i = tid; i < PAGE_SIZE * LATENT_DIM; i += blockDim.x) {
            int pos_in_page = i / LATENT_DIM;
            int d = i % LATENT_DIM;
            sh_lk[i] = float(latent_k_cache_fp8[cache_base + pos_in_page * num_heads * LATENT_DIM + d]) * kv_amax;
            sh_lv[i] = float(latent_v_cache_fp8[cache_base + pos_in_page * num_heads * LATENT_DIM + d]) * kv_amax;
        }
        __syncthreads();
        if (p + 1 < total_pages) {
            int next_page_id = page_table[block_seq * max_pages_per_seq + p + 1];
            __nv_fp8_e4m3* next_lk_ptr = const_cast<__nv_fp8_e4m3*>(latent_k_cache_fp8) +
                next_page_id * PAGE_SIZE * num_heads * LATENT_DIM + head * LATENT_DIM;
            __nv_fp8_e4m3* next_lv_ptr = const_cast<__nv_fp8_e4m3*>(latent_v_cache_fp8) +
                next_page_id * PAGE_SIZE * num_heads * LATENT_DIM + head * LATENT_DIM;
            flag_async_copy(sh_lk, next_lk_ptr, PAGE_SIZE * LATENT_DIM * sizeof(__nv_fp8_e4m3));
            flag_async_copy(sh_lv, next_lv_ptr, PAGE_SIZE * LATENT_DIM * sizeof(__nv_fp8_e4m3));
        }
        for (int ki = 0; ki < real_k_in_page; ki++) {
            ComputeDtype s = ComputeDtype(0.0f);
            for (int d = 0; d < LATENT_DIM; d++) s += sh_q[d] * sh_lk[ki * LATENT_DIM + d];
            s *= softmax_scale;
            ComputeDtype old_max = max_score;
            max_score = max(max_score, s);
            sum_exp = sum_exp * exp(old_max - max_score) + exp(s - max_score);
            ComputeDtype prob = exp(s - max_score) / sum_exp;
            for (int d = 0; d < HEAD_DIM; d++) acc_out[d] += prob * sh_lv[ki * LATENT_DIM + d];
        }
        __syncthreads();
    }
    for (int d = tid; d < HEAD_DIM; d += blockDim.x) out[q_off + d] = acc_out[d];
}

void launch_flash_mla_decode_fp8(
    const torch::Tensor& q_fp8,
    const torch::Tensor& lk_fp8,
    const torch::Tensor& lv_fp8,
    float q_amax,
    float kv_amax,
    const torch::Tensor& page_table,
    const torch::Tensor& seq_lens,
    torch::Tensor& out,
    float softmax_scale,
    int batch_size,
    int num_heads,
    int max_pages_per_seq
) {
    auto stream = get_flag_stream();
    constexpr int HEAD_DIM = 128;
    constexpr int LATENT_DIM = 512;
    constexpr int PAGE_SIZE = 64;
    dim3 grid(batch_size, num_heads);
    dim3 block(128);
    flash_mla_paged_decode_fp8_kernel<at::BFloat16, HEAD_DIM, LATENT_DIM, PAGE_SIZE>
    <<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(q_fp8.data_ptr()),
        reinterpret_cast<const __nv_fp8_e4m3*>(lk_fp8.data_ptr()),
        reinterpret_cast<const __nv_fp8_e4m3*>(lv_fp8.data_ptr()),
        q_amax, kv_amax,
        page_table.data_ptr<int>(),
        seq_lens.data_ptr<int>(),
        out.data_ptr<at::BFloat16>(),
        softmax_scale, batch_size, num_heads, max_pages_per_seq
    );
}

#pragma once
#include "attention_utils.cuh"

template<typename Dtype, int HEAD_DIM, int LATENT_DIM, int PAGE_SIZE>
__global__ void flash_mla_paged_decode_kernel(
    const Dtype* q,
    const Dtype* latent_k_cache,
    const Dtype* latent_v_cache,
    const int* page_table,
    const int* seq_lens,
    Dtype* out,
    float softmax_scale,
    int batch_size,
    int num_heads,
    int max_pages_per_seq
) {
    int tid = threadIdx.x;
    int block_seq = blockIdx.x;
    int head = blockIdx.y;

    __shared__ Dtype sh_q[HEAD_DIM];
    __shared__ Dtype sh_lk[PAGE_SIZE * LATENT_DIM];
    __shared__ Dtype sh_lv[PAGE_SIZE * LATENT_DIM];

    int q_off = block_seq * num_heads * HEAD_DIM + head * HEAD_DIM;
    for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
        sh_q[d] = q[q_off + d];
    }
    __syncthreads();

    int cur_seq_len = seq_lens[block_seq];
    if (cur_seq_len <= 0) {
        for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
            out[q_off + d] = Dtype(0.0f);
        }
        return;
    }

    Dtype acc_out[HEAD_DIM] = {Dtype(0.0f)};
    Dtype max_score = Dtype(-1e20f);
    Dtype sum_exp = Dtype(0.0f);
    int total_pages = (cur_seq_len + PAGE_SIZE - 1) / PAGE_SIZE;

    for (int p = 0; p < total_pages; p++) {
        int page_id = page_table[block_seq * max_pages_per_seq + p];
        int real_k_in_page = min(PAGE_SIZE, cur_seq_len - p * PAGE_SIZE);
        int cache_base = page_id * PAGE_SIZE * num_heads * LATENT_DIM + head * LATENT_DIM;

        for (int i = tid; i < PAGE_SIZE * LATENT_DIM; i += blockDim.x) {
            int pos_in_page = i / LATENT_DIM;
            int d = i % LATENT_DIM;
            sh_lk[i] = latent_k_cache[cache_base + pos_in_page * num_heads * LATENT_DIM + d];
            sh_lv[i] = latent_v_cache[cache_base + pos_in_page * num_heads * LATENT_DIM + d];
        }
        __syncthreads();

        if (p + 1 < total_pages) {
            int next_page_id = page_table[block_seq * max_pages_per_seq + p + 1];
            Dtype* next_lk_ptr = const_cast<Dtype*>(latent_k_cache) +
                next_page_id * PAGE_SIZE * num_heads * LATENT_DIM + head * LATENT_DIM;
            Dtype* next_lv_ptr = const_cast<Dtype*>(latent_v_cache) +
                next_page_id * PAGE_SIZE * num_heads * LATENT_DIM + head * LATENT_DIM;
            flag_async_copy(sh_lk, next_lk_ptr, PAGE_SIZE * LATENT_DIM * sizeof(Dtype));
            flag_async_copy(sh_lv, next_lv_ptr, PAGE_SIZE * LATENT_DIM * sizeof(Dtype));
        }

        for (int ki = 0; ki < real_k_in_page; ki++) {
            Dtype s = Dtype(0.0f);
            for (int d = 0; d < LATENT_DIM; d++) {
                s += sh_q[d] * sh_lk[ki * LATENT_DIM + d];
            }
            s *= softmax_scale;
            Dtype old_max = max_score;
            max_score = max(max_score, s);
            sum_exp = sum_exp * exp(old_max - max_score) + exp(s - max_score);
            Dtype prob = exp(s - max_score) / sum_exp;
            for (int d = 0; d < HEAD_DIM; d++) {
                acc_out[d] += prob * sh_lv[ki * LATENT_DIM + d];
            }
        }
        __syncthreads();
    }
    for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
        out[q_off + d] = acc_out[d];
    }
}

void launch_flash_mla_paged_decode(
    const torch::Tensor& q,
    const torch::Tensor& latent_k_cache,
    const torch::Tensor& latent_v_cache,
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
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(),
        "flash_mla_paged_decode", [&] {
        flash_mla_paged_decode_kernel<scalar_t, HEAD_DIM, LATENT_DIM, PAGE_SIZE>
        <<<grid, block, 0, stream>>>(
            q.data_ptr<scalar_t>(),
            latent_k_cache.data_ptr<scalar_t>(),
            latent_v_cache.data_ptr<scalar_t>(),
            page_table.data_ptr<int>(),
            seq_lens.data_ptr<int>(),
            out.data_ptr<scalar_t>(),
            softmax_scale, batch_size, num_heads, max_pages_per_seq
        );
    });
}

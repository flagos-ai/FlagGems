// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <ATen/WrapDimUtils.h>
#include <algorithm>
#include "flag_gems/backend_utils.h"
#include "flag_gems/device_info.h"
#include "flag_gems/operators.h"
#include "flag_gems/utils.h"
#include "triton_jit/triton_jit_function.h"
namespace flag_gems {
using namespace triton_jit;

namespace {

  const TritonJITFunction &get_kernel(const std::string &name) {
    static const std::string src_path = (utils::get_flag_gems_src_path() / "ops" / "softmax.py").string();
    return TritonJITFunction::get_instance(src_path, name);
  }

  void compute_mnk(const at::Tensor &tensor, int dim, int64_t &M, int64_t &N, int64_t &K) {
    const auto sizes = tensor.sizes();
    M = 1;
    N = sizes[dim];
    K = 1;
    for (int i = 0; i < dim; ++i) M *= sizes[i];
    for (int i = dim + 1; i < sizes.size(); ++i) K *= sizes[i];
  }

  unsigned int softmax_heur_num_warps(unsigned int tile_size) {
    if (tile_size < 2048) {
      return 4;
    }
    if (tile_size < 4096) {
      return 8;
    }
    return 16;
  }

  unsigned int softmax_heur_tile_n_inner(int64_t N) {
    if (N <= 32 * 1024) {
      return static_cast<unsigned int>(utils::next_power_of_2(N));
    }
    return 4096;
  }

  unsigned int softmax_heur_tile_k(int64_t M, int64_t K) {
    constexpr int MAX_TILE_K = 8192;
    const int num_sms = std::max(1, device::current_sm_count());
    unsigned int tile_k = 1;
    const int64_t upper_bound = std::min(K, static_cast<int64_t>(MAX_TILE_K));
    while (static_cast<int64_t>(tile_k) <= upper_bound) {
      const int64_t num_blocks = M * utils::cdiv(static_cast<int>(K), static_cast<int>(tile_k));
      const double num_waves = static_cast<double>(num_blocks) / num_sms;
      if (num_waves > 1.0 && static_cast<int64_t>(tile_k) * 2 <= upper_bound) {
        tile_k *= 2;
      } else {
        break;
      }
    }
    return tile_k;
  }

  unsigned int softmax_heur_one_tile_per_cta(unsigned int tile_n, int64_t N) {
    return tile_n >= static_cast<unsigned int>(N) ? 1u : 0u;
  }

  struct SoftmaxInnerConfig {
    unsigned int tile_n;
    unsigned int one_tile_per_cta;
    unsigned int num_warps;
  };

  SoftmaxInnerConfig compute_inner_config(int64_t N) {
    const unsigned int tile_n = softmax_heur_tile_n_inner(N);
    return {tile_n, softmax_heur_one_tile_per_cta(tile_n, N), softmax_heur_num_warps(tile_n)};
  }

  struct SoftmaxNonInnerConfig {
    unsigned int tile_n;
    unsigned int tile_k;
    unsigned int one_tile_per_cta;
    unsigned int num_warps;
  };

  SoftmaxNonInnerConfig compute_forward_non_inner_config(int64_t M, int64_t N, int64_t K) {
    const unsigned int tile_k = softmax_heur_tile_k(M, K);
    const unsigned int tile_n = static_cast<unsigned int>(utils::cdiv(8192, static_cast<int>(tile_k)));
    const unsigned int tile_size = tile_n * tile_k;
    return {tile_n, tile_k, softmax_heur_one_tile_per_cta(tile_n, N), softmax_heur_num_warps(tile_size)};
  }

  SoftmaxNonInnerConfig compute_backward_non_inner_config(int64_t M, int64_t N, int64_t K) {
    const unsigned int tile_k = softmax_heur_tile_k(M, K);
    const unsigned int tile_n = std::max(1u, 1024u / tile_k);
    const unsigned int tile_size = tile_n * tile_k;
    return {tile_n, tile_k, softmax_heur_one_tile_per_cta(tile_n, N), softmax_heur_num_warps(tile_size)};
  }

  struct SoftmaxBackwardInnerConfig {
    unsigned int tile_m;
    unsigned int tile_n;
    unsigned int one_tile_per_cta;
    unsigned int num_warps;
  };

  SoftmaxBackwardInnerConfig compute_backward_inner_config(int64_t N) {
    const unsigned int tile_n = softmax_heur_tile_n_inner(N);
    const unsigned int tile_m = std::max(1u, 1024u / tile_n);
    return {tile_m, tile_n, softmax_heur_one_tile_per_cta(tile_n, N), softmax_heur_num_warps(tile_n)};
  }

  at::Tensor softmax_forward(const at::Tensor &input, int dim) {
    at::Tensor output = at::empty_like(input, input.options());

    int64_t M, N, K;
    compute_mnk(input, dim, M, N, K);

    c10::DeviceGuard guard(input.device());
    backend::StreamType stream = backend::getCurrentStream();
    backend::RawStreamType raw_stream = backend::getRawStream(stream);

    constexpr unsigned int NUM_STAGES = 1;

    if (K == 1) {
      const SoftmaxInnerConfig config = compute_inner_config(N);
      const TritonJITFunction &kernel = get_kernel("softmax_kernel_inner");
      const unsigned int grid_x = static_cast<unsigned int>(M);

      kernel(raw_stream,
             grid_x,
             1,
             1,
             config.num_warps,
             NUM_STAGES,
             output,
             input,
             M,
             N,
             config.tile_n,
             config.one_tile_per_cta);
    } else {
      const SoftmaxNonInnerConfig config = compute_forward_non_inner_config(M, N, K);
      const TritonJITFunction &kernel = get_kernel("softmax_kernel_non_inner");
      const unsigned int grid_x = static_cast<unsigned int>(M);
      const unsigned int grid_y =
          static_cast<unsigned int>(utils::cdiv(static_cast<int>(K), static_cast<int>(config.tile_k)));

      kernel(raw_stream,
             grid_x,
             grid_y,
             1,
             config.num_warps,
             NUM_STAGES,
             output,
             input,
             M,
             N,
             K,
             config.tile_n,
             config.tile_k,
             config.one_tile_per_cta);
    }

    return output;
  }

  void compute_mnk_for_backward(const at::Tensor &tensor,
                                int dim,
                                int64_t &M,
                                int64_t &N,
                                int64_t &K,
                                int64_t &stride_m,
                                int64_t &stride_n,
                                int64_t &stride_k) {
    const auto sizes = tensor.sizes();
    const auto strides = tensor.strides();

    M = 1;
    for (int i = 0; i < dim; ++i) M *= sizes[i];
    N = sizes[dim];
    K = 1;
    for (int i = dim + 1; i < sizes.size(); ++i) K *= sizes[i];

    stride_m = (dim > 0) ? strides[dim - 1] : 0;
    stride_n = strides[dim];
    stride_k = (dim + 1 < sizes.size()) ? strides[dim + 1] : 1;

    if (K == 1) stride_k = 0;
    if (M == 1) stride_m = 0;
  }

  at::Tensor softmax_backward_impl(const at::Tensor &output, const at::Tensor &grad_output, int dim) {
    at::Tensor grad_input = at::empty_like(grad_output, grad_output.options());

    int64_t M, N, K;
    int64_t stride_m, stride_n, stride_k;
    compute_mnk_for_backward(output, dim, M, N, K, stride_m, stride_n, stride_k);

    c10::DeviceGuard guard(output.device());
    backend::StreamType stream = backend::getCurrentStream();
    backend::RawStreamType raw_stream = backend::getRawStream(stream);

    constexpr unsigned int NUM_STAGES = 1;

    if (K == 1) {
      const SoftmaxBackwardInnerConfig config = compute_backward_inner_config(N);
      const TritonJITFunction &kernel = get_kernel("softmax_backward_kernel_inner");
      const unsigned int grid_x =
          static_cast<unsigned int>(utils::cdiv(static_cast<int>(M), static_cast<int>(config.tile_m)));

      kernel(raw_stream,
             grid_x,
             1,
             1,
             config.num_warps,
             NUM_STAGES,
             output,
             grad_output,
             grad_input,
             M,
             N,
             config.tile_m,
             config.tile_n,
             config.one_tile_per_cta);
    } else {
      const SoftmaxNonInnerConfig config = compute_backward_non_inner_config(M, N, K);
      const TritonJITFunction &kernel = get_kernel("softmax_backward_kernel_non_inner");
      const unsigned int grid_x = static_cast<unsigned int>(M);
      const unsigned int grid_y =
          static_cast<unsigned int>(utils::cdiv(static_cast<int>(K), static_cast<int>(config.tile_k)));

      kernel(raw_stream,
             grid_x,
             grid_y,
             1,
             config.num_warps,
             NUM_STAGES,
             output,
             grad_output,
             grad_input,
             M,
             N,
             K,
             config.tile_n,
             config.tile_k,
             config.one_tile_per_cta);
    }

    return grad_input;
  }

}  // namespace

at::Tensor softmax(const at::Tensor &input, int64_t dim, bool half_to_float) {
  int64_t dim_ = at::maybe_wrap_dim(dim, input.dim());

  const at::ScalarType out_dtype =
      (half_to_float && input.scalar_type() == at::kHalf) ? at::kFloat : input.scalar_type();

  if (input.numel() == 0) {
    at::Tensor output = at::empty_like(input, input.options().dtype(out_dtype));
    output.zero_();
    return output;
  }

  at::Tensor input_tensor = input.contiguous();
  if (half_to_float && input.scalar_type() == at::kHalf) {
    input_tensor = input_tensor.to(at::kFloat);
  }

  return softmax_forward(input_tensor, static_cast<int>(dim_));
}

at::Tensor softmax_backward(const at::Tensor &grad_output,
                            const at::Tensor &output,
                            int64_t dim,
                            at::ScalarType input_dtype) {
  int64_t wrapped_dim = at::maybe_wrap_dim(dim, output.dim());

  if (output.numel() == 0) {
    at::Tensor grad_input = at::empty_like(grad_output, grad_output.options().dtype(input_dtype));
    grad_input.zero_();
    return grad_input;
  }

  at::Tensor grad_input = softmax_backward_impl(output, grad_output, wrapped_dim);

  if (grad_input.scalar_type() != input_dtype) {
    grad_input = grad_input.to(input_dtype);
  }

  return grad_input;
}

}  // namespace flag_gems

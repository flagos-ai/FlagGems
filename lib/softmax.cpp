#include <ATen/WrapDimUtils.h>
#include <iostream>
#include "c10/cuda/CUDAStream.h"
#include "flag_gems/operators.h"
#include "flag_gems/utils.h"
#include "triton_jit/triton_jit_function.h"
namespace flag_gems {
using namespace triton_jit;

namespace {

  // Load Triton JIT kernel from softmax.py
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

  // Forward kernel wrapper
  at::Tensor softmax_forward(const at::Tensor &input, int dim) {
    TORCH_CHECK(input.dim() >= 2, "Softmax input must be at least 2D");

    at::Tensor output = at::empty_like(input, input.options());

    int64_t M, N, K;
    compute_mnk(input, dim, M, N, K);

    constexpr unsigned int TILE_N = 128;
    constexpr unsigned int TILE_K = 1;
    constexpr unsigned int ONE_TILE_PER_CTA = 1;
    constexpr unsigned int NUM_WARPS = 4;
    constexpr unsigned int NUM_STAGES = 1;

    c10::DeviceGuard guard(input.device());
    c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
    CUstream raw_stream = static_cast<CUstream>(stream.stream());

    if (K == 1) {
      const TritonJITFunction &kernel = get_kernel("softmax_kernel_inner");
      unsigned int grid_x = static_cast<unsigned int>(M);

      kernel(raw_stream, grid_x, 1, 1, NUM_WARPS, NUM_STAGES, output, input, M, N, TILE_N, ONE_TILE_PER_CTA);
    } else {
      const TritonJITFunction &kernel = get_kernel("softmax_kernel_non_inner");
      unsigned int grid_x = static_cast<unsigned int>(M);
      unsigned int grid_y = static_cast<unsigned int>((K + TILE_K - 1) / TILE_K);

      kernel(raw_stream,
             grid_x,
             grid_y,
             1,
             NUM_WARPS,
             NUM_STAGES,
             output,
             input,
             M,
             N,
             K,
             TILE_N,
             TILE_K,
             ONE_TILE_PER_CTA);
    }

    return output;
  }

  // Backward kernel wrapper
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
    at::Tensor grad_output_contiguous = grad_output.contiguous();

    at::Tensor grad_input = at::empty_like(grad_output, grad_output.options());

    int64_t M, N, K;
    int64_t stride_m, stride_n, stride_k;
    compute_mnk_for_backward(output, dim, M, N, K, stride_m, stride_n, stride_k);

    constexpr unsigned int TILE_N = 128;
    constexpr unsigned int TILE_K = 1;
    constexpr unsigned int TILE_M = 64;
    constexpr unsigned int ONE_TILE_PER_CTA = 1;
    constexpr unsigned int NUM_WARPS = 4;
    constexpr unsigned int NUM_STAGES = 1;

    c10::DeviceGuard guard(output.device());
    c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
    CUstream raw_stream = static_cast<CUstream>(stream.stream());

    if (K == 1) {
      const TritonJITFunction &kernel = get_kernel("softmax_backward_kernel_inner");
      unsigned int grid_x = static_cast<unsigned int>((M + TILE_M - 1) / TILE_M);

      kernel(raw_stream,
             grid_x,
             1,
             1,
             NUM_WARPS,
             NUM_STAGES,
             output,
             grad_output,
             grad_input,
             M,
             N,
             TILE_M,
             TILE_N,
             ONE_TILE_PER_CTA);
    } else {
      const TritonJITFunction &kernel = get_kernel("softmax_backward_kernel_non_inner");
      unsigned int grid_x = static_cast<unsigned int>(M);
      unsigned int grid_y = static_cast<unsigned int>((K + TILE_K - 1) / TILE_K);

      kernel(raw_stream,
             grid_x,
             grid_y,
             1,
             NUM_WARPS,
             NUM_STAGES,
             output,
             grad_output,
             grad_input,
             M,
             N,
             K,
             TILE_N,
             TILE_K,
             ONE_TILE_PER_CTA);
    }

    return grad_input;
  }

}  // namespace

// Public API
at::Tensor softmax(const at::Tensor &input, int64_t dim, bool half_to_float) {
  int64_t dim_ = at::maybe_wrap_dim(dim, input.dim());

  at::Tensor input_tensor = input;
  if (half_to_float && input.scalar_type() == at::kHalf) {
    input_tensor = input_tensor.to(at::kFloat);
  }

  at::Tensor output = softmax_forward(input_tensor, static_cast<int>(dim_));

  return output;
}

at::Tensor softmax_backward(const at::Tensor &grad_output,
                            const at::Tensor &output,
                            int64_t dim,
                            at::ScalarType input_dtype) {
  int64_t wrapped_dim = at::maybe_wrap_dim(dim, output.dim());

  at::Tensor output_tensor = output;
  at::Tensor grad_output_tensor = grad_output;

  at::Tensor grad_input = softmax_backward_impl(output_tensor, grad_output_tensor, wrapped_dim);

  if (grad_input.scalar_type() != input_dtype) {
    grad_input = grad_input.to(input_dtype);
  }

  return grad_input;
}

}  // namespace flag_gems

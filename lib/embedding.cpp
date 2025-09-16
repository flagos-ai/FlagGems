#include "flag_gems/operators.h"
#include "flag_gems/utils.h"

#include <iostream>
#include <tuple>
#include "c10/cuda/CUDAStream.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
using namespace triton_jit;

at::Tensor embedding(const at::Tensor &weight,
                     const at::Tensor &indices,
                     int64_t padding_idx,
                     bool scale_grad_by_freq,
                     bool sparse) {
  TORCH_CHECK(!sparse, "Currently do not support sparse format");
  int64_t M = indices.numel();
  int64_t N = weight.size(-1);
  int64_t BLOCK_SIZE = utils::next_power_of_2(N);
  at::Tensor contiguous_indices = indices.contiguous();
  at::Tensor contiguous_weight = weight.contiguous();

  std::vector<int64_t> output_shape;
  output_shape.insert(output_shape.end(), indices.sizes().begin(), indices.sizes().end());
  output_shape.push_back(N);
  at::Tensor output =
      at::empty(output_shape, at::TensorOptions().dtype(weight.dtype()).device(indices.device()));
  const TritonJITFunction &f1 =
      TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / "ops" / "embedding.py"),
                                      "embedding_kernel");
  c10::DeviceGuard guard(output.device());
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());
  /*
  def embedding_kernel(
      out_ptr,  # pointer to the output
      in_ptr,  # pointer to the input
      weight_ptr,  # pointer to the weights
      N: tl.constexpr,  # number of columns in X
      BLOCK_SIZE: tl.constexpr,
  ):*/
  f1(raw_stream,
     M,
     1,
     1,
     /* num_warps = */ 8,
     /* num_stages = */ 1,
     output,
     contiguous_indices,
     contiguous_weight,
     N,
     BLOCK_SIZE);

  return output;
}

at::Tensor embedding_backward(const at::Tensor &grad_outputs,
                              const at::Tensor &indices,
                              int64_t num_weights,
                              int64_t padding_idx,
                              bool scale_grad_by_freq,
                              bool sparse) {
  TORCH_CHECK(!sparse, "Currently do not support sparse format");
  int64_t M = indices.numel();
  int64_t N = grad_outputs.size(-1);
  at::Tensor grad_inputs;
  if (grad_outputs.dtype() == at::kBFloat16) {
    grad_inputs = at::zeros({num_weights, N}, grad_outputs.options().dtype(at::kFloat));
  } else {
    grad_inputs = at::zeros({num_weights, N}, grad_outputs.options());
  }
  at::Tensor indice_freq;
  if (scale_grad_by_freq) {
    indice_freq = at::zeros({num_weights}, indices.options().dtype(at::kInt));
    int64_t INDICE_BLOCK_SIZE = 256;
    int64_t indice_grid = (M + INDICE_BLOCK_SIZE - 1) / INDICE_BLOCK_SIZE;
    const TritonJITFunction &f2 =
        TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / "ops" / "embedding.py"),
                                        "indice_freq_kernel");
    c10::DeviceGuard guard(grad_outputs.device());
    c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
    CUstream raw_stream = static_cast<CUstream>(stream.stream());
    /*
    def indice_freq_kernel(
        indices_freq,
        indices,  # pointer to the input
        elem_cnt: tl.constexpr,  # number of columns in X
        INDICE_BLOCK_SIZE: tl.constexpr,
    ):*/
    f2(raw_stream,
       indice_grid,
       1,
       1,
       /* num_warps = */ 8,
       /* num_stages = */ 1,
       indice_freq,
       indices,
       M,
       INDICE_BLOCK_SIZE);
  }
  int64_t BLOCK_SIZE = utils::next_power_of_2(N);
  bool HAS_PADDING_IDX = (padding_idx != -1);
  const TritonJITFunction &f3 =
      TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / "ops" / "embedding.py"),
                                      "embedding_backward_kernel");
  c10::DeviceGuard guard(grad_outputs.device());
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());
  /*
  def embedding_backward_kernel(
      grad_in,  # pointer to the gradient input
      grad_out,  # pointer to the gradient output
      indices,  # pointer to the input
      padding_idx,  # padding_idx
      HAS_PADDING_IDX: tl.constexpr,
      N: tl.constexpr,  # number of columns in X
      BLOCK_SIZE: tl.constexpr,
  ):*/
  f3(raw_stream,
     M,
     1,
     1,
     /* num_warps = */ 8,
     /* num_stages = */ 1,
     grad_inputs,
     grad_outputs,
     indices,
     padding_idx,
     HAS_PADDING_IDX,
     N,
     BLOCK_SIZE);
  if (scale_grad_by_freq) {
    const TritonJITFunction &f4 =
        TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / "ops" / "embedding.py"),
                                        "embedding_grad_scale_kernel");
    c10::DeviceGuard guard(grad_outputs.device());
    c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
    CUstream raw_stream = static_cast<CUstream>(stream.stream());
    /*
    def embedding_grad_scale_kernel(
        grad_out,
        indice_freq,
        n_rows,
        N,
        BLOCK_SIZE: tl.constexpr,
    ):*/
    f4(raw_stream,
       M,
       1,
       1,
       /* num_warps = */ 8,
       /* num_stages = */ 1,
       grad_inputs,
       indice_freq,
       num_weights,
       N,
       BLOCK_SIZE);
  }

  if (grad_outputs.dtype() == at::kBFloat16) {
    return grad_inputs.to(at::kBFloat16);
  }
  return grad_inputs;
}
}  // namespace flag_gems

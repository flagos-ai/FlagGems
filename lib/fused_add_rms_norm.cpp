#include "flag_gems/operators.h"
#include "flag_gems/utils.h"

#include <iostream>
#include "c10/cuda/CUDAStream.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
using namespace triton_jit;

// TODO(flaggems): Only supports 2D inputs and 1D weight (last-dim norm).
// Extend to support higher-rank inputs and generalized weight shapes

void fused_add_rms_norm(at::Tensor& input,         // [..., hidden_size]
                        at::Tensor& residual,      // [..., hidden_size]
                        const at::Tensor& weight,  // [hidden_size]
                        double epsilon) {          //  default 1e-5

  TORCH_CHECK(input.sizes() == residual.sizes(),
              "Input and residual must have the same shape, but got ",
              input.sizes(),
              " vs ",
              residual.sizes());
  int64_t hidden_size = input.size(-1);
  int64_t M = input.numel() / hidden_size;
  int64_t N = weight.size(0);  // assumes 1D weight
  int64_t BLOCK_SIZE = utils::next_power_of_2(N);

  auto input_strides = input.strides();
  auto residual_strides = residual.strides();

  const TritonJITFunction& f = TritonJITFunction::get_instance(
      std::string(utils::get_flag_gems_src_path() / "fused" / "fused_add_rms_norm.py"),
      "fused_add_rms_norm_kernel");

  // getCurrentCUDAStream ensures that the stream is initialized, a default stream for each device
  c10::DeviceGuard guard(input.device());
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());

  /* siguature info
def fused_add_rms_norm_kernel(
    in_ptr,  # pointer to the input
    re_ptr,  # pointer to the residual
    w_ptr,  # pointer to the weights
    in_stride_r,  # how much to increase the pointer when moving by 1 row
    in_stride_c,  # how much to increase the pointer when moving by 1 col
    r_stride_r,  # how much to increase the pointer when moving by 1 row
    r_stride_c,  # how much to increase the pointer when moving by 1 col
    N,  # number of columns in in_ptr
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
  ) */
  f(raw_stream,
    M,
    1,
    1,
    /* num_warps */ 8,
    /* num_stages */ 1,
    input,
    residual,
    weight,
    input_strides[0],
    input_strides[1],
    residual_strides[0],
    residual_strides[1],
    N,
    epsilon,
    BLOCK_SIZE);

  return;
}
}  // namespace flag_gems

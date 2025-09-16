#include "flag_gems/operators.h"
#include "flag_gems/utils.h"

#include <iostream>
#include "c10/cuda/CUDAStream.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
using namespace triton_jit;

// TODO(flaggems): Only supports 2D inputs and 1D weight (last-dim norm).
// Extend to support higher-rank inputs and generalized weight shapes
// like torch.nn.functional.rms_norm.

at::Tensor rms_norm(const at::Tensor& input,   // [..., hidden_size]
                    const at::Tensor& weight,  // [hidden_size]
                    double epsilon) {          //  default 1e-5

  int64_t hidden_size = input.size(-1);
  int64_t M = input.numel() / hidden_size;
  int64_t N = weight.size(0);  // assumes 1D weight
  int64_t BLOCK_SIZE = utils::next_power_of_2(N);

  at::Tensor out = at::empty(input.sizes(), input.options());
  at::Tensor inv_rms = at::empty({M}, at::TensorOptions().dtype(torch::kFloat32).device(input.device()));

  auto input_strides = input.strides();
  auto output_strides = out.strides();

  const TritonJITFunction& f =
      TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / "ops" / "rms_norm.py"),
                                      "rms_norm_kernel");

  // getCurrentCUDAStream ensures that the stream is initialized, a default stream for each device
  c10::DeviceGuard guard(out.device());
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());

  /* siguature info
  def rms_norm_kernel(
    Y,  # pointer to the output
    INV_RMS,  # pointer to inverse rms
    X,  # pointer to the input
    W,  # pointer to the weights
    y_stride_r,
    y_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr
  ) */
  f(raw_stream,
    M,
    1,
    1,
    /* num_warps */ 8,
    /* num_stages */ 1,
    out,
    inv_rms,
    input,
    weight,
    output_strides[0],
    output_strides[1],
    input_strides[0],
    input_strides[1],
    N,
    epsilon,
    BLOCK_SIZE);

  return out;
}
}  // namespace flag_gems

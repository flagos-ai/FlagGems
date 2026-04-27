#include <iostream>
#include "flag_gems/backend_utils.h"
#include "flag_gems/operators.h"
#include "flag_gems/utils.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
using namespace triton_jit;

namespace {

  int get_fused_add_rms_norm_num_warps(int64_t block_size) {
#if defined(FLAGGEMS_USE_IX)
    if (block_size < 2048) {
      return 4;
    }
    if (block_size < 4096) {
      return 8;
    }
    return 16;
#else
    return 8;
#endif
  }

}  // namespace

void fused_add_rms_norm(at::Tensor& input, at::Tensor& residual, const at::Tensor& weight, double epsilon) {
  TORCH_CHECK(input.sizes() == residual.sizes(),
              "Input and residual must have the same shape, but got ",
              input.sizes(),
              " vs ",
              residual.sizes());
  at::Tensor contig_weight = weight.contiguous();
  const float epsilon_val = static_cast<float>(epsilon);
  at::IntArrayRef normalized_shape = weight.sizes();
  int64_t dim = input.ndimension() - normalized_shape.size();
  int64_t M = 1;
  for (int i = 0; i < dim; ++i) {
    M *= input.size(i);
  }
  int64_t N = input.numel() / M;
  int64_t BLOCK_SIZE = utils::next_power_of_2(N);

  const TritonJITFunction& f = TritonJITFunction::get_instance(
      std::string(utils::get_flag_gems_src_path() / "fused" / "fused_add_rms_norm.py"),
      "fused_add_rms_norm_kernel");

  // getCurrentCUDAStream ensures that the stream is initialized, a default stream for each device
  c10::DeviceGuard guard(input.device());
  backend::StreamType stream = backend::getCurrentStream();
  backend::RawStreamType raw_stream = backend::getRawStream(stream);

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
    /* num_warps */ get_fused_add_rms_norm_num_warps(BLOCK_SIZE),
    /* num_stages */ 1,
    input,
    residual,
    contig_weight,
    N,
    1,
    N,
    1,
    N,
    epsilon_val,
    BLOCK_SIZE);

  return;
}
}  // namespace flag_gems

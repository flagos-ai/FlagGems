#include <iostream>
#include "flag_gems/backend_utils.h"
#include "flag_gems/operators.h"
#include "flag_gems/utils.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
using namespace triton_jit;

namespace {

  int get_rms_norm_num_warps(int64_t block_size) {
#if defined(FLAGGEMS_USE_IX)
    // Python launches this kernel without forcing num_warps. The previous C++
    // wrapper hard-coded 8 warps, which is too aggressive for small IX RMSNorm
    // tiles and leads to obviously incorrect results. Use a conservative
    // heuristic that matches the default Python behavior more closely.
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

at::Tensor rms_norm(const at::Tensor& input, const at::Tensor& weight, double epsilon) {
  at::Tensor contig_input = input.contiguous();
  at::Tensor contig_weight = weight.contiguous();
  const float epsilon_val = static_cast<float>(epsilon);
  at::IntArrayRef normalized_shape = contig_weight.sizes();
  int64_t dim = contig_input.ndimension() - normalized_shape.size();
  int64_t M = 1;
  for (int i = 0; i < dim; ++i) {
    M *= contig_input.size(i);
  }
  int64_t N = contig_input.numel() / M;
  int64_t BLOCK_SIZE = utils::next_power_of_2(N);

  at::Tensor out = at::empty(input.sizes(), input.options());
  at::Tensor inv_rms = at::empty({M}, at::TensorOptions().dtype(torch::kFloat32).device(input.device()));

  const TritonJITFunction& f =
      TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / "ops" / "rms_norm.py"),
                                      "rms_norm_kernel");

  // getCurrentCUDAStream ensures that the stream is initialized, a default stream for each device
  c10::DeviceGuard guard(out.device());
  backend::StreamType stream = backend::getCurrentStream();
  backend::RawStreamType raw_stream = backend::getRawStream(stream);

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
    /* num_warps */ get_rms_norm_num_warps(BLOCK_SIZE),
    /* num_stages */ 1,
    out,
    inv_rms,
    contig_input,
    contig_weight,
    N,
    1,
    N,
    1,
    N,
    epsilon_val,
    BLOCK_SIZE);

  return out;
}
}  // namespace flag_gems

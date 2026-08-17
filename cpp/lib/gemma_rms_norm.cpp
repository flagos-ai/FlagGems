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

#include <iostream>
#include "flag_gems/backend_utils.h"
#include "flag_gems/operators.h"
#include "flag_gems/utils.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
using namespace triton_jit;

namespace {

  int get_gemma_rms_norm_num_warps(int64_t block_size) {
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

// Gemma-style RMSNorm: y = x / sqrt(mean(x^2) + eps) * (1 + weight).
// Differs from rms_norm only by the +1 unit offset on the weight (done in
// fp32 inside the kernel); the C++ launch path is identical.
at::Tensor gemma_rms_norm(const at::Tensor& input, const at::Tensor& weight, double epsilon) {
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

  const TritonJITFunction& f = TritonJITFunction::get_instance(
      std::string(utils::get_flag_gems_src_path() / "ops" / "gemma_rms_norm.py"),
      "gemma_rms_norm_kernel");

  c10::DeviceGuard guard(out.device());
  backend::StreamType stream = backend::getCurrentStream();
  backend::RawStreamType raw_stream = backend::getRawStream(stream);

  f(raw_stream,
    M,
    1,
    1,
    /* num_warps */ get_gemma_rms_norm_num_warps(BLOCK_SIZE),
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

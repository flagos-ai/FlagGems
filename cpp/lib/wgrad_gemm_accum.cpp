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

#include "flag_gems/operators.h"

#if defined(FLAGGEMS_USE_CUDA) || defined(FLAGGEMS_USE_IX)

// Shared GemmEx body with the Torch JIT extension
// (src/flag_gems/csrc/wgrad_gemm_accum_kernel.h). Keep a single edit point.
#include "wgrad_gemm_accum_kernel.h"

namespace flag_gems {

void wgrad_gemm_accum_fp32(const at::Tensor &input_2d,
                           const at::Tensor &grad_output_2d,
                           at::Tensor &main_grad) {
  flag_gems_wgrad_detail::wgrad_gemm_accum_fp32_cuda_impl(input_2d, grad_output_2d, main_grad);
}

}  // namespace flag_gems

#else  // !CUDA && !IX

namespace flag_gems {

void wgrad_gemm_accum_fp32(const at::Tensor &input_2d,
                           const at::Tensor &grad_output_2d,
                           at::Tensor &main_grad) {
  (void)input_2d;
  (void)grad_output_2d;
  (void)main_grad;
  TORCH_CHECK(false, "wgrad_gemm_accum_fp32 GemmEx path requires CUDA/IX backend");
}

}  // namespace flag_gems

#endif

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

// JIT entry for wgrad GemmEx. Kernel body lives in wgrad_gemm_accum_kernel.h
// (shared with cpp/lib/wgrad_gemm_accum.cpp).

#include <torch/extension.h>

#include "wgrad_gemm_accum_kernel.h"

void wgrad_gemm_accum_fp32_cuda(const at::Tensor &input_2d,
                                const at::Tensor &grad_output_2d,
                                at::Tensor &main_grad) {
  flag_gems_wgrad_detail::wgrad_gemm_accum_fp32_cuda_impl(input_2d, grad_output_2d, main_grad);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("wgrad_gemm_accum_fp32",
        &wgrad_gemm_accum_fp32_cuda,
        "Apex-aligned wgrad GemmEx into fp32 main_grad",
        py::arg("input_2d"),
        py::arg("grad_output_2d"),
        py::arg("main_grad"));
}

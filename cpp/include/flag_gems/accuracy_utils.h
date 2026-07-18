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

#pragma once

#include <c10/core/ScalarType.h>
#include <torch/torch.h>

namespace flag_gems::accuracy_utils {

extern bool TO_CPU;

struct CheckCloseResult {
  bool ok;
  std::string message;
};

torch::Tensor to_reference(torch::Tensor inp, bool upcast = false);

std::pair<torch::Tensor, torch::Tensor> to_cpu(torch::Tensor res, torch::Tensor ref);

CheckCloseResult gems_assert_close(torch::Tensor res,
                                   torch::Tensor ref,
                                   c10::ScalarType dtype = c10::ScalarType::Undefined,
                                   bool equal_nan = false,
                                   int64_t reduce_dim = 1,
                                   float atol = 1e-4);

CheckCloseResult gems_assert_equal(torch::Tensor res, torch::Tensor ref, bool equal_nan = false);

// Temporary: relax precision for Triton div (no pointwise_dynamic support).
// Will remove once implementation supports pointwise_dynamic.
CheckCloseResult gems_assert_close_div_factor(torch::Tensor res,
                                              torch::Tensor ref,
                                              c10::ScalarType dtype = c10::ScalarType::Undefined,
                                              bool equal_nan = false,
                                              int64_t reduce_dim = 1,
                                              float atol = 1e-4,
                                              bool inplace = false);

}  // namespace flag_gems::accuracy_utils

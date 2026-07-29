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
#include "pointwise_runtime.h"

namespace flag_gems {

// gelu(Tensor self, *, str approximate='none') -> Tensor
at::Tensor gelu(const at::Tensor &self, c10::string_view approximate) {
  if (approximate == "tanh") {
    return pointwise_dynamic::gelu_tanh(self);
  }
  TORCH_CHECK(approximate == "none",
              "gelu: approximate must be 'none' or 'tanh', got '",
              approximate,
              "'");
  return pointwise_dynamic::gelu_none(self);
}

}  // namespace flag_gems

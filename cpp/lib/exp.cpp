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

// exp(Tensor self) -> Tensor
at::Tensor exp(const at::Tensor &self) {
  return pointwise_dynamic::exp_func(self);
}

}  // namespace flag_gems

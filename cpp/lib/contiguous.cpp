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

namespace flag_gems {

at::Tensor contiguous(const at::Tensor &self, at::MemoryFormat memory_format) {
  TORCH_CHECK(memory_format == at::MemoryFormat::Contiguous);
  if (self.is_contiguous(memory_format = memory_format)) {
    return self;
  }
  at::Tensor out = at::empty_like(self, memory_format = memory_format);
  copy_(out, self, false);
  return out;
}
}  // namespace flag_gems

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

#include <gtest/gtest.h>
#include "flag_gems/accuracy_utils.h"
#include "flag_gems/operators.h"
#include "flag_gems/test_utils.h"
#include "torch/torch.h"

TEST(rwkv_op_test, rwkv_mm_sparsity) {
  const torch::Device device = flag_gems::test::default_device();
  const int n = 16384, d = 4096;

  torch::Tensor k = torch::relu(torch::randn({n}, device));
  torch::Tensor v = torch::randn({n, d}, device);

  torch::Tensor ref_k = flag_gems::accuracy_utils::to_reference(k, false);
  torch::Tensor ref_v = flag_gems::accuracy_utils::to_reference(v, false);

  torch::Tensor k2d = ref_k.view({1, n});
  torch::Tensor out_triton = flag_gems::rwkv_mm_sparsity(k, v);
  torch::Tensor out_torch = torch::mm(k2d, ref_v).squeeze(0);

  auto result = flag_gems::accuracy_utils::gems_assert_close(out_triton, out_torch, k.scalar_type(), true);
  EXPECT_TRUE(result.ok) << result.message;
}

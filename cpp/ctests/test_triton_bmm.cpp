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

TEST(BmmTest, bmm) {
  const torch::Device device = flag_gems::test::default_device();
  const int B = 5, M = 256, K = 64, N = 128;

  torch::Tensor batch1 = torch::randn({B, M, K}, device);
  torch::Tensor batch2 = torch::randn({B, K, N}, device);

  torch::Tensor ref_batch1 = flag_gems::accuracy_utils::to_reference(batch1, /*upcast=*/false);
  torch::Tensor ref_batch2 = flag_gems::accuracy_utils::to_reference(batch2, /*upcast=*/false);

  torch::Tensor out_torch = at::bmm(ref_batch1, ref_batch2);
  torch::Tensor out_triton = flag_gems::bmm(batch1, batch2);

  auto result = flag_gems::accuracy_utils::gems_assert_close(out_triton, out_torch, batch1.scalar_type());
  EXPECT_TRUE(result.ok) << result.message;
}

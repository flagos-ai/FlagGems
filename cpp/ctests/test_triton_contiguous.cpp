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

TEST(contiguous_op_test, contiguous) {
  const torch::Device device = flag_gems::test::default_device();
  torch::Tensor inp = torch::randn({10, 10, 10}, device);
  inp = inp.index({torch::indexing::Slice(torch::indexing::None, torch::indexing::None, 2)});
  torch::Tensor ref_inp = flag_gems::accuracy_utils::to_reference(inp);

  EXPECT_FALSE(inp.is_contiguous());
  torch::Tensor ref_out = ref_inp.contiguous();
  torch::Tensor res_out = flag_gems::contiguous(inp);
  EXPECT_TRUE(res_out.is_contiguous());
  EXPECT_EQ(res_out.strides(), ref_out.strides());
  auto result = flag_gems::accuracy_utils::gems_assert_equal(res_out, ref_out);
  EXPECT_TRUE(result.ok) << result.message;
}

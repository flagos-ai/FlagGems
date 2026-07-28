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

#include "flag_gems/accuracy_utils.h"
#include "flag_gems/operators.h"
#include "flag_gems/test_utils.h"
#include "gtest/gtest.h"
#include "torch/torch.h"

TEST(IndexFillTest, ScalarFunctionalAndInplace) {
  const torch::Device device = flag_gems::test::default_device();
  const auto options = torch::TensorOptions().device(device).dtype(torch::kFloat32);
  const torch::Tensor input = torch::randn({4, 8}, options);
  const torch::Tensor index =
      torch::tensor({1, -1}, torch::TensorOptions().device(device).dtype(torch::kLong));

  torch::Tensor expected = input.clone();
  expected.index_fill_(1, index, 3.5);

  const torch::Tensor output = flag_gems::index_fill_scalar(input, 1, index, 3.5);
  auto result = flag_gems::accuracy_utils::gems_assert_close(output, expected);
  EXPECT_TRUE(result.ok) << result.message;

  torch::Tensor inplace = input.clone();
  flag_gems::index_fill_scalar_(inplace, 1, index, 3.5);
  result = flag_gems::accuracy_utils::gems_assert_close(inplace, expected);
  EXPECT_TRUE(result.ok) << result.message;
}

TEST(IndexFillTest, NormalizesNonContiguousIndex) {
  const torch::Device device = flag_gems::test::default_device();
  const auto options = torch::TensorOptions().device(device).dtype(torch::kFloat32);
  const torch::Tensor input = torch::randn({4, 8}, options);
  const torch::Tensor base_index =
      torch::arange(4, torch::TensorOptions().device(device).dtype(torch::kLong));
  const torch::Tensor index = base_index.slice(0, 0, 4, 2);

  torch::Tensor expected = input.clone();
  expected.index_fill_(1, index, -2.0);

  torch::Tensor actual = input.clone();
  flag_gems::index_fill_scalar_(actual, 1, index, -2.0);

  const auto result = flag_gems::accuracy_utils::gems_assert_close(actual, expected);
  EXPECT_TRUE(result.ok) << result.message;
}

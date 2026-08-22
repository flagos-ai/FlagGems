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
#include <torch/torch.h>

#include "flag_gems/accuracy_utils.h"
#include "flag_gems/operators.h"
#include "flag_gems/test_utils.h"

namespace {

using flag_gems::accuracy_utils::gems_assert_close;

TEST(RemainderScalarFirst, DirectScalarTensorRoute) {
  const torch::Device device = flag_gems::test::default_device();
  auto b = torch::randn({10}, torch::TensorOptions(device).dtype(torch::kFloat));
  constexpr double a_scalar = 3.5;

  auto result = flag_gems::remainder_st(a_scalar, b);
  auto b_ref = flag_gems::accuracy_utils::to_reference(b);
  auto ref = torch::remainder(a_scalar, b_ref);

  ASSERT_EQ(result.scalar_type(), ref.scalar_type());
  auto check = gems_assert_close(result, ref);
  EXPECT_TRUE(check.ok) << check.message;
}

TEST(RemainderScalarFirst, BoxedBothZeroDimTensorRoute) {
  const torch::Device device = flag_gems::test::default_device();
  auto a = torch::tensor(7.5, torch::TensorOptions(device).dtype(torch::kFloat));
  auto b = torch::tensor(2.0, torch::TensorOptions(device).dtype(torch::kFloat));

  auto result = flag_gems::remainder(a, b);
  auto a_ref = flag_gems::accuracy_utils::to_reference(a);
  auto b_ref = flag_gems::accuracy_utils::to_reference(b);
  auto ref = torch::remainder(a_ref, b_ref);

  ASSERT_EQ(result.scalar_type(), ref.scalar_type());
  auto check = gems_assert_close(result, ref);
  EXPECT_TRUE(check.ok) << check.message;
}

}  // namespace

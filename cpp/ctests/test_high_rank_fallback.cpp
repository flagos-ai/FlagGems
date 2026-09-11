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

#include <string>
#include <vector>

#include "flag_gems/operators.h"
#include "flag_gems/test_utils.h"
#include "pointwise_manifest.h"

namespace {

TEST(HighRankOps, AbsRankAboveConfiguredLimit) {
  const torch::Device device = flag_gems::test::default_device();
  const int rank = pointwise_dynamic::MAX_RANK + 1;
  std::vector<int64_t> shape(rank, 2);
  auto base = torch::randn(shape, torch::TensorOptions(device).dtype(torch::kFloat));
  auto high_rank = base.select(0, 0).unsqueeze(0).expand(shape);

  ASSERT_EQ(high_rank.dim(), rank);
  ASSERT_FALSE(high_rank.is_contiguous());
  ASSERT_FALSE(high_rank.is_non_overlapping_and_dense());

  try {
    (void)flag_gems::abs(high_rank);
    FAIL() << "expected a rank-limit error";
  } catch (const c10::Error& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("rank"), std::string::npos);
    EXPECT_NE(message.find(std::to_string(rank)), std::string::npos);
    EXPECT_NE(
        message.find(std::to_string(pointwise_dynamic::MAX_RANK)),
        std::string::npos);
  }
}

}  // namespace

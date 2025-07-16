#include <gtest/gtest.h>
#include "c10/util/Logging.h"
#include "flag_gems/operators.h"
#include "torch/torch.h"

class topktest
    : public ::testing::TestWithParam<std::tuple<int64_t, int64_t, int64_t, bool, torch::ScalarType>> {};

TEST_P(topktest, CompareWithPyTorch) {
  const torch::Device device(torch::kCUDA, 0);
  auto [batch_size, hiddensize, topk, largest, dtype] = GetParam();
  auto options = torch::TensorOptions().dtype(dtype).device(device).requires_grad(false);
  torch::Tensor x = torch::arange(0, hiddensize, options).repeat({batch_size, 1});
  for (int64_t i = 0; i < batch_size; ++i) {
    torch::Tensor perm =
        torch::randperm(hiddensize, torch::TensorOptions().dtype(torch::kLong).device(device));
    x[i] = x[i].index({perm});
  }
  auto x_orig = x.clone();

  auto [out_weight_torch, out_index_torch] = at::topk(x, topk, -1, largest, true);
  auto [out_weight_triton, out_index_triton] = flag_gems::topk(x, topk, -1, largest, true);

  EXPECT_TRUE(torch::allclose(out_weight_torch, out_weight_triton));
  EXPECT_TRUE(torch::equal(out_index_torch, out_index_triton));
}

INSTANTIATE_TEST_SUITE_P(special_op_test,
                         topktest,
                         ::testing::Combine(
                             // batch_size: [4, 8]
                             ::testing::Values(4, 8),
                             // hiddensize: [128, 256]
                             ::testing::Values(128, 256),
                             // topk: [5]
                             ::testing::Values(5),
                             // largest: [true, false]
                             ::testing::Values(true, false),
                             // dtype:
                             ::testing::Values(torch::kFloat32, torch::kFloat16, torch::kBFloat16)));

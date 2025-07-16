#include <gtest/gtest.h>
#include "c10/util/Logging.h"
#include "flag_gems/operators.h"
#include "torch/torch.h"

class EmbeddingTest : public ::testing::TestWithParam<
                          std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, bool, torch::ScalarType>> {
};
TEST_P(EmbeddingTest, CompareWithPyTorch) {
  const torch::Device device(torch::kCUDA, 0);
  auto [EmbeddingSize, Batch, M, N, padding_idx, scale_grad_by_freq, dtype] = GetParam();
  auto options = torch::TensorOptions().dtype(dtype).device(device);
  auto indices =
      torch::randint(0,
                     EmbeddingSize,
                     {Batch, M},
                     torch::TensorOptions().device(device).dtype(torch::kLong).requires_grad(false));
  auto embedding = torch::randn({EmbeddingSize, N}, options.requires_grad(true));
  auto out_torch = torch::nn::functional::embedding(indices,
                                                    embedding,
                                                    torch::nn::functional::EmbeddingFuncOptions()
                                                        .padding_idx(padding_idx)
                                                        .scale_grad_by_freq(scale_grad_by_freq)
                                                        .sparse(false));
  auto out_triton = flag_gems::embedding(embedding, indices, padding_idx, scale_grad_by_freq, false);
  EXPECT_TRUE(torch::allclose(out_torch, out_triton));
}
INSTANTIATE_TEST_SUITE_P(embedding_test,
                         EmbeddingTest,
                         ::testing::Combine(
                             // EmbeddingSize: 4096
                             ::testing::Values(4096),
                             // Batch: [2, 4]
                             ::testing::Values(2, 4),
                             // M: [4, 8]
                             ::testing::Values(4, 8),
                             // N: [128, 256, 4096]
                             ::testing::Values(128, 256, 4096),
                             // padding_idx: [None(-1), -1, 1, 2]
                             ::testing::Values(-1, -1, 1, 2),
                             // scale_grad_by_freq: [true, false]
                             ::testing::Values(true, false),
                             // dtype: [kFloat32, kFloat16, kBFloat16]
                             ::testing::Values(torch::kFloat32, torch::kFloat16, torch::kBFloat16)));

class EmbeddingBackwardTest
    : public ::testing::TestWithParam<
          std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, bool, torch::ScalarType>> {};
TEST_P(EmbeddingBackwardTest, FixedValueTest) {
  const torch::Device device(torch::kCUDA, 0);
  auto [EmbeddingSize, Batch, M, N, padding_idx, scale_grad_by_freq, dtype] = GetParam();
  auto options = torch::TensorOptions().dtype(dtype).device(device);
  auto grad = torch::randn({Batch, M, N}, options);
  auto indices =
      torch::randint(0, EmbeddingSize, {Batch, M}, torch::TensorOptions().device(device).dtype(torch::kLong));
  int64_t num_weights = EmbeddingSize;
  bool sparse = false;
  auto torch_in_grad =
      at::embedding_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse);

  auto triton_in_grad =
      flag_gems::embedding_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse);
  EXPECT_TRUE(torch::allclose(torch_in_grad, triton_in_grad));
}
INSTANTIATE_TEST_SUITE_P(embedding_backward_test,
                         EmbeddingBackwardTest,
                         ::testing::Combine(
                             // EmbeddingSize: 4096
                             ::testing::Values(4096),
                             // Batch: [2, 4]
                             ::testing::Values(2, 4),
                             // M: [4, 8]
                             ::testing::Values(4, 8),
                             // N: [128, 256, 4096]
                             ::testing::Values(128, 256, 4096),
                             // padding_idx: [-1, 1, 2]
                             ::testing::Values(-1, 1, 2),
                             // scale_grad_by_freq: [true, false]
                             ::testing::Values(true, false),
                             // dtype: [kFloat32, kFloat16, kBFloat16]
                             ::testing::Values(torch::kFloat32, torch::kFloat16, torch::kBFloat16)));

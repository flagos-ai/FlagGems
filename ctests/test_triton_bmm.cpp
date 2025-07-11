#include <gtest/gtest.h>
#include "flag_gems/operators.h"
#include "torch/torch.h"

TEST(blas_op_test, bmm) {
  const torch::Device device(torch::kCUDA, 0);
  const int B = 5, M = 256, K = 64, N = 128;

  torch::Tensor batch1 = torch::randn({B, M, K}, device);
  torch::Tensor batch2 = torch::randn({B, K, N}, device);

  torch::Tensor out_torch = at::bmm(batch1, batch2);
  torch::Tensor out_triton = flag_gems::bmm(batch1, batch2);

  EXPECT_TRUE(torch::allclose(out_torch, out_triton));
}

#include <gtest/gtest.h>
#include "flag_gems/operators.h"
#include "torch/torch.h"

TEST(rwkv_op_test, rwkv_mm_sparsity) {
  const torch::Device device(torch::kCUDA, 0);
  const int n = 16384, d = 4096;

  torch::Tensor k = torch::relu(torch::randn({n}, device));
  torch::Tensor v = torch::randn({n, d}, device);

  torch::Tensor k2d = k.view({1, n});
  torch::Tensor out_triton = flag_gems::rwkv_mm_sparsity(k, v);
  torch::Tensor out_torch = torch::mm(k2d, v);

  EXPECT_TRUE(torch::allclose(out_torch, out_triton, 1e-3, 1e-3));
}

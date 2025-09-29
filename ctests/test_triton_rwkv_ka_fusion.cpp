#include <gtest/gtest.h>
#include "flag_gems/operators.h"
#include "torch/torch.h"

TEST(rwkv_op_test, rwkv_ka_fusion) {
  const torch::Device device(torch::kCUDA, 0);
  const int T = 1024, H = 8, N = 64;
  const int C = H * N;

  torch::Tensor k = torch::randn({T, C}, device);
  torch::Tensor kk = torch::randn({C}, device);
  torch::Tensor a = torch::randn({T, C}, device);
  torch::Tensor ka = torch::randn({C}, device);

  at::Tensor o_k_triton, o_kk_triton, o_kka_triton;
  std::tie(o_k_triton, o_kk_triton, o_kka_triton) = flag_gems::rwkv_ka_fusion(k, kk, a, ka, H, N);

  torch::Tensor o_kk_torch =
      torch::nn::functional::normalize((k * kk.view({1, C})).view({T, H, N}),
                                       torch::nn::functional::NormalizeFuncOptions().dim(-1).p(2.0))
          .view({T, H * N});
  torch::Tensor o_k_torch = k * (1 + (a - 1) * ka.view({1, C}));
  torch::Tensor o_kka_torch = o_kk_torch * a;

  EXPECT_TRUE(torch::allclose(o_k_torch, o_k_triton, 1e-3, 1e-3));
  EXPECT_TRUE(torch::allclose(o_kk_torch, o_kk_triton, 1e-3, 1e-3));
  EXPECT_TRUE(torch::allclose(o_kka_torch, o_kka_triton, 1e-3, 1e-3));
}

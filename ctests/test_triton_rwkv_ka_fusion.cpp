#include <gtest/gtest.h>
#include "flag_gems/accuracy_utils.h"
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

  auto o_k_result = flag_gems::accuracy_utils::gems_assert_close(o_k_torch, o_k_triton);
  auto o_kk_result = flag_gems::accuracy_utils::gems_assert_close(o_kk_torch, o_kk_triton);
  auto o_kka_result = flag_gems::accuracy_utils::gems_assert_close(o_kka_torch, o_kka_triton);

  EXPECT_TRUE(o_k_result.ok) << o_k_result.message;
  EXPECT_TRUE(o_kk_result.ok) << o_kk_result.message;
  EXPECT_TRUE(o_kka_result.ok) << o_kka_result.message;
}

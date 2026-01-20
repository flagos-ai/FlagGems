#include <gtest/gtest.h>
#include "flag_gems/accuracy_utils.h"
#include "flag_gems/operators.h"
#include "torch/torch.h"

TEST(rwkv_op_test, rwkv_mm_sparsity) {
  const torch::Device device(torch::kCUDA, 0);
  const int n = 16384, d = 4096;

  torch::Tensor k = torch::relu(torch::randn({n}, device));
  torch::Tensor v = torch::randn({n, d}, device);

  torch::Tensor ref_k = flag_gems::accuracy_utils::to_reference(k, true);
  torch::Tensor ref_v = flag_gems::accuracy_utils::to_reference(v, true);

  torch::Tensor k2d = ref_k.view({1, n});
  torch::Tensor out_triton = flag_gems::rwkv_mm_sparsity(k, v);
  torch::Tensor out_torch = torch::mm(k2d, ref_v).squeeze(0);

  auto result = flag_gems::accuracy_utils::gems_assert_close(out_triton, out_torch, k.scalar_type(), true);
  EXPECT_TRUE(result.ok) << result.message;
}

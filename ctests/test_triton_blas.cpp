#include <gtest/gtest.h>
#include "flag_gems/operators.h"
#include "torch/torch.h"

TEST(blas_op_test, mm) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor a = torch::randn({10, 10}, device);
  torch::Tensor b = torch::randn({10, 10}, device);

  torch::Tensor out_torch = at::mm(a, b);
  torch::Tensor out_triton = flag_gems::mm_tensor(a, b);

  EXPECT_TRUE(torch::allclose(out_torch, out_triton));
}

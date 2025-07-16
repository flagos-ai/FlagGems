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

TEST(blas_op_test, addmm) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor bias = torch::randn({10, 10}, device);
  torch::Tensor mat1 = torch::randn({10, 10}, device);
  torch::Tensor mat2 = torch::randn({10, 10}, device);

  torch::Tensor out_torch = at::addmm(bias, mat1, mat2);
  torch::Tensor out_triton = flag_gems::addmm(bias, mat1, mat2);

  EXPECT_TRUE(torch::allclose(out_torch, out_triton));
}

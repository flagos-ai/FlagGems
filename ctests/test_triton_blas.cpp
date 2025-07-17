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

struct BmmTestParam {
  int64_t m;
  int64_t n;
  int64_t k;
  at::ScalarType dtype;
};

class BmmTest : public ::testing::TestWithParam<BmmTestParam> {};

TEST_P(BmmTest, addmm) {
  const BmmTestParam param = GetParam();
  const torch::Device device(torch::kCUDA, 0);
  const at::TensorOptions opt = at::TensorOptions().device(device).dtype(param.dtype);
  const at::Tensor bias = at::randn({param.m, param.n}, opt);
  const at::Tensor mat1 = at::randn({param.m, param.k}, opt);
  const at::Tensor mat2 = at::randn({param.k, param.n}, opt);

  at::Tensor out_torch = at::addmm(bias, mat1, mat2);
  at::Tensor out_triton = flag_gems::addmm(bias, mat1, mat2);

  EXPECT_TRUE(torch::allclose(out_torch, out_triton));
}

INSTANTIATE_TEST_SUITE_P(BmmTests,
                         BmmTest,
                         ::testing::Values(BmmTestParam {10, 10, 10, at::ScalarType::Float},
                                           BmmTestParam {10, 10, 10, at::ScalarType::Half},
                                           BmmTestParam {10, 10, 10, at::ScalarType::BFloat16}));

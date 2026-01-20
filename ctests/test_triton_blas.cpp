#include <gtest/gtest.h>
#include "flag_gems/accuracy_utils.h"
#include "flag_gems/operators.h"
#include "torch/torch.h"

TEST(blas_op_test, mm) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor a = torch::randn({10, 10}, device);
  torch::Tensor b = torch::randn({10, 10}, device);
  torch::Tensor ref_a = flag_gems::accuracy_utils::to_reference(a);
  torch::Tensor ref_b = flag_gems::accuracy_utils::to_reference(b);

  torch::Tensor out_torch = at::mm(ref_a, ref_b);
  torch::Tensor out_triton = flag_gems::mm_tensor(a, b);

  auto result = flag_gems::accuracy_utils::gems_assert_close(out_triton, out_torch, a.scalar_type());
  EXPECT_TRUE(result.ok) << result.message;
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

  const at::Tensor ref_bias = flag_gems::accuracy_utils::to_reference(bias);
  const at::Tensor ref_mat1 = flag_gems::accuracy_utils::to_reference(mat1);
  const at::Tensor ref_mat2 = flag_gems::accuracy_utils::to_reference(mat2);

  at::Tensor out_torch = at::addmm(ref_bias, ref_mat1, ref_mat2);
  at::Tensor out_triton = flag_gems::addmm(bias, mat1, mat2);

  auto result = flag_gems::accuracy_utils::gems_assert_close(out_triton, out_torch, bias.scalar_type());
  EXPECT_TRUE(result.ok) << result.message;
}

INSTANTIATE_TEST_SUITE_P(BmmTests,
                         BmmTest,
                         ::testing::Values(BmmTestParam {10, 10, 10, at::ScalarType::Float},
                                           BmmTestParam {10, 10, 10, at::ScalarType::Half},
                                           BmmTestParam {10, 10, 10, at::ScalarType::BFloat16}));

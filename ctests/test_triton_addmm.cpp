#include <gtest/gtest.h>
#include "flag_gems/accuracy_utils.h"
#include "flag_gems/operators.h"
#include "flag_gems/test_utils.h"
#include "torch/torch.h"

struct AddmmTestParam {
  int64_t m;
  int64_t n;
  int64_t k;
  at::ScalarType dtype;
};

class AddmmTest : public ::testing::TestWithParam<AddmmTestParam> {};

TEST_P(AddmmTest, addmm) {
  const AddmmTestParam param = GetParam();
  const torch::Device device = flag_gems::test::default_device();
  const at::TensorOptions opt = at::TensorOptions().device(device).dtype(param.dtype);
  const at::Tensor bias = at::randn({param.m, param.n}, opt);
  const at::Tensor mat1 = at::randn({param.m, param.k}, opt);
  const at::Tensor mat2 = at::randn({param.k, param.n}, opt);

  const at::Tensor ref_bias = flag_gems::accuracy_utils::to_reference(bias, /*upcast=*/false);
  const at::Tensor ref_mat1 = flag_gems::accuracy_utils::to_reference(mat1, /*upcast=*/false);
  const at::Tensor ref_mat2 = flag_gems::accuracy_utils::to_reference(mat2, /*upcast=*/false);
  at::Tensor out_torch = at::addmm(ref_bias, ref_mat1, ref_mat2);
  at::Tensor out_triton = flag_gems::addmm(bias, mat1, mat2);

  auto result = flag_gems::accuracy_utils::gems_assert_close(out_triton, out_torch, bias.scalar_type());
  EXPECT_TRUE(result.ok) << result.message;
}

INSTANTIATE_TEST_SUITE_P(AddmmTests,
                         AddmmTest,
                         ::testing::Values(AddmmTestParam {10, 10, 10, at::ScalarType::Float},
                                           AddmmTestParam {10, 10, 10, at::ScalarType::Half},
                                           AddmmTestParam {10, 10, 10, at::ScalarType::BFloat16}));

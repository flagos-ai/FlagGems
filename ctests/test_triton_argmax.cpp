#include <gtest/gtest.h>
#include "flag_gems/accuracy_utils.h"
#include "flag_gems/operators.h"
#include "torch/torch.h"

class reduction_op_test : public ::testing::TestWithParam<torch::ScalarType> {};

TEST_P(reduction_op_test, argmax) {
  const torch::Device device(torch::kCUDA, 0);
  auto dtype = GetParam();
  torch::Tensor input = torch::randn({1024, 1024}, device).to(dtype);
  torch::Tensor ref_input = flag_gems::accuracy_utils::to_reference(input);

  torch::Tensor ref_output = at::argmax(ref_input);
  torch::Tensor triton_output = flag_gems::argmax(input);

  auto result = flag_gems::accuracy_utils::gems_assert_equal(triton_output, ref_output);
  EXPECT_TRUE(result.ok) << result.message;
}

TEST_P(reduction_op_test, argmax_dim_specific) {
  const torch::Device device(torch::kCUDA, 0);
  auto dtype = GetParam();
  torch::Tensor input = torch::randn({64, 64, 128}, device).to(dtype);
  torch::Tensor ref_input = flag_gems::accuracy_utils::to_reference(input);

  torch::Tensor ref_dim0 = at::argmax(ref_input, 0);
  torch::Tensor triton_dim0 = flag_gems::argmax(input, 0);

  auto dim0_result = flag_gems::accuracy_utils::gems_assert_equal(triton_dim0, ref_dim0);
  EXPECT_TRUE(dim0_result.ok) << dim0_result.message;

  torch::Tensor ref_dim1 = at::argmax(ref_input, -1);
  torch::Tensor triton_dim1 = flag_gems::argmax(input, -1);

  auto dim1_result = flag_gems::accuracy_utils::gems_assert_equal(triton_dim1, ref_dim1);
  EXPECT_TRUE(dim1_result.ok) << dim1_result.message;
}

TEST_P(reduction_op_test, argmax_keepdim_option) {
  const torch::Device device(torch::kCUDA, 0);
  auto dtype = GetParam();
  torch::Tensor input = torch::randn({2, 4, 64, 64}, device).to(dtype);
  torch::Tensor ref_input = flag_gems::accuracy_utils::to_reference(input);

  torch::Tensor ref_keep = at::argmax(ref_input, 1, true);
  torch::Tensor triton_keep = flag_gems::argmax(input, 1, true);

  auto keep_result = flag_gems::accuracy_utils::gems_assert_equal(triton_keep, ref_keep);
  EXPECT_TRUE(keep_result.ok) << keep_result.message;

  EXPECT_EQ(ref_keep.sizes(), triton_keep.sizes());

  torch::Tensor ref_no_keep = at::argmax(ref_input, 1, false);
  torch::Tensor triton_no_keep = flag_gems::argmax(input, 1, false);

  auto no_keep_result = flag_gems::accuracy_utils::gems_assert_equal(triton_no_keep, ref_no_keep);
  EXPECT_TRUE(no_keep_result.ok) << no_keep_result.message;
}

INSTANTIATE_TEST_SUITE_P(DTypeTests,
                         reduction_op_test,
                         ::testing::Values(torch::kFloat32, torch::kFloat16, torch::kBFloat16));

#include <gtest/gtest.h>
#include "flag_gems/accuracy_utils.h"
#include "flag_gems/operators.h"
#include "torch/torch.h"

TEST(blas_op_test, div) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor a = torch::randn({64, 64}, device);
  torch::Tensor b = torch::randn({1, 64}, device).clamp_min(1e-3);
  torch::Tensor ref_a = flag_gems::accuracy_utils::to_reference(a);
  torch::Tensor ref_b = flag_gems::accuracy_utils::to_reference(b);

  auto out_torch = ref_a / ref_b;
  auto out_triton = flag_gems::true_div(a, b);

  auto result = flag_gems::accuracy_utils::gems_assert_close(out_triton, out_torch, a.scalar_type(), true);
  EXPECT_TRUE(result.ok) << result.message;
}

TEST(blas_op_test, true_div_) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor a = torch::randn({64, 64}, device);
  torch::Tensor b = torch::randn({64, 64}, device).clamp_min(1e-3);
  torch::Tensor ref_a = flag_gems::accuracy_utils::to_reference(a);
  torch::Tensor ref_b = flag_gems::accuracy_utils::to_reference(b);

  torch::Tensor a_clone = a.clone();
  auto out_torch = ref_a / ref_b;
  auto out_inplace = flag_gems::true_div_(a_clone, b);

  auto result = flag_gems::accuracy_utils::gems_assert_close(out_inplace, out_torch, a.scalar_type(), true);
  EXPECT_TRUE(result.ok) << result.message;
}

TEST(blas_op_test, trunc_div) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor a = torch::randn({64, 64}, device);
  torch::Tensor b = torch::randn({1, 64}, device).clamp_min(1e-3);
  torch::Tensor ref_a = flag_gems::accuracy_utils::to_reference(a);
  torch::Tensor ref_b = flag_gems::accuracy_utils::to_reference(b);

  auto out_torch = torch::trunc(ref_a / ref_b);
  auto out_triton = flag_gems::trunc_div(a, b);

  auto result = flag_gems::accuracy_utils::gems_assert_close(out_triton, out_torch, a.scalar_type(), true);
  EXPECT_TRUE(result.ok) << result.message;
}

TEST(blas_op_test, trunc_div_) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor a = torch::randn({64, 64}, device);
  torch::Tensor b = torch::randn({1, 64}, device).clamp_min(1e-3);
  torch::Tensor ref_a = flag_gems::accuracy_utils::to_reference(a);
  torch::Tensor ref_b = flag_gems::accuracy_utils::to_reference(b);

  torch::Tensor a_clone = a.clone();
  auto out_torch = torch::trunc(ref_a / ref_b);
  auto out_inplace = flag_gems::trunc_div_(a_clone, b);

  auto result = flag_gems::accuracy_utils::gems_assert_close(out_inplace, out_torch, a.scalar_type(), true);
  EXPECT_TRUE(result.ok) << result.message;
}

TEST(blas_op_test, floor_div) {
  const torch::Device device(torch::kCUDA, 0);

  torch::Tensor a = torch::randn({64, 64}, device);
  torch::Tensor b = torch::randn({1, 64}, device).clamp_min(1e-3);
  torch::Tensor ref_a = flag_gems::accuracy_utils::to_reference(a);
  torch::Tensor ref_b = flag_gems::accuracy_utils::to_reference(b);

  auto out_torch = torch::floor_divide(ref_a, ref_b);
  auto out_triton = flag_gems::floor_div(a, b);

  auto result = flag_gems::accuracy_utils::gems_assert_close(out_triton, out_torch, a.scalar_type(), true);
  EXPECT_TRUE(result.ok) << result.message;
}

TEST(blas_op_test, floor_div_) {
  const torch::Device device(torch::kCUDA, 0);

  torch::Tensor a = torch::randn({4, 8}, device) * 10;
  torch::Tensor b = torch::randn({1, 8}, device).clamp_min(1e-3);
  torch::Tensor ref_a = flag_gems::accuracy_utils::to_reference(a);
  torch::Tensor ref_b = flag_gems::accuracy_utils::to_reference(b);

  auto out_torch = torch::floor_divide(ref_a, ref_b);
  auto out_triton = flag_gems::floor_div_(a, b);

  auto result = flag_gems::accuracy_utils::gems_assert_close(out_triton, out_torch, a.scalar_type(), true);
  EXPECT_TRUE(result.ok) << result.message;
}

TEST(blas_op_test, div_mode) {
  const torch::Device device(torch::kCUDA, 0);

  torch::Tensor a = torch::randn({64, 64}, device);
  torch::Tensor b = torch::randn({1, 64}, device).clamp_min(1e-3);
  torch::Tensor ref_a = flag_gems::accuracy_utils::to_reference(a);
  torch::Tensor ref_b = flag_gems::accuracy_utils::to_reference(b);

  auto out_torch = at::div(ref_a, ref_b, c10::make_optional<std::string>("floor"));
  auto out_triton = flag_gems::div_mode(a, b, c10::make_optional<std::string>("floor"));

  auto result = flag_gems::accuracy_utils::gems_assert_close(out_triton, out_torch, a.scalar_type(), true);
  EXPECT_TRUE(result.ok) << result.message;
}

TEST(blas_op_test, div_mode_) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor a = torch::randn({64, 64}, device);
  torch::Tensor b = torch::randn({1, 64}, device).clamp_min(1e-3);
  torch::Tensor ref_a = flag_gems::accuracy_utils::to_reference(a);
  torch::Tensor ref_b = flag_gems::accuracy_utils::to_reference(b);

  torch::Tensor torch_out = ref_a.clone();
  torch_out.div_(ref_b, c10::make_optional<std::string>("floor"));
  torch::Tensor triton_out = a.clone();
  flag_gems::div_mode_(triton_out, b, c10::make_optional<std::string>("floor"));

  auto result = flag_gems::accuracy_utils::gems_assert_close(triton_out, torch_out, a.scalar_type(), true);
  EXPECT_TRUE(result.ok) << result.message;
}

TEST(blas_op_test, remainder) {
  const torch::Device device(torch::kCUDA, 0);

  torch::Tensor a = torch::randn({32, 32}, device) * 10;
  torch::Tensor b = torch::randn({32, 32}, device).clamp_min(0.5);
  torch::Tensor ref_a = flag_gems::accuracy_utils::to_reference(a);
  torch::Tensor ref_b = flag_gems::accuracy_utils::to_reference(b);

  auto out_torch = torch::remainder(ref_a, ref_b);
  auto out_triton = flag_gems::remainder(a, b);

  auto result = flag_gems::accuracy_utils::gems_assert_close(out_triton, out_torch, a.scalar_type(), true);
  EXPECT_TRUE(result.ok) << result.message;

  torch::Tensor a_int = torch::randint(-100, 100, {4, 4}, device);
  torch::Tensor b_int = torch::randint(1, 50, {4, 4}, device);
  torch::Tensor ref_a_int = flag_gems::accuracy_utils::to_reference(a_int);
  torch::Tensor ref_b_int = flag_gems::accuracy_utils::to_reference(b_int);

  auto out_torch_int = torch::remainder(ref_a_int, ref_b_int);
  auto out_triton_int = flag_gems::remainder(a_int, b_int);

  auto out_result =
      flag_gems::accuracy_utils::gems_assert_close(out_triton_int, out_torch_int, a.scalar_type(), true);
  EXPECT_TRUE(out_result.ok) << out_result.message;
}

TEST(blas_op_test, remainder_) {
  const torch::Device device(torch::kCUDA, 0);

  torch::Tensor a = torch::randn({32, 32}, device) * 10;
  torch::Tensor b = torch::randn({32, 32}, device).clamp_min(0.5);
  torch::Tensor ref_a = flag_gems::accuracy_utils::to_reference(a);
  torch::Tensor ref_b = flag_gems::accuracy_utils::to_reference(b);

  torch::Tensor a_clone = a.clone();

  auto out_torch = torch::remainder(ref_a, ref_b);
  auto out_triton = flag_gems::remainder_(a_clone, b);

  auto result = flag_gems::accuracy_utils::gems_assert_close(out_triton, out_torch, a.scalar_type(), true);
  EXPECT_TRUE(result.ok) << result.message;
  result = flag_gems::accuracy_utils::gems_assert_close(a_clone, out_triton, a.scalar_type(), true);
  EXPECT_TRUE(result.ok) << result.message;
}

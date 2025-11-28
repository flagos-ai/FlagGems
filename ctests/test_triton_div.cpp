#include <gtest/gtest.h>
#include "flag_gems/operators.h"
#include "torch/torch.h"

TEST(blas_op_test, div) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor a = torch::randn({64, 64}, device);
  torch::Tensor b = torch::randn({1, 64}, device).clamp_min(1e-3);

  auto out_torch = a / b;
  auto out_triton = flag_gems::true_div(a, b);

  EXPECT_TRUE(torch::allclose(out_torch, out_triton, 1e-4, 1e-6));
}

TEST(blas_op_test, true_div_) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor a = torch::randn({64, 64}, device);
  torch::Tensor b = torch::randn({64, 64}, device).clamp_min(1e-3);

  torch::Tensor a_clone = a.clone();
  auto out_torch = a / b;
  auto out_inplace = flag_gems::true_div_(a_clone, b);

  EXPECT_TRUE(torch::allclose(out_torch, out_inplace, 1e-4, 1e-6));
}

TEST(blas_op_test, trunc_div) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor a = torch::randn({64, 64}, device);
  torch::Tensor b = torch::randn({1, 64}, device).clamp_min(1e-3);

  auto out_torch = torch::trunc(a / b);
  auto out_triton = flag_gems::trunc_div(a, b);

  EXPECT_TRUE(torch::allclose(out_torch, out_triton, 1e-4, 1e-6));
}

TEST(blas_op_test, trunc_div_) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor a = torch::randn({64, 64}, device);
  torch::Tensor b = torch::randn({1, 64}, device).clamp_min(1e-3);

  torch::Tensor a_clone = a.clone();
  auto out_torch = torch::trunc(a / b);
  auto out_inplace = flag_gems::trunc_div_(a_clone, b);

  EXPECT_TRUE(torch::allclose(out_torch, out_inplace, 1e-4, 1e-6));
}

TEST(blas_op_test, floor_div) {
  const torch::Device device(torch::kCUDA, 0);

  torch::Tensor a = torch::randn({64, 64}, device);
  torch::Tensor b = torch::randn({1, 64}, device).clamp_min(1e-3);

  auto out_torch = torch::floor_divide(a, b);
  auto out_triton = flag_gems::floor_div(a, b);

  EXPECT_TRUE(torch::allclose(out_torch, out_triton, 1e-4, 1e-6));
}

TEST(blas_op_test, floor_div_) {
  const torch::Device device(torch::kCUDA, 0);

  torch::Tensor a = torch::randn({4, 8}, device) * 10;
  torch::Tensor b = torch::randn({1, 8}, device).clamp_min(1e-3);

  auto out_torch = torch::floor_divide(a, b);
  auto out_triton = flag_gems::floor_div_(a, b);

  EXPECT_TRUE(torch::allclose(out_torch, out_triton, 1e-4, 1e-6));
}

TEST(blas_op_test, div_mode) {
  const torch::Device device(torch::kCUDA, 0);

  torch::Tensor a = torch::randn({64, 64}, device);
  torch::Tensor b = torch::randn({1, 64}, device).clamp_min(1e-3);

  auto out_torch = at::div(a, b, c10::make_optional<std::string>("floor"));
  auto out_triton = flag_gems::div_mode(a, b, c10::make_optional<std::string>("floor"));

  EXPECT_TRUE(torch::allclose(out_torch, out_triton, 1e-4, 1e-6));
}

TEST(blas_op_test, div_mode_) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor a = torch::randn({64, 64}, device);
  torch::Tensor b = torch::randn({1, 64}, device).clamp_min(1e-3);
  torch::Tensor torch_out = a.clone();
  torch_out.div_(b, c10::make_optional<std::string>("floor"));
  torch::Tensor triton_out = a.clone();
  flag_gems::div_mode_(triton_out, b, c10::make_optional<std::string>("floor"));

  EXPECT_TRUE(torch::allclose(torch_out, triton_out, 1e-4, 1e-6));
}

TEST(blas_op_test, remainder) {
  const torch::Device device(torch::kCUDA, 0);

  torch::Tensor a = torch::randn({32, 32}, device) * 10;
  torch::Tensor b = torch::randn({32, 32}, device).clamp_min(0.5);

  auto out_torch = torch::remainder(a, b);
  auto out_triton = flag_gems::remainder(a, b);

  EXPECT_TRUE(torch::allclose(out_torch, out_triton, 1e-4, 1e-6));
}

TEST(blas_op_test, remainder_) {
  const torch::Device device(torch::kCUDA, 0);

  torch::Tensor a = torch::randn({32, 32}, device) * 10;
  torch::Tensor b = torch::randn({32, 32}, device).clamp_min(0.5);

  torch::Tensor a_clone = a.clone();

  auto out_torch = torch::remainder(a, b);

  auto out_triton = flag_gems::remainder_(a_clone, b);

  EXPECT_TRUE(torch::allclose(out_torch, out_triton, 1e-4, 1e-6));
  EXPECT_TRUE(torch::allclose(a_clone, out_triton, 1e-4, 1e-6));
}

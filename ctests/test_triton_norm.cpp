#include <gtest/gtest.h>
#include "flag_gems/operators.h"
#include "torch/torch.h"

class NormOpTest : public ::testing::TestWithParam<torch::Dtype> {};

TEST_P(NormOpTest, rms_norm) {
  torch::manual_seed(0);
  const torch::Device device(torch::kCUDA, 0);
  auto dtype = GetParam();

  torch::Tensor input = torch::randn({4, 8}, torch::TensorOptions().device(device).dtype(dtype));
  torch::Tensor weight = torch::randn({8}, torch::TensorOptions().device(device).dtype(dtype));
  double eps = 1e-5;

  auto compute_ref = [&](const torch::Tensor& input, const torch::Tensor& weight, double eps) {
    auto input_fp32 = input.to(torch::kFloat32);
    auto weight_fp32 = weight.to(torch::kFloat32);

    auto rms = input_fp32.pow(2).mean(-1, true).add(eps).sqrt();
    auto normed = input_fp32 / rms;
    auto out_fp32 = normed * weight_fp32;
    return out_fp32.to(input.scalar_type());
  };

  torch::Tensor out_torch = compute_ref(input, weight, eps);
  torch::Tensor out_triton = flag_gems::rms_norm(input, weight, eps);

  EXPECT_TRUE(torch::allclose(out_torch, out_triton, /*rtol=*/1e-2, /*atol=*/1e-3));
}

TEST_P(NormOpTest, fused_add_rms_norm) {
  torch::manual_seed(0);
  const torch::Device device(torch::kCUDA, 0);
  auto dtype = GetParam();

  torch::Tensor input = torch::randn({4, 8}, torch::TensorOptions().device(device).dtype(dtype));
  torch::Tensor residual = torch::randn_like(input);
  torch::Tensor weight = torch::randn({8}, torch::TensorOptions().device(device).dtype(dtype));
  double eps = 1e-5;

  auto compute_ref = [&](const torch::Tensor& input,
                         const torch::Tensor& residual,
                         const torch::Tensor& weight,
                         double eps) {
    auto input_fp32 = input.to(torch::kFloat32);
    auto residual_fp32 = residual.to(torch::kFloat32);
    auto weight_fp32 = weight.to(torch::kFloat32);

    auto fused = input_fp32 + residual_fp32;
    auto rms = fused.pow(2).mean(-1, true).add(eps).sqrt();
    auto normed = fused / rms;
    auto out_fp32 = normed * weight_fp32;
    return out_fp32.to(input.scalar_type());
  };

  torch::Tensor out_torch = compute_ref(input, residual, weight, eps);
  flag_gems::fused_add_rms_norm(input, residual, weight, eps);
  torch::Tensor out_triton = input;  // The input tensor is modified in-place

  EXPECT_TRUE(torch::allclose(out_torch, out_triton, /*rtol=*/1e-2, /*atol=*/1e-3));
}

INSTANTIATE_TEST_SUITE_P(DTypeTests,
                         NormOpTest,
                         ::testing::Values(torch::kFloat16, torch::kFloat32, torch::kBFloat16));

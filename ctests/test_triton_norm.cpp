#include <gtest/gtest.h>
#include <tuple>
#include <vector>
#include "flag_gems/operators.h"
#include "torch/torch.h"

// Test fixture that accepts input shape and weight shape
class NormOpTest
    : public ::testing::TestWithParam<std::tuple<torch::Dtype, std::vector<int64_t>, std::vector<int64_t>>> {
};
TEST_P(NormOpTest, rms_norm) {
  torch::manual_seed(0);
  const torch::Device device(torch::kCUDA, 0);
  // Extract parameters
  auto params = GetParam();
  auto dtype = std::get<0>(params);
  auto input_shape = std::get<1>(params);
  auto weight_shape = std::get<2>(params);
  torch::Tensor input = torch::randn(input_shape, torch::TensorOptions().device(device).dtype(dtype));
  torch::Tensor weight = torch::randn(weight_shape, torch::TensorOptions().device(device).dtype(dtype));
  double eps = 1e-5;
  auto compute_ref = [&](const torch::Tensor& input, const torch::Tensor& weight, double eps) {
    auto input_fp32 = input.to(torch::kFloat32);
    auto weight_fp32 = weight.to(torch::kFloat32);

    int64_t norm_dims = weight.dim();
    std::vector<int64_t> reduce_dims(norm_dims);
    for (int i = 0; i < norm_dims; ++i) {
      reduce_dims[i] = -(norm_dims - i);
    }

    auto rms = input_fp32.pow(2).mean(reduce_dims, true).add(eps).sqrt();
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
  // Extract parameters
  auto params = GetParam();
  auto dtype = std::get<0>(params);
  auto input_shape = std::get<1>(params);
  auto weight_shape = std::get<2>(params);
  torch::Tensor input = torch::randn(input_shape, torch::TensorOptions().device(device).dtype(dtype));
  torch::Tensor residual = torch::randn_like(input);
  torch::Tensor weight = torch::randn(weight_shape, torch::TensorOptions().device(device).dtype(dtype));
  double eps = 1e-5;
  auto compute_ref = [&](const torch::Tensor& input,
                         const torch::Tensor& residual,
                         const torch::Tensor& weight,
                         double eps) {
    auto input_fp32 = input.to(torch::kFloat32);
    auto residual_fp32 = residual.to(torch::kFloat32);
    auto weight_fp32 = weight.to(torch::kFloat32);
    auto fused = input_fp32 + residual_fp32;
    // For RMS norm, we compute over the last weight.ndim dimensions
    int64_t norm_dims = weight.dim();
    std::vector<int64_t> reduce_dims(norm_dims);
    for (int i = 0; i < norm_dims; ++i) {
      reduce_dims[i] = -(norm_dims - i);
    }
    auto rms = fused.pow(2).mean(reduce_dims, true).add(eps).sqrt();
    auto normed = fused / rms;
    auto out_fp32 = normed * weight_fp32;  // Broadcasting will handle the operation
    return out_fp32.to(input.scalar_type());
  };
  torch::Tensor out_torch = compute_ref(input, residual, weight, eps);
  flag_gems::fused_add_rms_norm(input, residual, weight, eps);
  torch::Tensor out_triton = input;  // The input tensor is modified in-place
  EXPECT_TRUE(torch::allclose(out_torch, out_triton, /*rtol=*/1e-2, /*atol=*/1e-3));
}
// Instantiate with combinations of dtypes, input shapes and weight shapes
INSTANTIATE_TEST_SUITE_P(
    DTypeAndShapeTests,
    NormOpTest,
    ::testing::Values(
        // Original 2D case: input {4, 8}, weight {8}
        std::make_tuple(torch::kFloat16, std::vector<int64_t> {4, 8}, std::vector<int64_t> {8}),
        std::make_tuple(torch::kFloat32, std::vector<int64_t> {4, 8}, std::vector<int64_t> {8}),
        std::make_tuple(torch::kBFloat16, std::vector<int64_t> {4, 8}, std::vector<int64_t> {8}),
        // 3D cases with weight matching last dim
        std::make_tuple(torch::kFloat16, std::vector<int64_t> {2, 4, 8}, std::vector<int64_t> {8}),
        std::make_tuple(torch::kFloat32, std::vector<int64_t> {2, 4, 8}, std::vector<int64_t> {8}),
        // 3D cases with weight matching last 2 dims
        std::make_tuple(torch::kFloat16, std::vector<int64_t> {2, 4, 8}, std::vector<int64_t> {4, 8}),
        std::make_tuple(torch::kFloat32, std::vector<int64_t> {3, 5, 16}, std::vector<int64_t> {5, 16}),
        // 4D cases
        std::make_tuple(torch::kFloat16, std::vector<int64_t> {2, 3, 4, 8}, std::vector<int64_t> {8}),
        std::make_tuple(torch::kFloat32, std::vector<int64_t> {2, 3, 4, 8}, std::vector<int64_t> {4, 8})));

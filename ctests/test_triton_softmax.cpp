#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include "flag_gems/accuracy_utils.h"
#include "flag_gems/operators.h"
#include "gtest/gtest.h"
#include "torch/torch.h"

TEST(TritonSoftmaxTest, ForwardInnerDim) {
  const torch::Device device(torch::kCUDA, 0);
  auto input = torch::randn({4, 16}, device).to(torch::kFloat16);
  int dim = 1;
  int wrapped_dim = at::maybe_wrap_dim(dim, input.dim());

  auto out_gems = flag_gems::softmax(input, wrapped_dim, false);
  auto out_torch = torch::softmax(input.to(torch::kFloat32), wrapped_dim).to(torch::kFloat16);

  auto result = flag_gems::accuracy_utils::gems_assert_close(out_gems, out_torch);
  EXPECT_TRUE(result.ok) << result.message;

  // Sum over each row (dim=1), the sums should all be 1
  auto row_sums = out_gems.sum(wrapped_dim);
  auto ones = torch::ones_like(row_sums);
  result = flag_gems::accuracy_utils::gems_assert_close(row_sums, ones);
  EXPECT_TRUE(result.ok) << result.message;
}

TEST(TritonSoftmaxTest, ForwardNonInnerDim) {
  const torch::Device device(torch::kCUDA, 0);
  auto input = torch::randn({2, 8, 3}, device).to(torch::kFloat16);
  int dim = 1;
  int wrapped_dim = at::maybe_wrap_dim(dim, input.dim());

  auto out_gems = flag_gems::softmax(input, wrapped_dim, false);
  auto out_torch = torch::softmax(input.to(torch::kFloat32), wrapped_dim).to(torch::kFloat16);

  auto result = flag_gems::accuracy_utils::gems_assert_close(out_gems, out_torch);
  EXPECT_TRUE(result.ok) << result.message;
  // Sum along dim=1 for verification
  auto sums = out_gems.sum(wrapped_dim);
  auto ones = torch::ones_like(sums);
  result = flag_gems::accuracy_utils::gems_assert_close(sums, ones);
  EXPECT_TRUE(result.ok) << result.message;
}

TEST(TritonSoftmaxTest, ForwardDim0) {
  const torch::Device device(torch::kCUDA, 0);
  auto input = torch::randn({5, 10}, device).to(torch::kFloat16);
  int dim = 0;
  int wrapped_dim = at::maybe_wrap_dim(dim, input.dim());

  auto out_gems = flag_gems::softmax(input, wrapped_dim, false);
  auto out_torch = torch::softmax(input.to(torch::kFloat32), wrapped_dim).to(torch::kFloat16);

  auto result = flag_gems::accuracy_utils::gems_assert_close(out_gems, out_torch);
  EXPECT_TRUE(result.ok) << result.message;

  // Sum along dim=0 for verification
  auto col_sums = out_gems.sum(wrapped_dim);
  auto ones = torch::ones_like(col_sums);
  result = flag_gems::accuracy_utils::gems_assert_close(col_sums, ones);
  EXPECT_TRUE(result.ok) << result.message;
}

TEST(TritonSoftmaxTest, ForwardNegativeDim) {
  const torch::Device device(torch::kCUDA, 0);
  auto input = torch::randn({2, 4, 8}, device).to(torch::kFloat16);
  int dim = -1;  // negative dim
  int wrapped_dim = at::maybe_wrap_dim(dim, input.dim());

  auto out_gems = flag_gems::softmax(input, wrapped_dim, false);
  auto out_torch = torch::softmax(input.to(torch::kFloat32), wrapped_dim).to(torch::kFloat16);

  auto result = flag_gems::accuracy_utils::gems_assert_close(out_gems, out_torch);
  EXPECT_TRUE(result.ok) << result.message;
}

TEST(TritonSoftmaxTest, BackwardInnerDim) {
  const torch::Device device(torch::kCUDA, 0);
  auto input = torch::randn({4, 16}, device).to(torch::kFloat32).set_requires_grad(true);
  int dim = 1;
  int wrapped_dim = at::maybe_wrap_dim(dim, input.dim());

  auto output_ref = torch::softmax(input, wrapped_dim);
  auto output_triton = flag_gems::softmax(input, wrapped_dim, false);

  auto grad_output = torch::randn_like(output_ref);

  torch::Tensor grad_input_ref;
  {
    pybind11::gil_scoped_release no_gil;  // Release GIL
    grad_input_ref = torch::autograd::grad({output_ref}, {input}, {grad_output})[0];
  }

  auto grad_input_triton =
      flag_gems::softmax_backward(grad_output, output_triton, wrapped_dim, input.scalar_type());

  auto result = flag_gems::accuracy_utils::gems_assert_close(grad_input_triton, grad_input_ref);
  EXPECT_TRUE(result.ok) << result.message;
}

TEST(TritonSoftmaxTest, BackwardNonInnerDim) {
  const torch::Device device(torch::kCUDA, 0);
  auto input = torch::randn({2, 8, 3}, device).to(torch::kFloat32).set_requires_grad(true);
  int dim = 1;
  int wrapped_dim = at::maybe_wrap_dim(dim, input.dim());

  auto output_ref = torch::softmax(input, wrapped_dim);
  auto output_triton = flag_gems::softmax(input, wrapped_dim, false);

  auto grad_output = torch::randn_like(output_ref);

  torch::Tensor grad_input_ref;
  {
    pybind11::gil_scoped_release no_gil;  // Release GIL
    grad_input_ref = torch::autograd::grad({output_ref}, {input}, {grad_output})[0];
  }

  auto grad_input_triton =
      flag_gems::softmax_backward(grad_output, output_triton, wrapped_dim, input.scalar_type());

  auto result = flag_gems::accuracy_utils::gems_assert_close(grad_input_triton, grad_input_ref);
  EXPECT_TRUE(result.ok) << result.message;
}

TEST(TritonSoftmaxTest, BackwardDim0) {
  const torch::Device device(torch::kCUDA, 0);
  auto input = torch::randn({5, 10}, device).to(torch::kFloat32).set_requires_grad(true);
  int dim = 0;
  int wrapped_dim = at::maybe_wrap_dim(dim, input.dim());

  auto output_ref = torch::softmax(input, wrapped_dim);
  auto output_triton = flag_gems::softmax(input, wrapped_dim, false);

  auto grad_output = torch::randn_like(output_ref);

  torch::Tensor grad_input_ref;
  {
    pybind11::gil_scoped_release no_gil;  // Release GIL
    grad_input_ref = torch::autograd::grad({output_ref}, {input}, {grad_output})[0];
  }

  auto grad_input_triton =
      flag_gems::softmax_backward(grad_output, output_triton, wrapped_dim, input.scalar_type());

  auto result = flag_gems::accuracy_utils::gems_assert_close(grad_input_triton, grad_input_ref);
  EXPECT_TRUE(result.ok) << result.message;
}

#include "c10/cuda/CUDAFunctions.h"
#include "flag_gems/operators.h"
#include "gtest/gtest.h"
#include "torch/torch.h"

TEST(FillTest, ScalarFill) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor t = torch::empty({4, 5}, torch::TensorOptions().device(device));
  c10::Scalar val = 3.14;

  torch::Tensor out_gems = flag_gems::fill_scalar(t, val);
  torch::Tensor out_ref = torch::full_like(t, val);

  EXPECT_TRUE(torch::allclose(out_gems, out_ref));
}

TEST(FillTest, TensorFill) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor t = torch::empty({3, 3}, torch::TensorOptions().device(device));
  torch::Tensor val = torch::tensor(7.5, torch::TensorOptions().device(device));

  torch::Tensor out_gems = flag_gems::fill_tensor(t, val);
  torch::Tensor out_ref = torch::full_like(t, val.item<double>());

  EXPECT_TRUE(torch::allclose(out_gems, out_ref));
}

TEST(FillTest, ScalarFillInplace) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor t = torch::empty({2, 2}, torch::TensorOptions().device(device));
  c10::Scalar val = -123;  // Use an integer scalar

  flag_gems::fill_scalar_(t, val);
  torch::Tensor ref = torch::full_like(t, val);

  EXPECT_TRUE(torch::allclose(t, ref));
}

TEST(FillTest, TensorFillInplace) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor t = torch::empty({2, 2}, torch::TensorOptions().device(device));
  torch::Tensor val = torch::tensor(-2.5, torch::TensorOptions().device(device));

  flag_gems::fill_tensor_(t, val);
  torch::Tensor ref = torch::full_like(t, val.item<double>());

  EXPECT_TRUE(torch::allclose(t, ref));
}

TEST(FillTest, EmptyTensor) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor t = torch::empty({0}, torch::TensorOptions().device(device));
  c10::Scalar val = 42;

  torch::Tensor out_gems = flag_gems::fill_scalar(t, val);
  torch::Tensor out_ref = torch::full_like(t, val);

  EXPECT_EQ(out_gems.numel(), 0);
  EXPECT_TRUE(torch::allclose(out_gems, out_ref));
}

TEST(FillTest, DifferentDtypesAndValues) {
  const torch::Device device(torch::kCUDA, 0);

  auto check_dtype_and_value = [&](auto dtype, const c10::Scalar& val) {
    torch::Tensor t = torch::empty({5, 5}, torch::TensorOptions().device(device).dtype(dtype));
    torch::Tensor out = flag_gems::fill_scalar(t, val);
    torch::Tensor ref = torch::full_like(t, val);
    // Use a tolerance for floating point comparisons, and direct comparison for
    // integers
    if (out.is_floating_point()) {
      EXPECT_TRUE(torch::allclose(out, ref));
    } else {
      EXPECT_TRUE(torch::equal(out, ref));
    }
  };

  // Test various combinations of tensor dtypes and scalar value types
  check_dtype_and_value(torch::kFloat32, 3.14f);
  check_dtype_and_value(torch::kFloat64, 3.1415926535);
  check_dtype_and_value(torch::kInt32, 12345);
  check_dtype_and_value(torch::kInt64, static_cast<int64_t>(9876543210));
  // Test filling an int tensor with a float scalar (should truncate)
  check_dtype_and_value(torch::kInt32, 5.99);
}

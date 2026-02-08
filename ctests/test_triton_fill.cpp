#include "c10/cuda/CUDAFunctions.h"
#include "flag_gems/accuracy_utils.h"
#include "flag_gems/operators.h"
#include "gtest/gtest.h"
#include "torch/torch.h"

TEST(FillTest, ScalarFill) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor t = torch::empty({4, 5}, torch::TensorOptions().device(device));
  c10::Scalar val = 3.14;

  torch::Tensor ref_t = flag_gems::accuracy_utils::to_reference(t);

  torch::Tensor out_gems = flag_gems::fill_scalar(t, val);
  torch::Tensor out_ref = torch::full_like(ref_t, val);

  auto result = flag_gems::accuracy_utils::gems_assert_close(out_gems, out_ref);
  EXPECT_TRUE(result.ok) << result.message;
}

TEST(FillTest, TensorFill) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor t = torch::empty({3, 3}, torch::TensorOptions().device(device));
  torch::Tensor val = torch::tensor(7.5, torch::TensorOptions().device(device));

  torch::Tensor ref_t = flag_gems::accuracy_utils::to_reference(t);
  torch::Tensor ref_val = flag_gems::accuracy_utils::to_reference(val);

  torch::Tensor out_gems = flag_gems::fill_tensor(t, val);
  torch::Tensor out_ref = torch::full_like(ref_t, ref_val.item<double>());

  auto result = flag_gems::accuracy_utils::gems_assert_close(out_gems, out_ref);
  EXPECT_TRUE(result.ok) << result.message;
}

TEST(FillTest, ScalarFillInplace) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor t = torch::empty({2, 2}, torch::TensorOptions().device(device));
  c10::Scalar val = -123;  // Use an integer scalar

  torch::Tensor ref_t = flag_gems::accuracy_utils::to_reference(t);

  flag_gems::fill_scalar_(t, val);
  torch::Tensor ref = torch::full_like(ref_t, val);

  auto result = flag_gems::accuracy_utils::gems_assert_close(t, ref);
  EXPECT_TRUE(result.ok) << result.message;
}

TEST(FillTest, TensorFillInplace) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor t = torch::empty({2, 2}, torch::TensorOptions().device(device));
  torch::Tensor val = torch::tensor(-2.5, torch::TensorOptions().device(device));

  torch::Tensor ref_t = flag_gems::accuracy_utils::to_reference(t);
  torch::Tensor ref_val = flag_gems::accuracy_utils::to_reference(val);

  flag_gems::fill_tensor_(t, val);
  torch::Tensor ref = torch::full_like(ref_t, ref_val.item<double>());

  auto result = flag_gems::accuracy_utils::gems_assert_close(t, ref);
  EXPECT_TRUE(result.ok) << result.message;
}

TEST(FillTest, EmptyTensor) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor t = torch::empty({0}, torch::TensorOptions().device(device));
  c10::Scalar val = 42;

  torch::Tensor ref_t = flag_gems::accuracy_utils::to_reference(t);

  torch::Tensor out_gems = flag_gems::fill_scalar(t, val);
  torch::Tensor out_ref = torch::full_like(ref_t, val);

  EXPECT_EQ(out_gems.numel(), 0);
  auto result = flag_gems::accuracy_utils::gems_assert_close(out_gems, out_ref);
  EXPECT_TRUE(result.ok) << result.message;
}

TEST(FillTest, DifferentDtypesAndValues) {
  const torch::Device device(torch::kCUDA, 0);

  auto check_dtype_and_value = [&](auto dtype, const c10::Scalar& val) {
    torch::Tensor t = torch::empty({5, 5}, torch::TensorOptions().device(device).dtype(dtype));

    torch::Tensor ref_t = flag_gems::accuracy_utils::to_reference(t);

    torch::Tensor out = flag_gems::fill_scalar(t, val);
    torch::Tensor ref = torch::full_like(ref_t, val);
    // Use a tolerance for floating point comparisons, and direct comparison for
    // integers
    flag_gems::accuracy_utils::CheckCloseResult result;
    if (out.is_floating_point()) {
      result = flag_gems::accuracy_utils::gems_assert_close(out, ref);
    } else {
      result = flag_gems::accuracy_utils::gems_assert_equal(out, ref);
    }
    EXPECT_TRUE(result.ok) << result.message;
  };

  // Test various combinations of tensor dtypes and scalar value types
  check_dtype_and_value(torch::kFloat32, 3.14f);
  check_dtype_and_value(torch::kFloat64, 3.1415926535);
  check_dtype_and_value(torch::kInt32, 12345);
  check_dtype_and_value(torch::kInt64, static_cast<int64_t>(9876543210));
  // Test filling an int tensor with a float scalar (should truncate)
  check_dtype_and_value(torch::kInt32, 5.99);
}

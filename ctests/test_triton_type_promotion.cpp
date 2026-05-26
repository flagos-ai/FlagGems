#include <gtest/gtest.h>
#include "flag_gems/accuracy_utils.h"
#include "flag_gems/operators.h"
#include "flag_gems/test_utils.h"
#include "torch/torch.h"

TEST(type_promotion_test, add_float32_tensor_float64_scalar) {
  const torch::Device device = flag_gems::test::default_device();

  // Create float32 tensor
  torch::Tensor x = torch::randn({4, 5}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

  // Create float64 scalar
  torch::Tensor z = torch::tensor(1.0, torch::TensorOptions().dtype(torch::kFloat64).device(device));

  // PyTorch reference
  torch::Tensor out_torch = torch::add(x, z);

  // FlagGems implementation
  torch::Tensor out_triton = flag_gems::add_tensor(x, z);

  // Check dtype promotion
  EXPECT_EQ(out_torch.dtype(), torch::kFloat64) << "PyTorch output should be float64";
  EXPECT_EQ(out_triton.dtype(), torch::kFloat64) << "FlagGems output should be float64";

  // Check numerical correctness
  auto result = flag_gems::accuracy_utils::gems_assert_close(out_triton, out_torch);
  EXPECT_TRUE(result.ok) << result.message;
}

TEST(type_promotion_test, add_float64_tensor_float32_scalar) {
  const torch::Device device = flag_gems::test::default_device();

  // Create float64 tensor
  torch::Tensor x = torch::randn({4, 5}, torch::TensorOptions().dtype(torch::kFloat64).device(device));

  // Create float32 scalar
  torch::Tensor z = torch::tensor(1.0f, torch::TensorOptions().dtype(torch::kFloat32).device(device));

  // PyTorch reference
  torch::Tensor out_torch = torch::add(x, z);

  // FlagGems implementation
  torch::Tensor out_triton = flag_gems::add_tensor(x, z);

  // Check dtype promotion
  EXPECT_EQ(out_torch.dtype(), torch::kFloat64) << "PyTorch output should be float64";
  EXPECT_EQ(out_triton.dtype(), torch::kFloat64) << "FlagGems output should be float64";

  // Check numerical correctness
  auto result = flag_gems::accuracy_utils::gems_assert_close(out_triton, out_torch);
  EXPECT_TRUE(result.ok) << result.message;
}

TEST(type_promotion_test, add_mixed_dtype_tensors) {
  const torch::Device device = flag_gems::test::default_device();

  // Create tensors with different dtypes
  torch::Tensor x = torch::randn({4, 5}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  torch::Tensor y = torch::randn({4, 5}, torch::TensorOptions().dtype(torch::kFloat64).device(device));

  // PyTorch reference
  torch::Tensor out_torch = torch::add(x, y);

  // FlagGems implementation
  torch::Tensor out_triton = flag_gems::add_tensor(x, y);

  // Check dtype promotion
  EXPECT_EQ(out_torch.dtype(), torch::kFloat64) << "PyTorch output should be float64";
  EXPECT_EQ(out_triton.dtype(), torch::kFloat64) << "FlagGems output should be float64";

  // Check numerical correctness
  auto result = flag_gems::accuracy_utils::gems_assert_close(out_triton, out_torch);
  EXPECT_TRUE(result.ok) << result.message;
}

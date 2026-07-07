#include <gtest/gtest.h>
#include "flag_gems/accuracy_utils.h"
#include "flag_gems/operators.h"
#include "flag_gems/test_utils.h"
#include "torch/torch.h"

// ==============================================================================
// Tests for add_tensor via pointwise_dynamic dispatch
// ==============================================================================

class AddTest : public ::testing::Test {
 protected:
  const torch::Device device = flag_gems::test::default_device();
};

// Basic: same shape, contiguous (fast path)
TEST_F(AddTest, SameShapeContiguous) {
  auto a = torch::randn({10, 10}, device);
  auto b = torch::randn({10, 10}, device);
  auto result = flag_gems::accuracy_utils::gems_assert_close(flag_gems::add_tensor(a, b), a + b);
  EXPECT_TRUE(result.ok) << result.message;
}

// Rank 1
TEST_F(AddTest, Rank1) {
  auto a = torch::randn({1024}, device);
  auto b = torch::randn({1024}, device);
  auto result = flag_gems::accuracy_utils::gems_assert_close(flag_gems::add_tensor(a, b), a + b);
  EXPECT_TRUE(result.ok) << result.message;
}

// Rank 3
TEST_F(AddTest, Rank3) {
  auto a = torch::randn({4, 5, 6}, device);
  auto b = torch::randn({4, 5, 6}, device);
  auto result = flag_gems::accuracy_utils::gems_assert_close(flag_gems::add_tensor(a, b), a + b);
  EXPECT_TRUE(result.ok) << result.message;
}

// Rank 4
TEST_F(AddTest, Rank4) {
  auto a = torch::randn({2, 3, 4, 5}, device);
  auto b = torch::randn({2, 3, 4, 5}, device);
  auto result = flag_gems::accuracy_utils::gems_assert_close(flag_gems::add_tensor(a, b), a + b);
  EXPECT_TRUE(result.ok) << result.message;
}

// Broadcast: (3,1) + (1,4) -> (3,4)
TEST_F(AddTest, Broadcast2D) {
  auto a = torch::randn({3, 1}, device);
  auto b = torch::randn({1, 4}, device);
  auto out = flag_gems::add_tensor(a, b);
  auto ref = a + b;
  ASSERT_EQ(out.sizes(), ref.sizes());
  auto result = flag_gems::accuracy_utils::gems_assert_close(out, ref);
  EXPECT_TRUE(result.ok) << result.message;
}

// Broadcast: (3,1,5) + (1,4,5) -> (3,4,5)
TEST_F(AddTest, Broadcast3D) {
  auto a = torch::randn({3, 1, 5}, device);
  auto b = torch::randn({1, 4, 5}, device);
  auto out = flag_gems::add_tensor(a, b);
  auto ref = a + b;
  ASSERT_EQ(out.sizes(), ref.sizes());
  auto result = flag_gems::accuracy_utils::gems_assert_close(out, ref);
  EXPECT_TRUE(result.ok) << result.message;
}

// Non-contiguous: transposed tensors
TEST_F(AddTest, NonContiguous) {
  auto a = torch::randn({4, 5}, device).t();  // 5x4, non-contiguous
  auto b = torch::randn({4, 5}, device).t();
  auto result = flag_gems::accuracy_utils::gems_assert_close(flag_gems::add_tensor(a, b), a + b);
  EXPECT_TRUE(result.ok) << result.message;
}

// Large tensor (stress test grid launch)
TEST_F(AddTest, LargeTensor) {
  auto a = torch::randn({1024, 1024}, device);
  auto b = torch::randn({1024, 1024}, device);
  auto result = flag_gems::accuracy_utils::gems_assert_close(flag_gems::add_tensor(a, b), a + b);
  EXPECT_TRUE(result.ok) << result.message;
}

// Empty tensor (numel == 0)
TEST_F(AddTest, EmptyTensor) {
  auto a = torch::randn({0, 4}, device);
  auto b = torch::randn({0, 4}, device);
  auto out = flag_gems::add_tensor(a, b);
  EXPECT_EQ(out.numel(), 0);
  EXPECT_EQ(out.sizes(), a.sizes());
}

// Dtype promotion: int + float -> float
TEST_F(AddTest, DtypePromotion) {
  auto a = torch::randint(0, 10, {10}, torch::TensorOptions(device).dtype(torch::kInt));
  auto b = torch::randn({10}, device);
  auto out = flag_gems::add_tensor(a, b);
  auto ref = a + b;
  EXPECT_EQ(out.scalar_type(), ref.scalar_type());
  auto result = flag_gems::accuracy_utils::gems_assert_close(out, ref);
  EXPECT_TRUE(result.ok) << result.message;
}

// Float16
TEST_F(AddTest, Float16) {
  auto a = torch::randn({10, 10}, torch::TensorOptions(device).dtype(torch::kHalf));
  auto b = torch::randn({10, 10}, torch::TensorOptions(device).dtype(torch::kHalf));
  auto out = flag_gems::add_tensor(a, b);
  auto ref = a + b;
  EXPECT_EQ(out.scalar_type(), torch::kHalf);
  auto result = flag_gems::accuracy_utils::gems_assert_close(out, ref);
  EXPECT_TRUE(result.ok) << result.message;
}

// BFloat16
TEST_F(AddTest, BFloat16) {
  auto a = torch::randn({10, 10}, torch::TensorOptions(device).dtype(torch::kBFloat16));
  auto b = torch::randn({10, 10}, torch::TensorOptions(device).dtype(torch::kBFloat16));
  auto out = flag_gems::add_tensor(a, b);
  auto ref = a + b;
  EXPECT_EQ(out.scalar_type(), torch::kBFloat16);
  auto result = flag_gems::accuracy_utils::gems_assert_close(out, ref);
  EXPECT_TRUE(result.ok) << result.message;
}

// Scalar broadcast: (1,) + (10,10) -> (10,10)
TEST_F(AddTest, ScalarBroadcast) {
  auto a = torch::randn({1}, device);
  auto b = torch::randn({10, 10}, device);
  auto out = flag_gems::add_tensor(a, b);
  auto ref = a + b;
  ASSERT_EQ(out.sizes(), ref.sizes());
  auto result = flag_gems::accuracy_utils::gems_assert_close(out, ref);
  EXPECT_TRUE(result.ok) << result.message;
}

// ==============================================================================
// Tests for add with alpha
// ==============================================================================

// Alpha scaling: a + alpha * b
TEST_F(AddTest, AlphaScaling) {
  auto a = torch::randn({10, 10}, device);
  auto b = torch::randn({10, 10}, device);
  double alpha = 2.5;
  auto out = flag_gems::add_tensor(a, b, alpha);
  auto ref = a + alpha * b;
  auto result = flag_gems::accuracy_utils::gems_assert_close(out, ref);
  EXPECT_TRUE(result.ok) << result.message;
}

// ==============================================================================
// Tests for add_tensor_scalar (Tensor + Scalar)
// ==============================================================================

TEST_F(AddTest, TensorPlusScalar) {
  auto a = torch::randn({10, 10}, device);
  double b = 3.14;
  auto out = flag_gems::add_scalar(a, b);
  auto ref = a + b;
  auto result = flag_gems::accuracy_utils::gems_assert_close(out, ref);
  EXPECT_TRUE(result.ok) << result.message;
}

TEST_F(AddTest, TensorPlusScalarWithAlpha) {
  auto a = torch::randn({10, 10}, device);
  double b = 2.0;
  double alpha = 3.0;
  auto out = flag_gems::add_scalar(a, b, alpha);
  auto ref = a + alpha * b;
  auto result = flag_gems::accuracy_utils::gems_assert_close(out, ref);
  EXPECT_TRUE(result.ok) << result.message;
}

// ==============================================================================
// Tests for add_ (inplace Tensor + Tensor)
// ==============================================================================

TEST_F(AddTest, InplaceTensorTensor) {
  auto a = torch::randn({10, 10}, device);
  auto a_clone = a.clone();
  auto b = torch::randn({10, 10}, device);
  auto ref = a_clone + b;
  flag_gems::add_tensor_inplace(a, b);
  auto result = flag_gems::accuracy_utils::gems_assert_close(a, ref);
  EXPECT_TRUE(result.ok) << result.message;
}

TEST_F(AddTest, InplaceTensorTensorWithAlpha) {
  auto a = torch::randn({10, 10}, device);
  auto a_clone = a.clone();
  auto b = torch::randn({10, 10}, device);
  double alpha = 2.5;
  auto ref = a_clone + alpha * b;
  flag_gems::add_tensor_inplace(a, b, alpha);
  auto result = flag_gems::accuracy_utils::gems_assert_close(a, ref);
  EXPECT_TRUE(result.ok) << result.message;
}

// ==============================================================================
// Tests for add_tensor_scalar_ (inplace Tensor + Scalar)
// ==============================================================================

TEST_F(AddTest, InplaceTensorScalar) {
  auto a = torch::randn({10, 10}, device);
  auto a_clone = a.clone();
  double b = 3.14;
  auto ref = a_clone + b;
  flag_gems::add_scalar_inplace(a, b);
  auto result = flag_gems::accuracy_utils::gems_assert_close(a, ref);
  EXPECT_TRUE(result.ok) << result.message;
}

TEST_F(AddTest, InplaceTensorScalarWithAlpha) {
  auto a = torch::randn({10, 10}, device);
  auto a_clone = a.clone();
  double b = 2.0;
  double alpha = 3.0;
  auto ref = a_clone + alpha * b;
  flag_gems::add_scalar_inplace(a, b, alpha);
  auto result = flag_gems::accuracy_utils::gems_assert_close(a, ref);
  EXPECT_TRUE(result.ok) << result.message;
}

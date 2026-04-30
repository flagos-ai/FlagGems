#include <gtest/gtest.h>
#include "flag_gems/accuracy_utils.h"
#include "flag_gems/operators.h"
#include "flag_gems/test_utils.h"
#include "torch/torch.h"

TEST(Int8MmTest, int8_mm) {
  const torch::Device device = flag_gems::test::default_device();
  constexpr int M = 64;
  constexpr int K = 64;
  constexpr int N = 64;

  torch::Tensor a = torch::randint(-128, 127, {M, K}, torch::kInt8).to(device);
  torch::Tensor b = torch::randint(-128, 127, {K, N}, torch::kInt8).to(device);

  // Reference: int8 x int8 -> int32 via at::mm on promoted type
  torch::Tensor ref = at::_int_mm(a, b);

  // Triton implementation
  torch::Tensor out = flag_gems::int8_mm(a, b);

  auto result = flag_gems::accuracy_utils::gems_assert_equal(out, ref);
  EXPECT_TRUE(result.ok) << result.message;
}

TEST(Int8MmTest, int8_mm_non_square) {
  const torch::Device device = flag_gems::test::default_device();
  constexpr int M = 32;
  constexpr int K = 128;
  constexpr int N = 64;

  torch::Tensor a = torch::randint(-128, 127, {M, K}, torch::kInt8).to(device);
  torch::Tensor b = torch::randint(-128, 127, {K, N}, torch::kInt8).to(device);

  torch::Tensor ref = at::_int_mm(a, b);
  torch::Tensor out = flag_gems::int8_mm(a, b);

  auto result = flag_gems::accuracy_utils::gems_assert_equal(out, ref);
  EXPECT_TRUE(result.ok) << result.message;
}

TEST(Int8MmTest, int8_mm_large) {
  const torch::Device device = flag_gems::test::default_device();
  constexpr int M = 8192;
  constexpr int K = 8192;
  constexpr int N = 8192;

  torch::Tensor a = torch::randint(-128, 127, {M, K}, torch::kInt8).to(device);
  torch::Tensor b = torch::randint(-128, 127, {K, N}, torch::kInt8).to(device);

  torch::Tensor ref = at::_int_mm(a, b);
  torch::Tensor out = flag_gems::int8_mm(a, b);

  auto result = flag_gems::accuracy_utils::gems_assert_equal(out, ref);
  EXPECT_TRUE(result.ok) << result.message;
}

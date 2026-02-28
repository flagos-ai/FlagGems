#include "flag_gems/accuracy_utils.h"
#include "flag_gems/operators.h"
#include "gtest/gtest.h"
#include "torch/torch.h"

TEST(CopyTest, ContiguousTensorCopy) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor t = torch::randn({4, 5}, torch::TensorOptions().device(device).dtype(torch::kFloat32));

  torch::Tensor out_gems = flag_gems::to_copy(t);
  torch::Tensor out_ref = t.clone();

  auto result = flag_gems::accuracy_utils::gems_assert_close(out_gems, out_ref);
  EXPECT_TRUE(result.ok) << result.message;
  EXPECT_EQ(out_gems.dtype(), t.dtype());
}

TEST(CopyTest, ContiguousTensorCopyWithDtype) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor t = torch::randn({3, 3}, torch::TensorOptions().device(device).dtype(torch::kFloat16));

  torch::Tensor out_gems = flag_gems::to_copy(t, torch::kFloat32);
  torch::Tensor out_ref = t.to(torch::kFloat32);

  auto result = flag_gems::accuracy_utils::gems_assert_close(out_gems, out_ref);
  EXPECT_TRUE(result.ok) << result.message;
  EXPECT_EQ(out_gems.dtype(), torch::kFloat32);
}

TEST(CopyTest, NonContiguousTensorCopy) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor t = torch::randn({2, 3, 4}, torch::TensorOptions().device(device));
  torch::Tensor t_transposed = t.transpose(0, 1);

  torch::Tensor out_gems = flag_gems::to_copy(t_transposed);
  torch::Tensor out_ref = t_transposed.clone();

  auto result = flag_gems::accuracy_utils::gems_assert_close(out_gems, out_ref);
  EXPECT_TRUE(result.ok) << result.message;
}

TEST(CopyTest, CopyInplaceContiguous) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor src = torch::randn({5, 5}, torch::TensorOptions().device(device));
  torch::Tensor dst = torch::empty_like(src);

  flag_gems::copy_(dst, src);

  auto result = flag_gems::accuracy_utils::gems_assert_close(dst, src);
  EXPECT_TRUE(result.ok) << result.message;
}

TEST(CopyTest, CopyInplaceNonContiguous) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor src = torch::randn({3, 4, 5}, torch::TensorOptions().device(device));
  torch::Tensor dst = torch::empty({5, 4, 3}, torch::TensorOptions().device(device));
  torch::Tensor src_transposed = src.transpose(0, 2);

  flag_gems::copy_(dst, src_transposed);

  auto result = flag_gems::accuracy_utils::gems_assert_close(dst, src_transposed);
  EXPECT_TRUE(result.ok) << result.message;
}

TEST(CopyTest, CopyBroadcasting) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor src = torch::randn({1, 5}, torch::TensorOptions().device(device));
  torch::Tensor dst = torch::empty({3, 5}, torch::TensorOptions().device(device));

  flag_gems::copy_(dst, src);

  torch::Tensor expected = src.expand_as(dst);
  auto result = flag_gems::accuracy_utils::gems_assert_close(dst, expected);
  EXPECT_TRUE(result.ok) << result.message;
}

TEST(CopyTest, EmptyTensor) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor src = torch::empty({0}, torch::TensorOptions().device(device));
  torch::Tensor dst = torch::empty_like(src);

  flag_gems::copy_(dst, src);

  EXPECT_EQ(dst.numel(), 0);
}

#include "flag_gems/accuracy_utils.h"
#include "flag_gems/operators.h"
#include "flag_gems/test_utils.h"
#include "gtest/gtest.h"
#include "torch/torch.h"

// Declared in xpu_registration.cpp (compiled into libflag_gems.so).
// After any late-loaded library (e.g. torch_xmlir) overrides our PrivateUse1
// kernels, this function re-registers the native XRE3 implementations so they
// win again.
extern "C" void xpu_reregister_kernels();

class CopyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // torch_xmlir is imported by the Triton JIT Python environment and its
    // static constructors override our aten::empty / _copy_from / etc.
    // Re-register our native XRE3 kernels before each test so they are
    // always the active dispatch targets.
    xpu_reregister_kernels();
  }
};

TEST_F(CopyTest, ContiguousTensorCopy) {
  const torch::Device device = flag_gems::test::default_device();
  torch::Tensor t = torch::randn({4, 5}, torch::TensorOptions().device(device).dtype(torch::kFloat32));

  torch::Tensor out_gems = flag_gems::to_copy(t);
  torch::Tensor out_ref = t.clone();

  auto result = flag_gems::accuracy_utils::gems_assert_close(out_gems, out_ref);
  EXPECT_TRUE(result.ok) << result.message;
  EXPECT_EQ(out_gems.dtype(), t.dtype());
}

TEST_F(CopyTest, ContiguousTensorCopyWithDtype) {
  const torch::Device device = flag_gems::test::default_device();
  torch::Tensor t = torch::randn({3, 3}, torch::TensorOptions().device(device).dtype(torch::kFloat16));

  torch::Tensor out_gems = flag_gems::to_copy(t, torch::kFloat32);
  torch::Tensor out_ref = t.to(torch::kFloat32);

  auto result = flag_gems::accuracy_utils::gems_assert_close(out_gems, out_ref);
  EXPECT_TRUE(result.ok) << result.message;
  EXPECT_EQ(out_gems.dtype(), torch::kFloat32);
}

TEST_F(CopyTest, NonContiguousTensorCopy) {
  const torch::Device device = flag_gems::test::default_device();
  torch::Tensor t = torch::randn({2, 3, 4}, torch::TensorOptions().device(device));
  torch::Tensor t_transposed = t.transpose(0, 1);

  torch::Tensor out_gems = flag_gems::to_copy(t_transposed);
  torch::Tensor out_ref = t_transposed.clone();

  auto result = flag_gems::accuracy_utils::gems_assert_close(out_gems, out_ref);
  EXPECT_TRUE(result.ok) << result.message;
}

TEST_F(CopyTest, CopyInplaceContiguous) {
  const torch::Device device = flag_gems::test::default_device();
  torch::Tensor src = torch::randn({5, 5}, torch::TensorOptions().device(device));
  torch::Tensor dst = torch::empty_like(src);

  flag_gems::copy_(dst, src);

  auto result = flag_gems::accuracy_utils::gems_assert_close(dst, src);
  EXPECT_TRUE(result.ok) << result.message;
}

TEST_F(CopyTest, CopyInplaceNonContiguous) {
  const torch::Device device = flag_gems::test::default_device();
  torch::Tensor src = torch::randn({3, 4, 5}, torch::TensorOptions().device(device));
  torch::Tensor dst = torch::empty({5, 4, 3}, torch::TensorOptions().device(device));
  torch::Tensor src_transposed = src.transpose(0, 2);

  flag_gems::copy_(dst, src_transposed);

  auto result = flag_gems::accuracy_utils::gems_assert_close(dst, src_transposed);
  EXPECT_TRUE(result.ok) << result.message;
}

TEST_F(CopyTest, CopyBroadcasting) {
  const torch::Device device = flag_gems::test::default_device();
  torch::Tensor src = torch::randn({1, 5}, torch::TensorOptions().device(device));
  torch::Tensor dst = torch::empty({3, 5}, torch::TensorOptions().device(device));

  flag_gems::copy_(dst, src);

  torch::Tensor expected = src.expand_as(dst);
  auto result = flag_gems::accuracy_utils::gems_assert_close(dst, expected);
  EXPECT_TRUE(result.ok) << result.message;
}

TEST_F(CopyTest, EmptyTensor) {
  const torch::Device device = flag_gems::test::default_device();
  torch::Tensor src = torch::empty({0}, torch::TensorOptions().device(device));
  torch::Tensor dst = torch::empty_like(src);

  flag_gems::copy_(dst, src);

  EXPECT_EQ(dst.numel(), 0);
}

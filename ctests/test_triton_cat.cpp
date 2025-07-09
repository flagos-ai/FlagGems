#include "flag_gems/operators.h"
#include "gtest/gtest.h"
#include "torch/torch.h"

TEST(TritonCatTest, basictest) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor t1 = torch::randn({2, 3}, device);
  torch::Tensor t2 = torch::randn({4, 3}, device);

  torch::Tensor out_torch = torch::cat({t1, t2}, 0);
  torch::Tensor out_gems = flag_gems::cat({t1, t2}, 0);

  EXPECT_TRUE(torch::equal(out_torch, out_gems));
}

TEST(TritonCatTest, 2dimtest) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor t1 = torch::randn({3, 2}, device);
  torch::Tensor t2 = torch::randn({3, 4}, device);

  int dim_to_test = 1;
  torch::Tensor out_torch = torch::cat({t1, t2}, dim_to_test);
  torch::Tensor out_gems = flag_gems::cat({t1, t2}, dim_to_test);

  EXPECT_TRUE(torch::equal(out_gems, out_torch));

  EXPECT_EQ(out_gems.size(0), 3);
  EXPECT_EQ(out_gems.size(1), 6);
}

TEST(TritonCatTest, 3dimtest) {
  const torch::Device device(torch::kCUDA, 0);

  torch::Tensor t1 = torch::randn({3, 2, 4}, device);
  torch::Tensor t2 = torch::randn({3, 5, 4}, device);

  int dim_to_test = 1;

  torch::Tensor out_torch = torch::cat({t1, t2}, dim_to_test);
  torch::Tensor out_gems = flag_gems::cat({t1, t2}, dim_to_test);

  EXPECT_TRUE(torch::equal(out_torch, out_gems));
}

TEST(TritonCatTest, 4dimtest) {
  const torch::Device device(torch::kCUDA, 0);
  auto options = torch::TensorOptions().device(device).dtype(torch::kFloat32);

  torch::Tensor t1 = torch::randn({2, 3, 4, 5}, options);
  torch::Tensor t2 = torch::randn({2, 6, 4, 5}, options);

  int dim_to_test = 1;
  torch::Tensor out_torch = torch::cat({t1, t2}, dim_to_test);
  torch::Tensor out_gems = flag_gems::cat({t1, t2}, dim_to_test);

  EXPECT_TRUE(torch::equal(out_torch, out_gems));
}

TEST(TritonCatTest, IntegerConcatenation) {
  const torch::Device device(torch::kCUDA, 0);
  auto options = torch::TensorOptions().device(device).dtype(torch::kInt32);

  torch::Tensor t1 = torch::randint(0, 100, {2, 3, 4}, options);
  torch::Tensor t2 = torch::randint(0, 100, {2, 3, 4}, options);

  int dim_to_test = 2;
  torch::Tensor out_torch = torch::cat({t1, t2}, dim_to_test);
  torch::Tensor out_gems = flag_gems::cat({t1, t2}, dim_to_test);

  EXPECT_TRUE(torch::equal(out_torch, out_gems));
}

TEST(TritonCatTest, EmptyTensorConcatenation) {
  const torch::Device device(torch::kCUDA, 0);
  auto options = torch::TensorOptions().device(device).dtype(torch::kFloat32);

  torch::Tensor t1 = torch::randn({0, 3}, options);
  torch::Tensor t2 = torch::randn({2, 3}, options);

  torch::Tensor out_torch = torch::cat({t1, t2}, 0);
  torch::Tensor out_gems = flag_gems::cat({t1, t2}, 0);

  EXPECT_TRUE(torch::equal(out_torch, out_gems));

  torch::Tensor t3 = torch::randn({0, 3}, options);
  torch::Tensor t4 = torch::randn({0, 3}, options);

  torch::Tensor out_torch_both_empty = torch::cat({t3, t4}, 0);
  torch::Tensor out_gems_both_empty = flag_gems::cat({t3, t4}, 0);

  EXPECT_TRUE(torch::equal(out_torch_both_empty, out_gems_both_empty));
  EXPECT_EQ(out_gems_both_empty.numel(), 0);
}

TEST(TritonCatTest, 3tensorcat) {
  const torch::Device device(torch::kCUDA, 0);
  auto options = torch::TensorOptions().device(device).dtype(torch::kFloat32);

  torch::Tensor t1 = torch::randn({2, 3}, options);
  torch::Tensor t2 = torch::randn({4, 3}, options);
  torch::Tensor t3 = torch::randn({1, 3}, options);

  int dim_to_test = 0;
  torch::Tensor out_torch = torch::cat({t1, t2, t3}, dim_to_test);
  torch::Tensor out_gems = flag_gems::cat({t1, t2, t3}, dim_to_test);

  EXPECT_TRUE(torch::equal(out_torch, out_gems));
  EXPECT_EQ(out_gems.size(0), 7);
  EXPECT_EQ(out_gems.size(1), 3);
}

TEST(TritonCatTest, HandlesNonContiguousInput) {
  const torch::Device device(torch::kCUDA, 0);
  auto options = torch::TensorOptions().device(device).dtype(torch::kFloat32);

  torch::Tensor t_base = torch::randn({2, 3, 4}, options);

  torch::Tensor t1_non_contiguous = t_base.transpose(1, 2);

  torch::Tensor t2_contiguous = torch::randn({2, 4, 5}, options);

  ASSERT_FALSE(t1_non_contiguous.is_contiguous());

  int dim_to_test = 2;
  torch::Tensor out_torch = torch::cat({t1_non_contiguous, t2_contiguous}, dim_to_test);
  torch::Tensor out_gems = flag_gems::cat({t1_non_contiguous, t2_contiguous}, dim_to_test);

  EXPECT_TRUE(torch::equal(out_torch, out_gems));

  EXPECT_EQ(out_gems.size(0), 2);
  EXPECT_EQ(out_gems.size(1), 4);
  EXPECT_EQ(out_gems.size(2), 8);
}

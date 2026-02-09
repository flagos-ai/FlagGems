#include <iostream>
#include <limits>
#include <tuple>
#include "flag_gems/accuracy_utils.h"
#include "flag_gems/operators.h"
#include "gtest/gtest.h"
#include "torch/torch.h"
TEST(TensorSortTest, Basic1DAscending) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor input = torch::tensor({5.0, 3.0, 1.0, 4.0, 2.0}, device);
  auto [values_ref, indices_ref] = torch::sort(input);
  auto [values_custom, indices_custom] = flag_gems::sort(input);
  auto value_result = flag_gems::accuracy_utils::gems_assert_equal(values_custom, values_ref);
  auto indice_result = flag_gems::accuracy_utils::gems_assert_equal(indices_custom, indices_ref);
  torch::Tensor expected = torch::tensor({1.0, 2.0, 3.0, 4.0, 5.0}, device);
  auto vc_result = flag_gems::accuracy_utils::gems_assert_equal(values_custom, expected);

  EXPECT_TRUE(value_result.ok) << value_result.message;
  EXPECT_TRUE(indice_result.ok) << indice_result.message;
  EXPECT_TRUE(vc_result.ok) << vc_result.message;
}
TEST(TensorSortTest, Basic1DDescending) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor input = torch::tensor({5.0, 3.0, 1.0, 4.0, 2.0}, device);
  auto [values_ref, indices_ref] = torch::sort(input, 0, true);
  auto [values_custom, indices_custom] = flag_gems::sort(input, 0, true);
  auto value_result = flag_gems::accuracy_utils::gems_assert_equal(values_custom, values_ref);
  auto indice_result = flag_gems::accuracy_utils::gems_assert_equal(indices_custom, indices_ref);
  torch::Tensor expected = torch::tensor({5.0, 4.0, 3.0, 2.0, 1.0}, device);
  auto vc_result = flag_gems::accuracy_utils::gems_assert_equal(values_custom, expected);

  EXPECT_TRUE(value_result.ok) << value_result.message;
  EXPECT_TRUE(indice_result.ok) << indice_result.message;
  EXPECT_TRUE(vc_result.ok) << vc_result.message;
}

TEST(TensorSortTest, Basic2DLastDimAscending) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor input = torch::tensor(
      {
          {5.0, 3.0, 1.0},
          {4.0, 2.0, 6.0},
          {9.0, 7.0, 8.0}
  },
      device);
  auto [values_ref, indices_ref] = torch::sort(input, -1, false);

  auto [values_custom, indices_custom] = flag_gems::sort(input, -1, false);
  auto value_result = flag_gems::accuracy_utils::gems_assert_equal(values_custom, values_ref);
  auto indice_result = flag_gems::accuracy_utils::gems_assert_equal(indices_custom, indices_ref);

  torch::Tensor expected = torch::tensor(
      {
          {1.0, 3.0, 5.0},
          {2.0, 4.0, 6.0},
          {7.0, 8.0, 9.0}
  },
      device);
  auto vc_result = flag_gems::accuracy_utils::gems_assert_equal(values_custom, expected);
  EXPECT_TRUE(value_result.ok) << value_result.message;
  EXPECT_TRUE(indice_result.ok) << indice_result.message;
  EXPECT_TRUE(vc_result.ok) << vc_result.message;
}

TEST(TensorSortTest, Basic2DFirstDimDescending) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor input = torch::tensor(
      {
          {5.0, 3.0, 1.0},
          {4.0, 2.0, 6.0},
          {9.0, 7.0, 8.0}
  },
      device);
  auto [values_ref, indices_ref] = torch::sort(input, 0, true);
  auto [values_custom, indices_custom] = flag_gems::sort(input, 0, true);
  torch::Tensor expected = torch::tensor(
      {
          {9.0, 7.0, 8.0},
          {5.0, 3.0, 6.0},
          {4.0, 2.0, 1.0}
  },
      device);
  auto result = flag_gems::accuracy_utils::gems_assert_equal(values_custom, expected);
  EXPECT_TRUE(result.ok) << result.message;
}

TEST(TensorSortTest, 3DTensor) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor input = torch::tensor(
                            {
                                {{5, 3}, {1, 4}},
                                {{9, 2}, {7, 8}},
                                {{6, 0}, {3, 1}}
  },
                            device)
                            .to(torch::kFloat);

  {
    auto [values_ref, indices_ref] = torch::sort(input, -1);
    auto [values_custom, indices_custom] = flag_gems::sort(input, -1);

    auto value_result = flag_gems::accuracy_utils::gems_assert_equal(values_custom, values_ref);
    auto indice_result = flag_gems::accuracy_utils::gems_assert_equal(indices_custom, indices_ref);

    EXPECT_TRUE(value_result.ok) << value_result.message;
    EXPECT_TRUE(indice_result.ok) << indice_result.message;
  }

  {
    auto [values_ref, indices_ref] = torch::sort(input, 0);
    auto [values_custom, indices_custom] = flag_gems::sort(input, 0);

    auto value_result = flag_gems::accuracy_utils::gems_assert_equal(values_custom, values_ref);
    auto indice_result = flag_gems::accuracy_utils::gems_assert_equal(indices_custom, indices_ref);

    EXPECT_TRUE(value_result.ok) << value_result.message;
    EXPECT_TRUE(indice_result.ok) << indice_result.message;
  }

  {
    auto [values_ref, indices_ref] = torch::sort(input, 1);
    auto [values_custom, indices_custom] = flag_gems::sort(input, 1);

    auto value_result = flag_gems::accuracy_utils::gems_assert_equal(values_custom, values_ref);
    auto indice_result = flag_gems::accuracy_utils::gems_assert_equal(indices_custom, indices_ref);

    EXPECT_TRUE(value_result.ok) << value_result.message;
    EXPECT_TRUE(indice_result.ok) << indice_result.message;
  }
}

TEST(TensorSortTest, EmptyTensor) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor input = torch::empty({0}, torch::dtype(torch::kFloat).device(device));
  auto [values_ref, indices_ref] = torch::sort(input);
  auto [values_custom, indices_custom] = flag_gems::sort(input);
  auto value_result = flag_gems::accuracy_utils::gems_assert_equal(values_custom, values_ref);
  auto indice_result = flag_gems::accuracy_utils::gems_assert_equal(indices_custom, indices_ref);
  EXPECT_EQ(values_custom.numel(), 0);

  EXPECT_TRUE(value_result.ok) << value_result.message;
  EXPECT_TRUE(indice_result.ok) << indice_result.message;
}

TEST(TensorSortTest, SingleElement) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor input = torch::tensor({42.0}, device);
  auto [values_ref, indices_ref] = torch::sort(input);
  auto [values_custom, indices_custom] = flag_gems::sort(input);
  auto value_result = flag_gems::accuracy_utils::gems_assert_equal(values_custom, values_ref);
  auto indice_result = flag_gems::accuracy_utils::gems_assert_equal(indices_custom, indices_ref);
  auto vc_result = flag_gems::accuracy_utils::gems_assert_equal(values_custom, input);

  EXPECT_TRUE(value_result.ok) << value_result.message;
  EXPECT_TRUE(indice_result.ok) << indice_result.message;
  EXPECT_TRUE(vc_result.ok) << vc_result.message;
}

TEST(TensorSortTest, NegativeValues) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor input = torch::tensor({-5.0, -3.0, -1.0, -4.0, -2.0}, device);
  auto [values_ref, indices_ref] = torch::sort(input);
  auto [values_custom, indices_custom] = flag_gems::sort(input);
  auto value_result = flag_gems::accuracy_utils::gems_assert_equal(values_custom, values_ref);
  auto indice_result = flag_gems::accuracy_utils::gems_assert_equal(indices_custom, indices_ref);
  torch::Tensor expected = torch::tensor({-5.0, -4.0, -3.0, -2.0, -1.0}, device);
  auto vc_result = flag_gems::accuracy_utils::gems_assert_equal(values_custom, expected);

  EXPECT_TRUE(value_result.ok) << value_result.message;
  EXPECT_TRUE(indice_result.ok) << indice_result.message;
  EXPECT_TRUE(vc_result.ok) << vc_result.message;
}

TEST(TensorSortTest, MixedPositiveNegative) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor input = torch::tensor({5.0, -3.0, 0.0, -4.0, 2.0}, device);
  auto [values_ref, indices_ref] = torch::sort(input);
  auto [values_custom, indices_custom] = flag_gems::sort(input);
  auto value_result = flag_gems::accuracy_utils::gems_assert_equal(values_custom, values_ref);
  auto indice_result = flag_gems::accuracy_utils::gems_assert_equal(indices_custom, indices_ref);
  torch::Tensor expected = torch::tensor({-4.0, -3.0, 0.0, 2.0, 5.0}, device);
  auto vc_result = flag_gems::accuracy_utils::gems_assert_equal(values_custom, expected);

  EXPECT_TRUE(value_result.ok) << value_result.message;
  EXPECT_TRUE(indice_result.ok) << indice_result.message;
  EXPECT_TRUE(vc_result.ok) << vc_result.message;
}

TEST(TensorSortTest, LargeTensor) {
  const torch::Device device(torch::kCUDA, 0);
  const int64_t size = 10000;
  torch::Tensor input = torch::randn({size}, device);
  auto [values_ref, indices_ref] = torch::sort(input);
  auto [values_custom, indices_custom] = flag_gems::sort(input);
  auto value_result = flag_gems::accuracy_utils::gems_assert_equal(values_custom, values_ref);
  torch::Tensor gathered = input.gather(0, indices_custom);
  auto vc_result = flag_gems::accuracy_utils::gems_assert_equal(gathered, values_custom);
  torch::Tensor diff = values_custom.diff();
  EXPECT_GE(diff.min().item<float>(), 0);
  EXPECT_TRUE(value_result.ok) << value_result.message;
  EXPECT_TRUE(vc_result.ok) << vc_result.message;
}

TEST(TensorSortStableTest, Basic1DAscending) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor input = torch::tensor({5.0, 3.0, 1.0, 4.0, 2.0}, device);
  auto [values_ref, indices_ref] = torch::sort(input, false);
  auto [values_custom, indices_custom] = flag_gems::sort_stable(input, false);
  auto value_result = flag_gems::accuracy_utils::gems_assert_equal(values_custom, values_ref);
  auto indice_result = flag_gems::accuracy_utils::gems_assert_equal(indices_custom, indices_ref);
  torch::Tensor expected = torch::tensor({1.0, 2.0, 3.0, 4.0, 5.0}, device);
  auto vc_result = flag_gems::accuracy_utils::gems_assert_equal(values_custom, expected);

  EXPECT_TRUE(value_result.ok) << value_result.message;
  EXPECT_TRUE(indice_result.ok) << indice_result.message;
  EXPECT_TRUE(vc_result.ok) << vc_result.message;
}

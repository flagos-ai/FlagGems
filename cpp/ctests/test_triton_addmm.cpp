// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>
#include <algorithm>
#include <limits>
#include <tuple>
#include "flag_gems/accuracy_utils.h"
#include "flag_gems/operators.h"
#include "flag_gems/test_utils.h"
#include "torch/torch.h"

struct AddmmTestParam {
  int64_t m;
  int64_t n;
  int64_t k;
  at::ScalarType dtype;
};

class AddmmTest : public ::testing::TestWithParam<AddmmTestParam> {};

TEST_P(AddmmTest, addmm) {
  const AddmmTestParam param = GetParam();
  const torch::Device device = flag_gems::test::default_device();
  const at::TensorOptions opt = at::TensorOptions().device(device).dtype(param.dtype);
  const at::Tensor bias = at::randn({param.m, param.n}, opt);
  const at::Tensor mat1 = at::randn({param.m, param.k}, opt);
  const at::Tensor mat2 = at::randn({param.k, param.n}, opt);

  auto ref_dtype = (param.dtype == at::ScalarType::Double) ? at::kDouble : at::kFloat;
  const at::Tensor ref_bias = bias.to(ref_dtype);
  const at::Tensor ref_mat1 = mat1.to(ref_dtype);
  const at::Tensor ref_mat2 = mat2.to(ref_dtype);
  at::Tensor out_torch = at::addmm(ref_bias, ref_mat1, ref_mat2).to(param.dtype);
  at::Tensor out_triton = flag_gems::addmm(bias, mat1, mat2);

  auto result = flag_gems::accuracy_utils::gems_assert_close(out_triton,
                                                             out_torch,
                                                             bias.scalar_type(),
                                                             /*equal_nan=*/false,
                                                             /*reduce_dim=*/param.k);
  EXPECT_TRUE(result.ok) << result.message;
}

INSTANTIATE_TEST_SUITE_P(AddmmTests,
                         AddmmTest,
                         ::testing::Values(AddmmTestParam {10, 10, 10, at::ScalarType::Float},
                                           AddmmTestParam {10, 10, 10, at::ScalarType::Half},
                                           AddmmTestParam {10, 10, 10, at::ScalarType::BFloat16}));

namespace {

enum class BiasLayout { Scalar, Vector, Row, Column, Matrix };

at::Tensor make_bias(BiasLayout layout, int64_t m, int64_t n, const at::TensorOptions& options) {
  switch (layout) {
    case BiasLayout::Scalar:
      return at::randn({}, options);
    case BiasLayout::Vector:
      return at::randn({n * 2}, options).slice(0, 0, n * 2, 2);
    case BiasLayout::Row:
      return at::randn({1, n * 2}, options).slice(1, 0, n * 2, 2);
    case BiasLayout::Column:
      return at::randn({m, 1}, options);
    case BiasLayout::Matrix:
      return at::randn({m, n * 2}, options).slice(1, 0, n * 2, 2);
  }
  TORCH_CHECK(false, "Unknown bias layout");
}

void expect_addmm_matches_reference(const at::Tensor& bias,
                                    const at::Tensor& mat1,
                                    const at::Tensor& mat2,
                                    const at::Scalar& beta,
                                    const at::Scalar& alpha) {
  const auto dtype = mat1.scalar_type();
  const auto reference_dtype = dtype == at::kDouble ? at::kDouble : at::kFloat;
  const auto reference =
      at::addmm(bias.to(reference_dtype), mat1.to(reference_dtype), mat2.to(reference_dtype), beta, alpha)
          .to(dtype);
  const auto output = flag_gems::addmm(bias, mat1, mat2, beta, alpha);
  const auto result =
      flag_gems::accuracy_utils::gems_assert_close(output,
                                                   reference,
                                                   dtype,
                                                   /*equal_nan=*/false,
                                                   /*reduce_dim=*/std::max<int64_t>(mat1.size(1), 1));
  EXPECT_TRUE(result.ok) << result.message;

  // The out launch must also honor a non-contiguous output's strides.
  const int64_t m = mat1.size(0), n = mat2.size(1);
  auto out = at::empty({m, n * 2}, mat1.options()).slice(1, 0, n * 2, 2);
  void* const out_data = out.data_ptr();
  const auto& returned = flag_gems::addmm_out(bias, mat1, mat2, beta, alpha, out);
  EXPECT_EQ(returned.data_ptr(), out_data);
  const auto out_result =
      flag_gems::accuracy_utils::gems_assert_close(out,
                                                   reference,
                                                   dtype,
                                                   /*equal_nan=*/false,
                                                   /*reduce_dim=*/std::max<int64_t>(mat1.size(1), 1));
  EXPECT_TRUE(out_result.ok) << out_result.message;
}

}  // namespace

class AddmmBiasTest : public ::testing::TestWithParam<std::tuple<at::ScalarType, BiasLayout>> {};

TEST_P(AddmmBiasTest, BiasLayoutsAndOutVariant) {
  const auto [dtype, layout] = GetParam();
  constexpr int64_t m = 5, n = 7, k = 9;
  const auto options = at::TensorOptions().device(flag_gems::test::default_device()).dtype(dtype);
  torch::manual_seed(0);
  const auto bias = make_bias(layout, m, n, options);
  const auto mat1 = at::randn({m, k}, options);
  const auto mat2 = at::randn({k, n}, options);
  expect_addmm_matches_reference(bias, mat1, mat2, /*beta=*/1.25, /*alpha=*/0.5);
}

INSTANTIATE_TEST_SUITE_P(AddmmBiasTests,
                         AddmmBiasTest,
                         ::testing::Combine(::testing::Values(at::kFloat, at::kHalf, at::kBFloat16),
                                            ::testing::Values(BiasLayout::Scalar,
                                                              BiasLayout::Vector,
                                                              BiasLayout::Row,
                                                              BiasLayout::Column,
                                                              BiasLayout::Matrix)));

TEST(AddmmEdgeCases, EmptyReduction) {
  constexpr int64_t m = 5, n = 7;
  for (auto dtype : {at::kFloat, at::kHalf, at::kBFloat16}) {
    const auto options = at::TensorOptions().device(flag_gems::test::default_device()).dtype(dtype);
    const auto mat1 = at::empty({m, 0}, options);
    const auto mat2 = at::empty({0, n}, options);
    for (auto layout : {BiasLayout::Scalar, BiasLayout::Vector, BiasLayout::Matrix}) {
      const auto bias = make_bias(layout, m, n, options);
      expect_addmm_matches_reference(bias, mat1, mat2, /*beta=*/1.25, /*alpha=*/0.5);
    }
  }
}

TEST(AddmmEdgeCases, ZeroBetaIgnoresNonFiniteBias) {
  constexpr int64_t m = 5, n = 7;
  const auto options = at::TensorOptions().device(flag_gems::test::default_device()).dtype(at::kFloat);
  for (int64_t k : {0, 9}) {
    const auto mat1 = at::randn({m, k}, options);
    const auto mat2 = at::randn({k, n}, options);
    for (auto layout : {BiasLayout::Scalar, BiasLayout::Vector, BiasLayout::Matrix}) {
      auto bias = make_bias(layout, m, n, options);
      bias.fill_(std::numeric_limits<float>::quiet_NaN());
      expect_addmm_matches_reference(bias, mat1, mat2, /*beta=*/0, /*alpha=*/0.5);
    }
  }
}

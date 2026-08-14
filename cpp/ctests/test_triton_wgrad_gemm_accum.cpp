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
#include "flag_gems/accuracy_utils.h"
#include "flag_gems/operators.h"
#include "flag_gems/test_utils.h"
#include "torch/torch.h"

namespace {

struct WgradParam {
  int64_t k;
  int64_t in_features;
  int64_t out_features;
  at::ScalarType act_dtype;
};

class WgradGemmAccumFp32Test : public ::testing::TestWithParam<WgradParam> {};

TEST_P(WgradGemmAccumFp32Test, accum_matches_torch_fp32_ref) {
  const WgradParam param = GetParam();
  const torch::Device device = flag_gems::test::default_device();
  const at::TensorOptions act_opt = at::TensorOptions().device(device).dtype(param.act_dtype);
  const at::TensorOptions fp32_opt = at::TensorOptions().device(device).dtype(at::kFloat);

  const at::Tensor input = at::randn({param.k, param.in_features}, act_opt);
  const at::Tensor grad_output = at::randn({param.k, param.out_features}, act_opt);
  const at::Tensor main_seed = at::randn({param.out_features, param.in_features}, fp32_opt);

  at::Tensor main_ref = main_seed.clone();
  {
    const at::Tensor input_f = input.to(at::kFloat).contiguous();
    const at::Tensor grad_f = grad_output.to(at::kFloat).contiguous();
    main_ref.add_(grad_f.t().contiguous().mm(input_f));
  }

  at::Tensor main_out = main_seed.clone();
  flag_gems::wgrad_gemm_accum_fp32(input, grad_output, main_out);

  auto result = flag_gems::accuracy_utils::gems_assert_close(main_out,
                                                             main_ref,
                                                             at::kFloat,
                                                             /*equal_nan=*/false,
                                                             /*reduce_dim=*/param.k);
  EXPECT_TRUE(result.ok) << result.message;
}

TEST(WgradGemmAccumFp32Test, zero_k_is_noop) {
  const torch::Device device = flag_gems::test::default_device();
  const at::TensorOptions act_opt = at::TensorOptions().device(device).dtype(at::kHalf);
  const at::TensorOptions fp32_opt = at::TensorOptions().device(device).dtype(at::kFloat);

  const at::Tensor input = at::empty({0, 16}, act_opt);
  const at::Tensor grad_output = at::empty({0, 32}, act_opt);
  at::Tensor main_grad = at::randn({32, 16}, fp32_opt);
  const at::Tensor main_before = main_grad.clone();

  flag_gems::wgrad_gemm_accum_fp32(input, grad_output, main_grad);
  EXPECT_TRUE(at::equal(main_grad, main_before));
}

TEST(WgradGemmAccumFp32Test, zero_in_features_raises) {
  const torch::Device device = flag_gems::test::default_device();
  const at::TensorOptions act_opt = at::TensorOptions().device(device).dtype(at::kHalf);
  const at::TensorOptions fp32_opt = at::TensorOptions().device(device).dtype(at::kFloat);

  const at::Tensor input = at::empty({8, 0}, act_opt);
  const at::Tensor grad_output = at::randn({8, 32}, act_opt);
  at::Tensor main_grad = at::empty({32, 0}, fp32_opt);

  EXPECT_THROW(flag_gems::wgrad_gemm_accum_fp32(input, grad_output, main_grad), c10::Error);
}

TEST(WgradGemmAccumFp32Test, zero_out_features_raises) {
  const torch::Device device = flag_gems::test::default_device();
  const at::TensorOptions act_opt = at::TensorOptions().device(device).dtype(at::kHalf);
  const at::TensorOptions fp32_opt = at::TensorOptions().device(device).dtype(at::kFloat);

  const at::Tensor input = at::randn({8, 16}, act_opt);
  const at::Tensor grad_output = at::empty({8, 0}, act_opt);
  at::Tensor main_grad = at::empty({0, 16}, fp32_opt);

  EXPECT_THROW(flag_gems::wgrad_gemm_accum_fp32(input, grad_output, main_grad), c10::Error);
}

INSTANTIATE_TEST_SUITE_P(
    WgradShapes,
    WgradGemmAccumFp32Test,
    ::testing::Values(WgradParam {8, 16, 32, at::ScalarType::Half},
                      WgradParam {8, 16, 32, at::ScalarType::Float},
                      WgradParam {4, 32, 64, at::ScalarType::BFloat16}));

}  // namespace

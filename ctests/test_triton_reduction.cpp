#include <gtest/gtest.h>
#include "c10/util/Logging.h"
#include "flag_gems/operators.h"
#include "torch/torch.h"

TEST(reduction_op_test, sum) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor a = torch::randn({32, 1024}, device);

  torch::Tensor out_torch = at::sum(a);
  torch::Tensor out_triton = flag_gems::sum(a);
  if (!torch::allclose(out_torch, out_triton, 1e-5, 1e-8)) {
    LOG(INFO) << "Difference:\n" << out_torch - out_triton;
  }

  EXPECT_TRUE(torch::allclose(out_torch, out_triton, 1e-5, 1e-8));
}

TEST(reduction_op_test, sum_dim_to_sum) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor a = torch::randn({32, 1024}, device);

  torch::Tensor out_torch = at::sum(a, {at::IntArrayRef {}}, false, c10::nullopt);
  torch::Tensor out_triton = flag_gems::sum_dim(a, {at::IntArrayRef {}}, false, c10::nullopt);
  if (!torch::allclose(out_torch, out_triton, 1e-3, 1e-3)) {
    LOG(INFO) << "Difference:\n" << out_torch - out_triton;
  }

  EXPECT_TRUE(torch::allclose(out_torch, out_triton, 1e-3, 1e-3));
}

TEST(reduction_op_test, sum_dim_inner) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor a = torch::randn({32, 1024}, device);

  torch::Tensor out_torch = at::sum(a, {1});
  torch::Tensor out_triton = flag_gems::sum_dim(a, {1});
  if (!torch::allclose(out_torch, out_triton, 1e-3, 1e-3)) {
    LOG(INFO) << "Difference:\n" << out_torch - out_triton;
  }

  EXPECT_TRUE(torch::allclose(out_torch, out_triton, 1e-3, 1e-3));
}

TEST(reduction_op_test, sum_dim_non_inner) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor a = torch::randn({32, 1024, 32}, device);

  torch::Tensor out_torch = at::sum(a, {1});
  torch::Tensor out_triton = flag_gems::sum_dim(a, {1});
  if (!torch::allclose(out_torch, out_triton, 1e-3, 1e-3)) {
    LOG(INFO) << "Difference:\n" << out_torch - out_triton;
  }

  EXPECT_TRUE(torch::allclose(out_torch, out_triton, 1e-3, 1e-3));
}

TEST(reduction_op_test, sum_dim_multi) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor a = torch::randn({32, 1024, 32}, device);

  torch::Tensor out_torch = at::sum(a, {2, 0});
  torch::Tensor out_triton = flag_gems::sum_dim(a, {2, 0});
  if (!torch::allclose(out_torch, out_triton, 1e-3, 1e-3)) {
    LOG(INFO) << "Difference:\n" << out_torch - out_triton;
  }
  EXPECT_TRUE(torch::allclose(out_torch, out_triton, 1e-3, 1e-3));
}

TEST(reduction_op_test, nonzero) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor a = torch::randn({32, 1024}, device);
  a = a > 0.5;

  torch::Tensor out_torch = at::nonzero(a);
  torch::Tensor out_triton = flag_gems::nonzero(a);
  if (!torch::allclose(out_torch, out_triton, 1e-5, 1e-8)) {
    LOG(INFO) << "Difference:\n" << out_torch - out_triton;
  }

  EXPECT_TRUE(torch::allclose(out_torch, out_triton, 1e-5, 1e-8));
}

struct MaxDimTestParam {
  int64_t m;
  int64_t n;
  bool keepdim;
  int64_t dim_to_keep;
  at::ScalarType dtype;
};

class MaxDimTest : public ::testing::TestWithParam<MaxDimTestParam> {};

TEST_P(MaxDimTest, max_dim) {
  const MaxDimTestParam param = GetParam();
  const torch::Device device(torch::kCUDA, 0);
  const at::TensorOptions opt = at::TensorOptions().device(device).dtype(param.dtype);
  torch::Tensor input = torch::randn({param.m, param.n}, opt);
  auto out_torch = at::max(input, param.dim_to_keep, param.keepdim);
  torch::Tensor max_torch = std::get<0>(out_torch);
  torch::Tensor index_torch = std::get<1>(out_torch);
  auto out_triton = flag_gems::max_dim(input, param.dim_to_keep, param.keepdim);
  torch::Tensor max_triton = std::get<0>(out_triton);
  torch::Tensor index_triton = std::get<1>(out_triton);
  if (!torch::allclose(max_torch, max_triton, 1e-5, 1e-8)) {
    LOG(INFO) << "Max value difference (keepdim=" << param.keepdim << "):\n" << max_torch - max_triton;
  }
  EXPECT_TRUE(torch::allclose(max_torch, max_triton, 1e-5, 1e-8));
  if (!torch::allclose(index_torch, index_triton, 1e-5, 1e-8)) {
    LOG(INFO) << "Index difference (keepdim=" << param.keepdim << "):\n" << index_torch - index_triton;
  }
  EXPECT_TRUE(torch::allclose(index_torch, index_triton, 0, 0));
}

INSTANTIATE_TEST_SUITE_P(MaxDimTests,
                         MaxDimTest,
                         ::testing::Values(MaxDimTestParam {32, 1024, true, 0, at::ScalarType::Float},
                                           MaxDimTestParam {32, 1024, true, 1, at::ScalarType::Float},
                                           MaxDimTestParam {32, 1024, false, 0, at::ScalarType::Float},
                                           MaxDimTestParam {32, 1024, false, 1, at::ScalarType::Float},
                                           MaxDimTestParam {32, 1024, true, 0, at::ScalarType::Half},
                                           MaxDimTestParam {32, 1024, true, 1, at::ScalarType::Half},
                                           MaxDimTestParam {32, 1024, false, 0, at::ScalarType::Half},
                                           MaxDimTestParam {32, 1024, false, 1, at::ScalarType::Half},
                                           MaxDimTestParam {32, 1024, true, 0, at::ScalarType::BFloat16},
                                           MaxDimTestParam {32, 1024, true, 1, at::ScalarType::BFloat16},
                                           MaxDimTestParam {32, 1024, false, 0, at::ScalarType::BFloat16},
                                           MaxDimTestParam {32, 1024, false, 1, at::ScalarType::BFloat16}));

TEST(MaxTest, max) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor input = torch::randn({32, 1024}, device);
  auto out_torch = at::max(input);
  auto out_triton = flag_gems::max(input);
  EXPECT_TRUE(out_triton.sizes() == torch::IntArrayRef {}) << "out_triton is not a scalar tensor";
  if (!torch::allclose(out_torch, out_triton, 1e-5, 1e-8)) {
    LOG(INFO) << "Max value differenc:\n" << out_torch - out_triton;
  }
  EXPECT_TRUE(torch::allclose(out_torch, out_triton, 1e-5, 1e-8));
}

class MaxDimMaxTest : public ::testing::TestWithParam<MaxDimTestParam> {};

TEST_P(MaxDimMaxTest, max_dim_max) {
  const MaxDimTestParam param = GetParam();
  const torch::Device device(torch::kCUDA, 0);
  const at::TensorOptions opt = at::TensorOptions().device(device).dtype(param.dtype);
  torch::Tensor input = torch::randn({param.m, param.n}, opt);
  auto out_torch = at::max(input, param.dim_to_keep, param.keepdim);
  torch::Tensor max_torch = std::get<0>(out_torch);
  torch::Tensor index_torch = std::get<1>(out_torch);
  torch::Tensor out_value = torch::empty_like(max_torch);
  torch::Tensor out_index = torch::empty_like(index_torch);
  auto out_triton = flag_gems::max_dim_max(input, param.dim_to_keep, param.keepdim, out_value, out_index);
  torch::Tensor max_triton = std::get<0>(out_triton);
  torch::Tensor index_triton = std::get<1>(out_triton);
  if (!torch::allclose(max_torch, max_triton, 1e-5, 1e-8)) {
    LOG(INFO) << "Max value difference (keepdim=" << param.keepdim << "):\n" << max_torch - max_triton;
  }
  EXPECT_TRUE(torch::allclose(max_torch, max_triton, 1e-5, 1e-8));
  if (!torch::allclose(index_torch, index_triton, 1e-5, 1e-8)) {
    LOG(INFO) << "Index difference (keepdim=" << param.keepdim << "):\n" << index_torch - index_triton;
  }
  EXPECT_TRUE(torch::allclose(index_torch, index_triton, 0, 0));
}

INSTANTIATE_TEST_SUITE_P(MaxDimMaxTests,
                         MaxDimMaxTest,
                         ::testing::Values(MaxDimTestParam {32, 1024, true, 0, at::ScalarType::Float},
                                           MaxDimTestParam {32, 1024, true, 1, at::ScalarType::Float},
                                           MaxDimTestParam {32, 1024, false, 0, at::ScalarType::Float},
                                           MaxDimTestParam {32, 1024, false, 1, at::ScalarType::Float},
                                           MaxDimTestParam {32, 1024, true, 0, at::ScalarType::Half},
                                           MaxDimTestParam {32, 1024, true, 1, at::ScalarType::Half},
                                           MaxDimTestParam {32, 1024, false, 0, at::ScalarType::Half},
                                           MaxDimTestParam {32, 1024, false, 1, at::ScalarType::Half},
                                           MaxDimTestParam {32, 1024, true, 0, at::ScalarType::BFloat16},
                                           MaxDimTestParam {32, 1024, true, 1, at::ScalarType::BFloat16},
                                           MaxDimTestParam {32, 1024, false, 0, at::ScalarType::BFloat16},
                                           MaxDimTestParam {32, 1024, false, 1, at::ScalarType::BFloat16}));

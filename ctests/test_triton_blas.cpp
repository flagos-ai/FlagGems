#include <gtest/gtest.h>
#include <map>
#include "flag_gems/operators.h"
#include "torch/torch.h"

std::map<at::ScalarType, double> RESOLUTION = {
    {   at::ScalarType::Float, 1.3e-6},
    {    at::ScalarType::Half,   1e-3},
    {at::ScalarType::BFloat16,  0.016},
};

TEST(blas_op_test, mm) {
  torch::manual_seed(0);
  at::ScalarType dtype = at::ScalarType::Float;
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor a = torch::randn({10, 10}, device).to(dtype);
  torch::Tensor b = torch::randn({10, 10}, device).to(dtype);

  torch::Tensor out_torch = at::mm(a.to(at::ScalarType::Double), b.to(at::ScalarType::Double));
  torch::Tensor out_triton = flag_gems::mm_tensor(a, b);

  EXPECT_TRUE(torch::allclose(out_torch.to(dtype), out_triton, /*rtol=*/RESOLUTION[dtype], /*atol=*/1e-4));
}

// struct BmmTestParam {
//   int64_t m;
//   int64_t n;
//   int64_t k;
//   at::ScalarType dtype;
// };

// class BmmTest : public ::testing::TestWithParam<BmmTestParam> {};

// TEST_P(BmmTest, addmm) {
//   torch::manual_seed(0);
//   const BmmTestParam param = GetParam();
//   const torch::Device device(torch::kCUDA, 0);
//   const at::TensorOptions opt = at::TensorOptions().device(device).dtype(param.dtype);
//   const at::Tensor bias = at::randn({param.m, param.n}, opt);
//   const at::Tensor mat1 = at::randn({param.m, param.k}, opt);
//   const at::Tensor mat2 = at::randn({param.k, param.n}, opt);

//   at::Tensor out_torch = at::addmm(bias, mat1, mat2);
//   at::Tensor out_triton = flag_gems::addmm(bias, mat1, mat2);

//   EXPECT_TRUE(torch::allclose(out_torch, out_triton));
// }

// INSTANTIATE_TEST_SUITE_P(BmmTests,
//                          BmmTest,
//                          ::testing::Values(BmmTestParam {10, 10, 10, at::ScalarType::Float},
//                                            BmmTestParam {10, 10, 10, at::ScalarType::Half},
//                                            BmmTestParam {10, 10, 10, at::ScalarType::BFloat16}));

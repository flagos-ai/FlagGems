#include <gtest/gtest.h>
#include "flag_gems/operators.h"
#include "torch/torch.h"

TEST(reduction_op_test, sum) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor a = torch::randn({32, 4096}, device);

  torch::Tensor out_torch = at::sum(a, {1});
  torch::Tensor out_triton = flag_gems::sum_dim(a, {1});

  if (!torch::allclose(out_torch, out_triton, 1e-5, 1e-8)) {
    std::cout << "Difference:\n" << out_torch - out_triton << std::endl;
  }

  EXPECT_TRUE(torch::allclose(out_torch, out_triton, 1e-5, 1e-8));
}

#include <gtest/gtest.h>
#include "c10/util/Logging.h"
#include "flag_gems/operators.h"
#include "torch/torch.h"

TEST(reduction_op_test, sum) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor a = torch::randn({32, 1024}, device);

  torch::Tensor out_torch = at::sum(a, {1});
  torch::Tensor out_triton = flag_gems::sum_dim(a, {1});

  if (!torch::allclose(out_torch, out_triton, 1e-5, 1e-8)) {
    LOG(INFO) << "Difference:\n" << out_torch - out_triton;
  }

  EXPECT_TRUE(torch::allclose(out_torch, out_triton, 1e-5, 1e-8));
}

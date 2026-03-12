#include <gtest/gtest.h>
#include "flag_gems/accuracy_utils.h"
#include "flag_gems/operators.h"
#include "torch/torch.h"

TEST(pointwise_op_test, add) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor a = torch::randn({10, 10}, device);
  torch::Tensor b = torch::randn({10, 10}, device);

  torch::Tensor out_torch = a + b;
  torch::Tensor out_triton = flag_gems::add_tensor(a, b);

  auto result = flag_gems::accuracy_utils::gems_assert_close(out_triton, out_torch);
  EXPECT_TRUE(result.ok) << result.message;
}

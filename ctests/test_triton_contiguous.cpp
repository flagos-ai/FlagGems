#include <gtest/gtest.h>
#include "flag_gems/operators.h"
#include "torch/torch.h"

TEST(contiguous_op_test, contiguous) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor inp = torch::randn({10, 10, 10}, device);
  inp = inp.index({torch::indexing::Slice(torch::indexing::None, torch::indexing::None, 2)});
  EXPECT_FALSE(inp.is_contiguous());
  torch::Tensor ref_out = inp.contiguous();
  torch::Tensor res_out = flag_gems::contiguous(inp);
  EXPECT_TRUE(res_out.is_contiguous());
  EXPECT_EQ(res_out.strides(), ref_out.strides());
  EXPECT_TRUE(torch::allclose(ref_out, res_out));
}

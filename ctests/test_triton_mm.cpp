#include <gtest/gtest.h>
#include "flag_gems/accuracy_utils.h"
#include "flag_gems/operators.h"
#include "torch/torch.h"

TEST(MmTest, mm) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor a = torch::randn({10, 10}, device);
  torch::Tensor b = torch::randn({10, 10}, device);
  torch::Tensor ref_a = flag_gems::accuracy_utils::to_reference(a, /*upcast=*/false);
  torch::Tensor ref_b = flag_gems::accuracy_utils::to_reference(b, /*upcast=*/false);

  torch::Tensor out_torch = at::mm(ref_a, ref_b);
  torch::Tensor out_triton = flag_gems::mm_tensor(a, b);

  auto result = flag_gems::accuracy_utils::gems_assert_close(out_triton, out_torch, a.scalar_type());
  EXPECT_TRUE(result.ok) << result.message;
}

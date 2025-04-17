#include "flag_gems/operators.h"

#include "torch/torch.h"

int main() {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor a = torch::randn({4096, 4096}, device);

  torch::Tensor tmp1 = at::sum(a, {1});
  torch::Tensor tmp2 = flag_gems::sum_dim(a, {1});
  std::cout << "ATEN:\n" << tmp1 << std::endl;
  std::cout << "TRITON:\n" << tmp2 << std::endl;

  for (int i = 0; i < 10; i++) {
    torch::Tensor out1 = at::sum(a, {1});
  }

  for (int i = 0; i < 10; i++) {
    torch::Tensor out2 = flag_gems::sum_dim(a, {1});
  }
  return 0;
}

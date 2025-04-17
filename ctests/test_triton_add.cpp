#include "flag_gems/operators.h"
#include "torch/torch.h"

int main() {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor a = torch::randn({10, 10}, device);
  torch::Tensor b = torch::randn({10, 10}, device);
  torch::Tensor tmp1 = a + b;
  torch::Tensor tmp2 = flag_gems::add_tensor(a, b);
  std::cout << "ATEN:\n" << tmp1 << std::endl;
  std::cout << "TRITON:\n" << tmp2 << std::endl;

  for (int i = 0; i < 10; i++) {
    torch::Tensor out1 = a + b;
  }

  for (int i = 0; i < 10; i++) {
    torch::Tensor out2 = flag_gems::add_tensor(a, b);
  }
  return 0;
}

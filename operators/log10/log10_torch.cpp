#include <torch/extension.h>
#include <torch/torch.h>
#include <cmath>

// 前向传播
torch::Tensor log10_forward(const torch::Tensor& input) {
    TORCH_CHECK(input.dim() >= 1, "log10: input tensor dimension must be >= 1");
    TORCH_CHECK(input.is_floating_point(), "log10: only float type supported");

    auto eps = 1e-15;
    auto safe_input = torch::clamp(input, eps, 1e15);
    return torch::log10(safe_input);
}

// 反向传播（带校验）
torch::Tensor log10_backward(const torch::Tensor& grad_output, const torch::Tensor& input) {
    TORCH_CHECK(input.dim() >= 1, "log10: input tensor dimension must be >= 1");
    TORCH_CHECK(input.is_floating_point(), "log10: only float type supported");
    TORCH_CHECK(grad_output.sizes() == input.sizes(), "log10: grad_output shape must match input");

    auto eps = 1e-15;
    auto safe_input = torch::clamp(input, eps, 1e15);
    return grad_output / (safe_input * std::log(10.0));
}

PYBIND11_MODULE(log10_torch, m) {
    m.def("log10_forward", &log10_forward, "log10 forward", py::arg("input"));
    m.def("log10_backward", &log10_backward, "log10 backward", py::arg("grad_output"), py::arg("input"));
}

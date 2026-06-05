import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic


@pointwise_dynamic(is_tensor=[True, True, False, False])
@triton.jit
def softplus_backward_kernel(grad_output, x, beta, threshold):
    # softplus(x) = log(1 + exp(x)) for x <= threshold, else x
    # d/dx softplus(x) = sigmoid(x) = exp(x) / (1 + exp(x)) = 1 / (1 + exp(-x))
    # When x * beta > threshold, softplus(x) ≈ x, so derivative = 1
    # When x * beta <= threshold, derivative = sigmoid(x * beta)
    z = x.to(tl.float32) * beta
    # sigmoid(z) = 1 / (1 + exp(-z))
    sigmoid_z = 1.0 / (1.0 + tl.exp(-z))
    # Use 1 when z > threshold, else sigmoid(z)
    grad_input = tl.where(z > threshold, grad_output, grad_output * sigmoid_z)
    return grad_input


def softplus_backward(
    grad_output: torch.Tensor, input: torch.Tensor, beta: float, threshold: float
):
    return softplus_backward_kernel(grad_output, input, beta, threshold)

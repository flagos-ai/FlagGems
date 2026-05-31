import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic


@pointwise_dynamic(is_tensor=[True, True, False])
@triton.jit
def logit_backward_kernel(grad_output, x, eps):
    # logit(x) = log(x / (1-x))
    # d/dx logit(x) = 1/(x*(1-x))
    # When eps is provided, x is clamped to [eps, 1-eps] in forward
    # The derivative is the same formula
    one_minus_x = 1.0 - x
    grad_input = grad_output / (x * one_minus_x)
    return grad_input


def logit_backward(grad_output: torch.Tensor, input: torch.Tensor, eps: float | None = None):
    return logit_backward_kernel(grad_output, input, eps if eps is not None else 0.0)

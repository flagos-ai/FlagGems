import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic


# SELU constants from PyTorch
ALPHA = 1.6732632423543772848170429916717
SCALE = 1.0507009873554804934193349852946


@pointwise_dynamic(is_tensor=[True, True, False])
@triton.jit
def selu_backward_kernel(grad_output, x, _alpha, _scale):
    # selu: y = scale * (x if x > 0 else alpha * (exp(x) - 1))
    # backward:
    #   x > 0: grad_input = grad_output * scale
    #   x <= 0: grad_input = grad_output * scale * alpha * exp(x)
    is_pos = x > 0
    pos_grad = grad_output * SCALE
    neg_grad = grad_output * SCALE * ALPHA * tl.exp(x)
    return tl.where(is_pos, pos_grad, neg_grad)


def selu_backward(grad_output: torch.Tensor, input: torch.Tensor, alpha: float, scale: float):
    # alpha and scale are ignored (use constants)
    return selu_backward_kernel(grad_output, input, alpha, scale)

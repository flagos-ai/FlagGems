import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic


@pointwise_dynamic(is_tensor=[True, True, False])
@triton.jit
def hardshrink_backward_kernel(grad_output, x, lambd):
    mask = tl.abs(x) > lambd
    return tl.where(mask, grad_output, 0.0)


def hardshrink_backward(grad_output: torch.Tensor, input: torch.Tensor, lambd: float):
    return hardshrink_backward_kernel(grad_output, input, lambd)

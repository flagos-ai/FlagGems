import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic


@pointwise_dynamic(is_tensor=[True, True, True])
@triton.jit
def minimum_backward_kernel(grad_output, x, y):
    # minimum(x, y) = x if x < y else y
    # backward for x: grad_x = grad_output if x < y else 0
    mask = x < y
    return tl.where(mask, grad_output, 0.0)


def minimum_backward(grad_output: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
    """Backward for minimum - returns gradient for x."""
    return minimum_backward_kernel(grad_output, x, y)

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic


@pointwise_dynamic(is_tensor=[True, True, True])
@triton.jit
def maximum_backward_kernel(grad_output, x, y):
    # maximum(x, y) = x if x > y else y
    # backward for x: grad_x = grad_output if x > y else 0
    mask = x > y
    return tl.where(mask, grad_output, 0.0)


def maximum_backward(grad_output: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
    """Backward for maximum - returns gradient for x."""
    return maximum_backward_kernel(grad_output, x, y)

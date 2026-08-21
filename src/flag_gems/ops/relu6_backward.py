import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic


@pointwise_dynamic(is_tensor=[True, True, False])
@triton.jit
def relu6_backward_kernel(grad_output, x, _inplace):
    # relu6(x) = min(max(x, 0), 6)
    # backward: grad_input = grad_output if 0 < x < 6, else 0
    mask = (x > 0) & (x < 6)
    return tl.where(mask, grad_output, 0.0)


def relu6_backward(
    grad_output: torch.Tensor, input: torch.Tensor, inplace: bool = False
):
    return relu6_backward_kernel(grad_output, input, inplace)

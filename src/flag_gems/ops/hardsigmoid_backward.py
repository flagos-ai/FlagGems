import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic


@pointwise_dynamic(is_tensor=[True, True, False])
@triton.jit
def hardsigmoid_backward_kernel(grad_output, x, _inplace):
    # hardsigmoid: y = clamp(x/6 + 0.5, 0, 1)
    # backward: grad_input = grad_output * (1/6) when 0 < x < 6, else 0
    # x/6 + 0.5 > 0  => x > -3
    # x/6 + 0.5 < 1  => x < 3
    # So gradient is (1/6) when -3 < x < 3, else 0
    mask = (x > -3.0) & (x < 3.0)
    return tl.where(mask, grad_output * (1.0 / 6.0), 0.0)


def hardsigmoid_backward(
    grad_output: torch.Tensor, input: torch.Tensor, inplace: bool = False
):
    return hardsigmoid_backward_kernel(grad_output, input, inplace)

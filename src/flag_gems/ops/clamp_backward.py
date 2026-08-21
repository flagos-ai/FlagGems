import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic


@pointwise_dynamic(is_tensor=[True, True, True, True])
@triton.jit
def clamp_tensor_backward_kernel(grad_output, x, min_val, max_val):
    # clamp(x, min, max): grad_input = grad_output if min < x < max, else 0
    mask = (x > min_val) & (x < max_val)
    return tl.where(mask, grad_output, 0.0)


@pointwise_dynamic(is_tensor=[True, True, False, True])
@triton.jit
def clamp_max_tensor_backward_kernel(grad_output, x, _min_none, max_val):
    # clamp(x, None, max): grad_input = grad_output if x < max, else 0
    mask = x < max_val
    return tl.where(mask, grad_output, 0.0)


@pointwise_dynamic(is_tensor=[True, True, True, False])
@triton.jit
def clamp_min_tensor_backward_kernel(grad_output, x, min_val, _max_none):
    # clamp(x, min, None): grad_input = grad_output if x > min, else 0
    mask = x > min_val
    return tl.where(mask, grad_output, 0.0)


def clamp_tensor_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    min_val: torch.Tensor,
    max_val: torch.Tensor,
):
    return clamp_tensor_backward_kernel(grad_output, input, min_val, max_val)


def clamp_min_tensor_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    min_val: torch.Tensor,
    max_val: torch.Tensor,
):
    return clamp_min_tensor_backward_kernel(grad_output, input, min_val, max_val)


def clamp_max_tensor_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    min_val: torch.Tensor,
    max_val: torch.Tensor,
):
    return clamp_max_tensor_backward_kernel(grad_output, input, min_val, max_val)

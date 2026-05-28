import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic


@pointwise_dynamic(is_tensor=[True, True, True, True])
@triton.jit
def where_self_backward_kernel(grad_output, condition, self, other):
    # where(cond, self, other) = self if cond else other
    # backward for self: grad = grad_output if cond else 0
    return tl.where(condition, grad_output, 0.0)


def where_self_backward(grad_output: torch.Tensor, condition: torch.Tensor, self: torch.Tensor, other: torch.Tensor):
    """Backward for where - returns gradient for self."""
    return where_self_backward_kernel(grad_output, condition, self, other)


@pointwise_dynamic(is_tensor=[True, True, True, True])
@triton.jit
def where_other_backward_kernel(grad_output, condition, self, other):
    # where(cond, self, other) = self if cond else other
    # backward for other: grad = 0 if cond else grad_output
    return tl.where(condition, 0.0, grad_output)


def where_other_backward(grad_output: torch.Tensor, condition: torch.Tensor, self: torch.Tensor, other: torch.Tensor):
    """Backward for where - returns gradient for other."""
    return where_other_backward_kernel(grad_output, condition, self, other)

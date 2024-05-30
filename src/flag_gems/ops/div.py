import torch
import triton
import triton.language as tl
import logging
from ..utils import pointwise_dynamic


@pointwise_dynamic
@triton.jit
def div_func(x, y):
    return x / y


@pointwise_dynamic(is_tensor=[True, False])
@triton.jit
def div_func_tensor_scalar(x, y):
    return x / y


@pointwise_dynamic(is_tensor=[False, True])
@triton.jit
def div_func_scalar_tensor(x, y):
    return x / y


def div(A, B):
    logging.debug("GEMS DIV")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        O = div_func(A, B)
        return O
    elif isinstance(A, torch.Tensor):
        O = div_func_tensor_scalar(A, B)
        return O
    elif isinstance(B, torch.Tensor):
        O = div_func_scalar_tensor(A, B)
        return O
    else:
        # Both scalar
        return A / B

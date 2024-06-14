import logging

import torch
import triton

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
        return div_func(A, B)
    elif isinstance(A, torch.Tensor):
        return div_func_tensor_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        return div_func_scalar_tensor(A, B)
    else:
        # Both scalar
        return A / B

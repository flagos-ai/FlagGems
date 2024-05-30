import torch
import triton
import triton.language as tl
import logging
from ..utils import pointwise_dynamic


@pointwise_dynamic(is_tensor=[True, True, False])
@triton.jit
def add_func(x, y, alpha):
    return x + y * alpha


@pointwise_dynamic(is_tensor=[True, False, False])
@triton.jit
def add_func_tensor_scalar(x, y, alpha):
    return x + y * alpha


@pointwise_dynamic(is_tensor=[False, True, False])
@triton.jit
def add_func_scalar_tensor(x, y, alpha):
    return x + y * alpha


def add(A, B, *, alpha=1):
    logging.debug("GEMS ADD")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        O = add_func(A, B, alpha)
        return O
    elif isinstance(A, torch.Tensor):
        O = add_func_tensor_scalar(A, B, alpha)
        return O
    elif isinstance(B, torch.Tensor):
        O = add_func_scalar_tensor(A, B, alpha)
        return O
    else:
        return A + B * alpha

import torch
import triton
import triton.language as tl
import logging
from ..utils import pointwise_dynamic


@pointwise_dynamic(is_tensor=[True, True, False])
@triton.jit
def sub_func(x, y, alpha):
    return x - y * alpha


@pointwise_dynamic(is_tensor=[True, False, False])
@triton.jit
def sub_func_tensor_scalar(x, y, alpha):
    return x - y * alpha


@pointwise_dynamic(is_tensor=[False, True, False])
@triton.jit
def sub_func_scalar_tensor(x, y, alpha):
    return x - y * alpha


def sub(A, B, *, alpha=1):
    logging.debug("GEMS SUB")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        O = sub_func(A, B, alpha)
        return O
    elif isinstance(A, torch.Tensor):
        O = sub_func_tensor_scalar(A, B, alpha)
        return O
    elif isinstance(B, torch.Tensor):
        O = sub_func_scalar_tensor(A, B, alpha)
        return O
    else:
        # Both scalar
        return A - B * alpha

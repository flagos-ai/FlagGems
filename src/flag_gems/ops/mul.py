import torch
import triton
import triton.language as tl
import logging
from ..utils import pointwise_dynamic


@pointwise_dynamic
@triton.jit
def mul_func(x, y):
    return x * y


@pointwise_dynamic(is_tensor=[True, False])
@triton.jit
def mul_func_scalar(x, y):
    return x * y


def mul(A, B):
    logging.debug("GEMS MUL")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        O = mul_func(A, B)
        return O
    elif isinstance(A, torch.Tensor):
        O = mul_func_scalar(A, B)
        return O
    elif isinstance(B, torch.Tensor):
        O = mul_func_scalar(B, A)
        return O
    else:
        # Both scalar
        return A * B

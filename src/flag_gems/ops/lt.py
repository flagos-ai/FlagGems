import torch
import triton
import triton.language as tl
import logging
from ..utils import pointwise_dynamic


@pointwise_dynamic(output_dtypes=[torch.bool])
@triton.jit
def lt_func(x, y):
    return x.to(tl.float32) < y


def lt(A, B):
    logging.debug("GEMS LT")
    O = lt_func(A, B)
    return O


@pointwise_dynamic(is_tensor=[True, False], output_dtypes=[torch.bool])
@triton.jit
def lt_func_scalar(x, y):
    return x.to(tl.float32) < y


def lt_scalar(A, B):
    logging.debug("GEMS LT SCALAR")
    O = lt_func_scalar(A, B)
    return O

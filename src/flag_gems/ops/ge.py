import torch
import triton
import triton.language as tl
import logging
from ..utils import pointwise_dynamic


@pointwise_dynamic(output_dtypes=[torch.bool])
@triton.jit
def ge_func(x, y):
    return x.to(tl.float32) >= y


def ge(A, B):
    logging.debug("GEMS GE")
    O = ge_func(A, B)
    return O


@pointwise_dynamic(is_tensor=[True, False], output_dtypes=[torch.bool])
@triton.jit
def ge_func_scalar(x, y):
    return x.to(tl.float32) >= y


def ge_scalar(A, B):
    logging.debug("GEMS GE SCALAR")
    O = ge_func_scalar(A, B)
    return O

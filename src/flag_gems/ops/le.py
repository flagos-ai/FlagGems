import torch
import triton
import triton.language as tl
import logging
from ..utils import pointwise_dynamic


@pointwise_dynamic(output_dtypes=[torch.bool])
@triton.jit
def le_func(x, y):
    return x.to(tl.float32) <= y


def le(A, B):
    logging.debug("GEMS LE")
    O = le_func(A, B)
    return O


@pointwise_dynamic(is_tensor=[True, False], output_dtypes=[torch.bool])
@triton.jit
def le_func_scalar(x, y):
    return x.to(tl.float32) <= y


def le_scalar(A, B):
    logging.debug("GEMS LE SCALAR")
    O = le_func_scalar(A, B)
    return O

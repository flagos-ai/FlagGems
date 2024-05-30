import triton
import logging
from ..utils import pointwise_dynamic


@pointwise_dynamic
@triton.jit
def bitwise_and_func(x, y):
    return x & y


def bitwise_and_tensor(A, B):
    logging.debug("GEMS BITWISE AND")
    O = bitwise_and_func(A, B)
    return O


@pointwise_dynamic(is_tensor=[True, False])
@triton.jit
def bitwise_and_func_scalar(x, y):
    return x & y


def bitwise_and_scalar(A, B):
    logging.debug("GEMS BITWISE AND SCALAR")
    O = bitwise_and_func_scalar(A, B)
    return O


def bitwise_and_scalar_tensor(A, B):
    logging.debug("GEMS BITWISE AND SCALAR TENSOR")
    O = bitwise_and_func_scalar(B, A)
    return O

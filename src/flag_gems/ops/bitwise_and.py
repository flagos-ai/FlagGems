import logging

import triton

from ..utils import pointwise_dynamic


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_and_func(x, y):
    return x & y


def bitwise_and_tensor(A, B):
    logging.debug("GEMS BITWISE AND")
    return bitwise_and_func(A, B)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_and_func_scalar(x, y):
    return x & y


def bitwise_and_scalar(A, B):
    logging.debug("GEMS BITWISE AND SCALAR")
    return bitwise_and_func_scalar(A, B)


def bitwise_and_scalar_tensor(A, B):
    logging.debug("GEMS BITWISE AND SCALAR TENSOR")
    return bitwise_and_func_scalar(B, A)

import logging

import triton
import triton.language as tl

from ..utils import pointwise_dynamic


@pointwise_dynamic(promotion_methods=[[0, 1, "BOOL_TO_LONG"]])
@triton.jit
def pow_func(x, exponent):
    return tl.math.pow(x.to(tl.float32), exponent)


def pow_tensor_tensor(A, exponent):
    logging.debug("GEMS POW_TENSOR_TENSOR")
    return pow_func(A, exponent)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[[0, 1, "BOOL_TO_LONG"]])
@triton.jit
def pow_func_tensor_scalar(x, exponent):
    return tl.math.pow(x.to(tl.float32), exponent)


def pow_tensor_scalar(A, exponent):
    logging.debug("GEMS POW_TENSOR_SCALAR")
    return pow_func_tensor_scalar(A, exponent)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[[0, 1, "BOOL_TO_LONG"]])
@triton.jit
def pow_func_scalar_tensor(x, exponent):
    return tl.math.pow(x.to(tl.float32), exponent)


def pow_scalar(A, exponent):
    logging.debug("GEMS POW_SCALAR")
    return pow_func_scalar_tensor(A, exponent)

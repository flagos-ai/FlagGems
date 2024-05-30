import triton
import triton.language as tl
import logging
from ..utils import pointwise_dynamic


@pointwise_dynamic
@triton.jit
def pow_func(x, exponent):
    return tl.math.pow(x.to(tl.float32), exponent)


def pow_tensor_tensor(A, exponent):
    logging.debug("GEMS POW_TENSOR_TENSOR")
    O = pow_func(A, exponent)
    return O


@pointwise_dynamic(is_tensor=[True, False])
@triton.jit
def pow_func_tensor_scalar(x, exponent):
    return tl.math.pow(x.to(tl.float32), exponent)


def pow_tensor_scalar(A, exponent):
    logging.debug("GEMS POW_TENSOR_SCALAR")
    O = pow_func_tensor_scalar(A, exponent)
    return O


@pointwise_dynamic(is_tensor=[False, True])
@triton.jit
def pow_func_scalar_tensor(x, exponent):
    return tl.math.pow(x.to(tl.float32), exponent)


def pow_scalar(A, exponent):
    logging.debug("GEMS POW_SCALAR")
    O = pow_func_scalar_tensor(A, exponent)
    return O

import logging

import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))
_pow = tl_extra_shim.pow


@pointwise_dynamic(promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func(x, exponent):
    return _pow(x.to(tl.float32), exponent)


def pow_tensor_tensor(A, exponent):
    logger.debug("GEMS_CAMBRICON POW_TENSOR_TENSOR")
    return pow_func(A, exponent)


def pow_tensor_tensor_(A, exponent):
    logger.debug("GEMS_CAMBRICON POW_TENSOR_TENSOR_")
    return pow_func(A, exponent, out0=A)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func_tensor_scalar(x, exponent):
    return _pow(x.to(tl.float32), exponent)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func_tensor_scalar_int(x, exponent):
    tmp = x.to(dtype=tl.float32)
    result = tl.full(x.shape, 1, tmp.dtype)
    n = tl.abs(exponent)
    if exponent == 0:
        result = result
    elif n == 1:
        result = tmp
    elif n == 2:
        result = tmp * tmp
    elif n == 3:
        result = tmp * tmp
        result = result * tmp
    elif n == 4:
        result = tmp * tmp
        result = result * result
    else:
        while n > 0:
            if n % 2 == 1:
                result = result * tmp
            tmp = tmp * tmp
            n = n // 2
    if exponent < 0:
        result = 1 / result
    return result


def pow_tensor_scalar(A, exponent):
    logger.debug("GEMS_CAMBRICON POW_TENSOR_SCALAR")
    if int(exponent) == exponent:
        return pow_func_tensor_scalar_int(A, exponent)
    return pow_func_tensor_scalar(A, exponent)


def pow_tensor_scalar_(A, exponent):
    logger.debug("GEMS_CAMBRICON POW_TENSOR_SCALAR_")
    return pow_func_tensor_scalar(A, exponent, out0=A)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func_scalar_tensor(x, exponent):
    return _pow(x.to(tl.float32), exponent)


def pow_scalar(A, exponent):
    logger.debug("GEMS_CAMBRICON POW_SCALAR")
    return pow_func_scalar_tensor(A, exponent)

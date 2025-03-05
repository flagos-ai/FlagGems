import logging

import triton
import triton.language as tl

from ..utils import pointwise_dynamic, tl_extra_shim

_pow = tl_extra_shim.pow


@pointwise_dynamic(promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func(x, exponent):
    if x.type.element_ty == tl.bfloat16:
        return _pow(x.to(tl.float32), exponent)
    elif x.type.element_ty == tl.float16:
        return _pow(x.to(tl.float32), exponent)
    else:
        return _pow(x.to(tl.float64), exponent)


def pow_tensor_tensor(A, exponent):
    logging.debug("GEMS POW_TENSOR_TENSOR")
    return pow_func(A, exponent)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func_tensor_scalar(x, exponent):
    if x.type.element_ty == tl.bfloat16:
        return _pow(x.to(tl.float32), exponent)
    elif x.type.element_ty == tl.float16:
        return _pow(x.to(tl.float32), exponent)
    else:
        return _pow(x.to(tl.float64), exponent)


def pow_tensor_scalar(A, exponent):
    logging.debug("GEMS POW_TENSOR_SCALAR")
    return pow_func_tensor_scalar(A, exponent)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func_scalar_tensor(x, exponent):
    if exponent.type.element_ty == tl.bfloat16:
        return _pow(x.to(tl.float32), exponent)
    elif exponent.type.element_ty == tl.float16:
        return _pow(x.to(tl.float32), exponent)
    else:
        return _pow(x.to(tl.float64), exponent)


def pow_scalar(A, exponent):
    logging.debug("GEMS POW_SCALAR")
    return pow_func_scalar_tensor(A, exponent)

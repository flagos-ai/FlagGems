import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def sophgo_pow_func(x, exponent):
    # Implement pow using tl.exp and tl.log: x^y = exp(y * log(x))
    x_f32 = x.to(tl.float32)
    exponent_f32 = exponent.to(tl.float32)

    # For integer exponents, use absolute value to handle negative numbers
    abs_x = tl.abs(x_f32)
    x_safe = tl.where(abs_x > 0.0, abs_x, 1e-10)
    result = tl.exp(exponent_f32 * tl.log(x_safe))

    # If x is 0, result is 0 (for positive exponents)
    is_zero = tl.abs(x_f32) < 1e-10
    result = tl.where(is_zero, 0.0, result)

    return result


def pow_tensor_tensor(A, exponent):
    logger.debug("SOPHGO POW_TENSOR_TENSOR")
    return sophgo_pow_func(A, exponent)


def pow_tensor_tensor_(A, exponent):
    logger.debug("SOPHGO POW_TENSOR_TENSOR_")
    return sophgo_pow_func(A, exponent, out0=A)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def sophgo_pow_func_tensor_scalar(x, exponent):
    # Implement pow using tl.exp and tl.log: x^y = exp(y * log(x))
    x_f32 = x.to(tl.float32)

    # For integer exponents, use absolute value to handle negative numbers
    # Example: (-3)^2 = 9, (-3)^3 = -27
    abs_x = tl.abs(x_f32)
    x_safe = tl.where(abs_x > 0.0, abs_x, 1e-10)
    result = tl.exp(exponent * tl.log(x_safe))

    is_zero = tl.abs(x_f32) < 1e-10

    # If x is 0, result is 0 (for positive exponents)
    result = tl.where(is_zero, 0.0, result)

    return result


def pow_tensor_scalar(A, exponent):
    logger.debug("SOPHGO POW_TENSOR_SCALAR")
    return sophgo_pow_func_tensor_scalar(A, exponent)


def pow_tensor_scalar_(A, exponent):
    logger.debug("SOPHGO POW_TENSOR_SCALAR_")
    return sophgo_pow_func_tensor_scalar(A, exponent, out0=A)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def sophgo_pow_func_scalar_tensor(x, exponent):
    # Implement pow using tl.exp and tl.log: x^y = exp(y * log(x))
    x_f32 = x.to(tl.float32)
    exponent_f32 = exponent.to(tl.float32)

    # For integer exponents, use absolute value to handle negative numbers
    abs_x = tl.abs(x_f32)
    x_safe = tl.where(abs_x > 0.0, abs_x, 1e-10)
    result = tl.exp(exponent_f32 * tl.log(x_safe))

    # If x is 0, result is 0 (for positive exponents)
    is_zero = tl.abs(x_f32) < 1e-10
    result = tl.where(is_zero, 0.0, result)

    return result


def pow_scalar(A, exponent):
    logger.debug("SOPHGO POW_SCALAR")
    return sophgo_pow_func_scalar_tensor(A, exponent)

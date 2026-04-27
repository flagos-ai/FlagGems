import logging

import triton

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_xor_func(x, y):
    return x ^ y


def bitwise_xor_tensor(A, B):
    logger.debug("GEMS BITWISE OR")
    return bitwise_xor_func(A, B)


def bitwise_xor_tensor_(A, B):
    logger.debug("GEMS BITWISE OR_")
    return bitwise_xor_func(A, B, out0=A)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_xor_func_scalar(x, y):
    return x ^ y


def bitwise_xor_scalar(A, B):
    logger.debug("GEMS BITWISE OR SCALAR")
    return bitwise_xor_func_scalar(A, B)


def bitwise_xor_scalar_(A, B):
    logger.debug("GEMS BITWISE OR_ SCALAR")
    return bitwise_xor_func_scalar(A, B, out0=A)


def bitwise_xor_scalar_tensor(A, B):
    logger.debug("GEMS BITWISE OR SCALAR TENSOR")
    return bitwise_xor_func_scalar(B, A)

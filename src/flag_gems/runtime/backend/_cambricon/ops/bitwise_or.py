import logging

import triton

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_or_func(x, y, inplace):
    return x | y


def bitwise_or_tensor(A, B):
    logger.debug("GEMS_CAMBRICON BITWISE OR")
    return bitwise_or_func(A, B, False)


def bitwise_or_tensor_(A, B):
    logger.debug("GEMS_CAMBRICON BITWISE OR_")
    return bitwise_or_func(A, B, True, out0=A)


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def bitwise_or_func_scalar(x, y, inplace):
    return x | y


def bitwise_or_scalar(A, B):
    logger.debug("GEMS_CAMBRICON BITWISE OR SCALAR")
    return bitwise_or_func_scalar(A, B, False)


def bitwise_or_scalar_(A, B):
    logger.debug("GEMS_CAMBRICON OR_ SCALAR")
    return bitwise_or_func_scalar(A, B, True, out0=A)


def bitwise_or_scalar_tensor(A, B):
    logger.debug("GEMS_CAMBRICON BITWISE OR SCALAR TENSOR")
    return bitwise_or_func_scalar(B, A, False)

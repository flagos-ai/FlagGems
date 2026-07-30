import logging

import triton

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def rshift_func(x, y):
    return x >> y


def rshift_tensor(A, B):
    logger.debug("GEMS RSHIFT_TENSOR")
    return rshift_func(A, B)


def rshift_tensor_(A, B):
    logger.debug("GEMS RSHIFT_TENSOR_")
    return rshift_func(A, B, out0=A)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def rshift_func_scalar(x, y):
    return x >> y


def rshift_scalar(A, B):
    logger.debug("GEMS RSHIFT_SCALAR")
    return rshift_func_scalar(A, B)


def rshift_scalar_(A, B):
    logger.debug("GEMS RSHIFT_SCALAR_")
    return rshift_func_scalar(A, B, out0=A)

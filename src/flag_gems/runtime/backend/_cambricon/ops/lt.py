import logging

import triton

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def lt_func(x, y):
    return x < y


def lt(A, B):
    logger.debug("GEMS_CAMBRICON LT")
    return lt_func(A, B)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def lt_func_scalar(x, y):
    return x < y


def lt_scalar(A, B):
    logger.debug("GEMS_CAMBRICON LT SCALAR")
    return lt_func_scalar(A, B)

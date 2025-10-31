import logging

import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def ge_func(x, y):
    return x.to(tl.float32) >= y


def ge(A, B):
    logger.debug("GEMS GE")
    res = ge_func(A, B)
    return res


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def ge_func_scalar(x, y):
    return x.to(tl.float32) >= y


def ge_scalar(A, B):
    logger.debug("GEMS GE SCALAR")
    return ge_func_scalar(A, B)

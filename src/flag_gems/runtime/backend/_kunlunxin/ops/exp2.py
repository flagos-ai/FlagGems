import logging

import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def exp2_func(x):
    LN2 = 0.69314718056
    return tl.exp(x.to(tl.float32) * LN2)


def exp2(A):
    logger.debug("GEMS EXP")
    return exp2_func(A)


def exp2_(A):
    logger.debug("GEMS EXP_")
    return exp2_func(A, out0=A)

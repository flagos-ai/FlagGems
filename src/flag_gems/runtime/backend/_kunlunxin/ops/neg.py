import logging

import triton

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def neg_func(x):
    return -x


def neg(A):
    logger.debug("GEMS NEG")
    return neg_func(A)


def neg_(A):
    logger.debug("GEMS NEG_")
    return neg_func(A, out0=A)

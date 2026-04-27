import logging

import triton

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def square_func(x):
    return x * x


def square(A):
    logger.debug("GEMS SQUARE")
    return square_func(A)


def square_out(A, *, out=None):
    logger.debug("GEMS SQUARE_OUT")
    if out is None:
        return square_func(A)
    square_func(A, out0=out)
    return out


def square_(A):
    logger.debug("GEMS SQUARE_")
    square_func(A, out0=A)
    return A

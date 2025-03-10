import logging

import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic


@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")])
@triton.jit
def abs_func(x):
    return tl.abs(x)


def abs(A):
    logging.debug("GEMS ABS")
    return abs_func(A)


def abs_(A):
    logging.debug("GEMS ABS_")
    abs_func(A, out0=A)
    return A

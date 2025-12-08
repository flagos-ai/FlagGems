import logging

import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "COMPLEX_TO_FLOAT")])
@triton.jit
def abs_func(x, inplace):
    return tl.abs(x)


def abs(A):
    logger.debug("GEMS_CAMBRICON ABS")
    return abs_func(A, False)


def abs_(A):
    logger.debug("GEMS_CAMBRICON ABS_")
    abs_func(A, True, out0=A)
    return A

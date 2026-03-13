import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def asinh_func(x):
    x32 = x.to(tl.float32)
    abs_x = tl.abs(x32)
    result = tl.log(abs_x + tl.sqrt(abs_x * abs_x + 1.0))
    return tl.where(x32 < 0, -result, result)


def asinh(A):
    logger.debug("GEMS ASINH")
    return asinh_func(A)


def asinh_(A):
    logger.debug("GEMS ASINH_")
    asinh_func(A, out0=A)
    return A

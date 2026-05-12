import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def sqrt_func(x):
    return tl.sqrt(x.to(tl.float32))


def sqrt(A):
    logger.debug("GEMS SQRT")
    return sqrt_func(A)


def sqrt_(A):
    logger.debug("GEMS SQRT_")
    sqrt_func(A, out0=A)
    return A

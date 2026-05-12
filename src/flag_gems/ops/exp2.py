import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def exp2_func(x):
    return tl.exp2(x.to(tl.float32))


def exp2(A):
    logger.debug("GEMS EXP")
    return exp2_func(A)


def exp2_(A):
    logger.debug("GEMS EXP_")
    return exp2_func(A, out0=A)

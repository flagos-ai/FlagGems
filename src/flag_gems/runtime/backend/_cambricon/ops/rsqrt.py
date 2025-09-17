import logging

import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def rsqrt_func(x):
    return tl.rsqrt(x.to(tl.float32))


def rsqrt(A):
    logger.debug("GEMS_CAMBRICON RSQRT")
    return rsqrt_func(A)


def rsqrt_(A):
    logger.debug("GEMS_CAMBRICON RSQRT_")
    return rsqrt_func(A, out0=A)

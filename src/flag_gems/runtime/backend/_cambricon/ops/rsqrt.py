import logging

import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def rsqrt_func(x, inplace):
    return tl.rsqrt(x.to(tl.float32))


def rsqrt(A):
    logger.debug("GEMS_CAMBRICON RSQRT")
    return rsqrt_func(A, False)


def rsqrt_(A):
    logger.debug("GEMS_CAMBRICON RSQRT_")
    return rsqrt_func(A, True, out0=A)

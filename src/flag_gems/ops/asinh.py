import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def asinh_func(x):
    x_32 = x.to(tl.float32)
    res = tl.log(x_32 + tl.sqrt(x_32 * x_32 + 1.0))
    return res.to(x.dtype)

def asinh(A):
    logger.debug("GEMS asinh")
    return asinh_func(A)


def asinh_(A):
    logger.debug("GEMS asinh_")
    asinh_func(A, out0=A)
    return A

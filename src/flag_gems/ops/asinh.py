import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def asinh_kernel(x):
    x = x.to(tl.float32)
    return tl.log(x + tl.sqrt(x * x + 1.0))


def asinh(A):
    logger.debug("GEMS ASINH")
    return asinh_kernel(A)


def asinh_(A):
    logger.debug("GEMS ASINH_")
    asinh_kernel(A, out0=A)
    return A


def asinh_out(A, out):
    logger.debug("GEMS ASINH_OUT")
    return asinh_kernel(A, out0=out)

import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def asinh_func(x):
    x_fp32 = x.to(tl.float32)
    return tl.log(x_fp32 + tl.sqrt(x_fp32 * x_fp32 + 1.0))


def asinh(A):
    logger.debug("GEMS ASINH")
    return asinh_func(A)


def asinh_(A):
    logger.debug("GEMS ASINH_")
    asinh_func(A, out0=A)
    return A

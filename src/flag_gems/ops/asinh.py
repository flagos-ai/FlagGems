import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


# asinh(x) = log(x + sqrt(x^2 + 1))
# Uses float32 intermediate for numerical precision.
# INT_TO_FLOAT promotion handles integer input tensors.
@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def asinh_func(x):
    x_fp32 = x.to(tl.float32)
    return tl.log(x_fp32 + tl.sqrt(x_fp32 * x_fp32 + 1.0))


def asinh(A):
    logger.debug("GEMS ASINH")
    return asinh_func(A)


def asinh_out(A, out):
    logger.debug("GEMS ASINH_OUT")
    return asinh_func(A, out0=out)

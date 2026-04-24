import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


# asinh(x) = sign(x) * log(|x| + sqrt(x^2 + 1))
# The sign(x) * log(|x| + ...) form preserves sign on -inf input
# (the naive x + sqrt(x^2+1) form evaluates to -inf + inf = NaN).
# Uses float32 intermediate for numerical precision.
# INT_TO_FLOAT promotion handles integer input tensors.
@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def asinh_func(x):
    x_fp32 = x.to(tl.float32)
    abs_x = tl.abs(x_fp32)
    y = tl.log(abs_x + tl.sqrt(abs_x * abs_x + 1.0))
    return tl.where(x_fp32 < 0.0, -y, y)


def asinh(A):
    logger.debug("GEMS ASINH")
    return asinh_func(A)


def asinh_out(A, out):
    logger.debug("GEMS ASINH_OUT")
    return asinh_func(A, out0=out)

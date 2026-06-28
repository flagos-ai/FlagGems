import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


# arcsinh(x) = sign(x) * log(|x| + sqrt(x^2 + 1))
# The sign(x) * log(|x| + ...) form preserves sign on negative input
# (the naive x + sqrt(x^2+1) form evaluates to -inf + inf = NaN).
@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def arcsinh_func(x):
    x_fp32 = x.to(tl.float32)
    abs_x = tl.abs(x_fp32)
    y = tl.log(abs_x + tl.sqrt(abs_x * abs_x + 1.0))
    return tl.where(x_fp32 < 0.0, -y, y)


def arcsinh(A):
    logger.debug("GEMS ARCSINH")
    return arcsinh_func(A)


def arcsinh_out(A, out):
    logger.debug("GEMS ARCSINH_OUT")
    arcsinh_func(A, out0=out)
    return out

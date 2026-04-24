import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def asinh_func(x):
    # Use sign(x) * log(|x| + sqrt(x^2 + 1)) to avoid cancellation
    # for large negative inputs.
    x_f = x.to(tl.float32)
    abs_x = tl.abs(x_f)
    sign_x = tl.where(x_f < 0, -1.0, 1.0)
    return sign_x * tl.log(abs_x + tl.sqrt(abs_x * abs_x + 1.0))


def asinh(A, *, out=None):
    logger.debug("GEMS ASINH")
    if out is not None:
        return asinh_func(A, out0=out)
    return asinh_func(A)


def asinh_(A):
    logger.debug("GEMS ASINH_")
    asinh_func(A, out0=A)
    return A


def asinh_out(A, *, out=None):
    logger.debug("GEMS ASINH OUT")
    return asinh_func(A, out0=out)

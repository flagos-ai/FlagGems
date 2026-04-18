import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def asinh_func(x):
    # asinh(x) = log(x + sqrt(x^2 + 1))
    # Numerically stable for large |x|: asinh(x) ≈ sign(x)*log(2|x|) when |x| >> 1
    x_f = x.to(tl.float32)
    return tl.log(x_f + tl.sqrt(x_f * x_f + 1.0))


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

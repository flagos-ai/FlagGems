import logging

import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def cosh_func(x):
    x_fp32 = x.to(tl.float32)
    return (tl.exp(x_fp32) + tl.exp(-x_fp32)) * 0.5


def cosh(A):
    logger.debug("GEMS COSH")
    return cosh_func(A)


def cosh_(A):
    logger.debug("GEMS COSH_")
    cosh_func(A, out0=A)
    return A


def cosh_out(A, *, out=None):
    logger.debug("GEMS COSH_OUT")
    if out is None:
        return cosh_func(A)
    cosh_func(A, out0=out)
    return out

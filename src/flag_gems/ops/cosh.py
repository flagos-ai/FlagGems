import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def cosh_func(x):
    x = x.to(tl.float32)
    return 0.5 * (tl.exp(x) + tl.exp(-x))


def cosh(A):
    logger.debug("GEMS COSH")
    return cosh_func(A)


def cosh_(A):
    logger.debug("GEMS COSH_")
    return cosh_func(A, out0=A)


def cosh_out(A, out):
    logger.debug("GEMS COSH_OUT")
    return cosh_func(A, out0=out)

import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def cosh_func(x):
    x_32 = x.to(tl.float32)
    return 0.5 * (tl.exp(x_32) + tl.exp(-x_32))

def cosh(A):
    logger.debug("GEMS cosh")
    return cosh_func(A)


def cosh_(A):
    logger.debug("GEMS cosh_")
    cosh_func(A, out0=A)
    return A

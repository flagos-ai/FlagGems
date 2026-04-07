import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


# cosh(x) = (exp(x) + exp(-x)) / 2
# Uses float32 intermediate computation for numerical accuracy
# with INT_TO_FLOAT promotion for integer input support.
@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def cosh_func(x):
    x_fp32 = x.to(tl.float32)
    return 0.5 * (tl.exp(x_fp32) + tl.exp(-x_fp32))


def cosh(A):
    logger.debug("GEMS COSH")
    return cosh_func(A)


def cosh_(A):
    logger.debug("GEMS COSH_")
    cosh_func(A, out0=A)
    return A

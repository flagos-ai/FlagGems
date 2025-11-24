import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def tan_func(x):
    y = tl_extra_shim.tan(x.to(tl.float32))
    return y


def tan(A):
    logger.debug("GEMS TAN")
    return tan_func(A)


def tan_(A):
    logger.debug("GEMS TAN_")
    tan_func(A, out0=A)
    return A

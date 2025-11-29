import logging

import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def sqrt_func(x, inplace):
    return tl.sqrt(x.to(tl.float32))


def sqrt(A):
    logger.debug("GEMS_CAMBRICON SQRT")
    return sqrt_func(A, False)


def sqrt_(A):
    logger.debug("GEMS_CAMBRICON SQRT_")
    sqrt_func(A, True, out0=A)
    return A

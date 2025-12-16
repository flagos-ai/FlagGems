import logging

import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def sin_func(x, inplace):
    return tl.sin(x.to(tl.float32))


def sin(A):
    logger.debug("GEMS_CAMBRICON SIN")
    return sin_func(A, False)


def sin_(A):
    logger.debug("GEMS_CAMBRICON SIN_")
    sin_func(A, True, out0=A)
    return A

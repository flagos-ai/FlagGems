import logging

import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def exp2_func(x, inplace):
    return tl.exp2(x.to(tl.float32))


def exp2(A):
    logger.debug("GEMS_CAMBRICON EXP2")
    return exp2_func(A, False)


def exp2_(A):
    logger.debug("GEMS_CAMBRICON EXP2_")
    return exp2_func(A, True, out0=A)

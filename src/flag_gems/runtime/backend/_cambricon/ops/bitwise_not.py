import logging

import triton

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def bitwise_not_func(x, inplace):
    return ~x


def bitwise_not(A):
    logger.debug("GEMS_CAMBRICON BITWISE_NOT")
    return bitwise_not_func(A, False)


def bitwise_not_(A):
    logger.debug("GEMS_CAMBRICON BITWISE_NOT_")
    bitwise_not_func(A, True, out0=A)
    return A

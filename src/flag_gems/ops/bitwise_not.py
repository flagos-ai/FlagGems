import logging

import triton

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def bitwise_not_func(x):
    return ~x


def bitwise_not(A):
    logger.debug("GEMS BITWISE NOT")
    return bitwise_not_func(A)


def bitwise_not_(A):
    logger.debug("GEMS BITWISE NOT_")
    bitwise_not_func(A, out0=A)
    return A

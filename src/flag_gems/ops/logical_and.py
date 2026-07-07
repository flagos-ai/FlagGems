import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def logical_and_func(x, y):
    return x.to(tl.int1).logical_and(y.to(tl.int1))


def logical_and(A, B):
    logger.debug("GEMS LOGICAL_AND")
    return logical_and_func(A, B)


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def logical_and_func_(x, y):
    return tl.where((x != 0) & (y != 0), 1, 0)


def logical_and_(A, B):
    logger.debug("GEMS LOGICAL_AND_")
    logical_and_func_(A, B, out0=A)
    return A

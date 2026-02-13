import logging

import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def logical_and_func(x, y):
    return x.to(tl.int1).logical_and(y.to(tl.int1))


def logical_and(A, B):
    logger.debug("GEMS_CAMBRICON LOGICAL_AND")
    return logical_and_func(A, B)


@pointwise_dynamic(
    is_tensor=[True, True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")]
)
@triton.jit
def logical_and_func_(x, y, inplace):
    return tl.where((x != 0) & (y != 0), 1, 0)


def logical_and_(A, B):
    logger.debug("GEMS_CAMBRICON LOGICAL_AND_")
    logical_and_func_(A, B, True, out0=A)
    return A

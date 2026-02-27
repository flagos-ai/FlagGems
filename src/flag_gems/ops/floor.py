import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def floor_func(x):
    return tl.floor(x.to(tl.float32)).to(x.dtype)


def floor(A):
    logger.debug("GEMS FLOOR")
    return floor_func(A)


def floor_out(A, *, out=None):
    logger.debug("GEMS FLOOR_OUT")
    if out is None:
        return floor_func(A)
    floor_func(A, out0=out)
    return out


def floor_(A):
    logger.debug("GEMS FLOOR_")
    floor_func(A, out0=A)
    return A

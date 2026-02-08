import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def ceil_func(x):
    return tl.ceil(x.to(tl.float32)).to(x.dtype)


def ceil(A):
    logger.debug("GEMS CEIL")
    return ceil_func(A)


def ceil_out(A, *, out=None):
    logger.debug("GEMS CEIL_OUT")
    if out is None:
        return ceil_func(A)
    ceil_func(A, out0=out)
    return out


def ceil_(A):
    logger.debug("GEMS CEIL_")
    ceil_func(A, out0=A)
    return A

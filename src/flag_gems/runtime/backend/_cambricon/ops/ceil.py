import logging

import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def ceil_func(x, inplace):
    return tl.ceil(x.to(tl.float32)).to(x.dtype)


def ceil(A):
    logger.debug("GEMS_CAMBRICON CEIL")
    return ceil_func(A, False)


def ceil_out(A, *, out=None):
    logger.debug("GEMS_CAMBRICON CEIL_OUT")
    if out is None:
        return ceil_func(A, False)
    ceil_func(A, False, out0=out)
    return out


def ceil_(A):
    logger.debug("GEMS_CAMBRICON CEIL_")
    ceil_func(A, True, out0=A)
    return A

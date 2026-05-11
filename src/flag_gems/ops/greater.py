import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def greater_func(x, y):
    return x.to(tl.float32) > y


def greater(A, B):
    logger.debug("GEMS GREATER")
    return greater_func(A, B)


def greater_out(A, B, *, out=None):
    logger.debug("GEMS GREATER_OUT")
    if out is None:
        return greater_func(A, B)
    greater_func(A, B, out0=out)
    return out


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def greater_func_scalar(x, y):
    return x.to(tl.float32) > y


def greater_scalar(A, B):
    logger.debug("GEMS GREATER_SCALAR")
    return greater_func_scalar(A, B)


def greater_scalar_out(A, B, *, out=None):
    logger.debug("GEMS GREATER_SCALAR_OUT")
    if out is None:
        return greater_func_scalar(A, B)
    greater_func_scalar(A, B, out0=out)
    return out

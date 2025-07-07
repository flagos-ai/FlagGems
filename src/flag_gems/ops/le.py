import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def le_func(x, y):
    return x.to(tl.float32) <= y


def le(A, B):
    logger.debug("GEMS LE")
    return le_func(A, B)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def le_func_scalar(x, y):
    return x.to(tl.float32) <= y


def le_scalar(A, B):
    logger.debug("GEMS LE SCALAR")
    return le_func_scalar(A, B)

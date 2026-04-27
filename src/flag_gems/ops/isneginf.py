import logging

import triton

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "ALWAYS_BOOL")])
@triton.jit
def isneginf_func(x):
    return x == float("-inf")


def isneginf(A):
    logger.debug("GEMS ISNEGINF")
    return isneginf_func(A)


def isneginf_out(A, *, out=None):
    logger.debug("GEMS ISNEGINF_OUT")
    if out is None:
        return isneginf_func(A)
    isneginf_func(A, out0=out)
    return out

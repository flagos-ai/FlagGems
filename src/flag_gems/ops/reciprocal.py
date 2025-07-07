import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def reciprocal_func(x):
    return 1.0 / x.to(tl.float32)


def reciprocal(A):
    logger.debug("GEMS RECIPROCAL")
    return reciprocal_func(A)


def reciprocal_(A):
    logger.debug("GEMS RECIPROCAL_")
    return reciprocal_func(A, out0=A)

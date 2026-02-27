import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def log1p_func(x):
    return tl.log(1.0 + x.to(tl.float32)).to(x.dtype)


def log1p(A):
    logger.debug("GEMS LOG1P")
    return log1p_func(A)


def log1p_(A):
    logger.debug("GEMS LOG1P_")
    log1p_func(A, out0=A)
    return A

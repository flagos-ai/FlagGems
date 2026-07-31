import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def sign_func(x):
    return tl.where(x > 0, 1.0, tl.where(x < 0, -1.0, 0.0))


def sign(A):
    logger.debug("GEMS SIGN")
    return sign_func(A)


def sign_(A):
    logger.debug("GEMS SIGN_")
    sign_func(A, out0=A)
    return A

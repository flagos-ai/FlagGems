import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def expm1_func(x):
    return (tl.exp(x.to(tl.float32)) - 1.0).to(x.dtype)


def expm1_(A):
    logger.debug("GEMS EXPM1_")
    expm1_func(A, out0=A)
    return A
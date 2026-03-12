import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")])
@triton.jit
def log10_func(x):
    return tl.log2(x.to(tl.float32)) / tl.log2(10.0)


def log10(A):
    logger.debug("GEMS LOG10")
    return log10_func(A)


def log10_(A):
    logger.debug("GEMS LOG10_")
    return log10_func(A, out0=A)
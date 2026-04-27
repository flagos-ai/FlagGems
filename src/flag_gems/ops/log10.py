import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def log10_func(x):
    return tl.log(x.to(tl.float32)) * 0.4342944819032518


def log10(A):
    logger.debug("GEMS LOG10")
    return log10_func(A)


def log10_(A):
    logger.debug("GEMS LOG10_")
    return log10_func(A, out0=A)


def log10_out(A, out):
    logger.debug("GEMS LOG10_OUT")
    return log10_func(A, out0=out)

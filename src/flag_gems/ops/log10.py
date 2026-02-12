import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

# log10(x) = log2(x) / log2(10) = log2(x) * log10(2)
_LOG10_2 = 0.3010299956639812  # math.log10(2)


@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")])
@triton.jit
def log10_func(x):
    return tl.log2(x.to(tl.float32)) * _LOG10_2


def log10(A):
    logger.debug("GEMS LOG10")
    return log10_func(A)


def log10_(A):
    logger.debug("GEMS LOG10_")
    log10_func(A, out0=A)
    return A


def log10_out(A, out):
    logger.debug("GEMS LOG10_OUT")
    log10_func(A, out0=out)
    return out

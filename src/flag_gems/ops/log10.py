import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

# log10(x) = log(x) * log10(e)  where log10(e) = 1/ln(10)
_LOG10E = 0.4342944819032518


@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")])
@triton.jit
def log10_kernel(x):
    return tl.log(x.to(tl.float32)) * _LOG10E


def log10(A):
    logger.debug("GEMS LOG10")
    return log10_kernel(A)


def log10_(A):
    logger.debug("GEMS LOG10_")
    log10_kernel(A, out0=A)
    return A

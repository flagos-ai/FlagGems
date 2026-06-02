import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")])
@triton.jit
def erfc_func(x):
    # erfc(x) = 1 - erf(x)
    return 1.0 - tl.erf(x.to(tl.float32))


def erfc(A):
    logger.debug("GEMS ERFC")
    return erfc_func(A)


def erfc_(A):
    logger.debug("GEMS ERFC_")
    erfc_func(A, out0=A)
    return A

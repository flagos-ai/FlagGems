import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")])
@triton.jit
def absolute_func(x):
    return tl.abs(x)


def absolute(A):
    logger.debug("GEMS ABSOLUTE")
    return absolute_func(A)

import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def xlogy_kernel(x, y):
    # x * log(y), with x=0 -> 0 regardless of y
    zero = x * 0.0  # type-preserving zero
    result = x * tl.log(y)
    return tl.where(x == zero, zero, result)


def xlogy(input, other):
    logger.debug("GEMS XLOGY")
    return xlogy_kernel(input, other)

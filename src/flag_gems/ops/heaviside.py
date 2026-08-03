import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def heaviside_kernel(x, values):
    # x < 0 -> 0, x == 0 -> values, x > 0 -> 1
    zero = x * 0.0
    result = tl.where(x < zero, zero, tl.where(x == zero, values, zero + 1.0))
    return result


def heaviside(input, values):
    logger.debug("GEMS HEAVISIDE")
    return heaviside_kernel(input, values)

import logging

import triton

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_right_shift_kernel(a, b):
    return a >> b


def bitwise_right_shift(self, other, *, out=None):
    logger.debug("GEMS BITWISE_RIGHT_SHIFT")
    return bitwise_right_shift_kernel(self, other, out=out)

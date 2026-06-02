import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def floor_inplace_func(x):
    return tl.floor(x.to(tl.float32)).to(x.dtype)


def floor_(A):
    logger.debug("GEMS FLOOR_")
    floor_inplace_func(A, out0=A)
    return A

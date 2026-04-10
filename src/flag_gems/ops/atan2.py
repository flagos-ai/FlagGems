import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

_atan2 = tl_extra_shim.atan2

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def atan2_kernel(x, y):
    return _atan2(x.to(tl.float32), y.to(tl.float32))


def atan2(input, other):
    logger.debug("GEMS ATAN2")
    return atan2_kernel(input, other)


def atan2_out(input, other, out):
    logger.debug("GEMS ATAN2_OUT")
    return atan2_kernel(input, other, out0=out)

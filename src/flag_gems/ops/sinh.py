import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

_sinh = tl_extra_shim.sinh
logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def sinh_kernel(x):
    return _sinh(x.to(tl.float32))


def sinh(A):
    logger.debug("GEMS SINH")
    return sinh_kernel(A)


def sinh_(A):
    logger.debug("GEMS SINH_")
    sinh_kernel(A, out0=A)
    return A

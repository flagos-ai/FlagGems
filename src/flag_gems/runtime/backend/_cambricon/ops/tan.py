import logging

import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def tan_func(x, inplace):
    y = tl_extra_shim.tan(x.to(tl.float32))
    return y


def tan(A):
    logger.debug("GEMS_CAMBRICON TAN")
    return tan_func(A, False)


def tan_(A):
    logger.debug("GEMS_CAMBRICON TAN_")
    tan_func(A, True, out0=A)
    return A

import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

_atanh = tl_extra_shim.atanh
logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def atanh_kernel(x):
    return _atanh(x.to(tl.float32))


def atanh(x):
    logger.debug("GEMS ATANH")
    return atanh_kernel(x)


def atanh_(x):
    logger.debug("GEMS ATANH_")
    atanh_kernel(x, out0=x)
    return x

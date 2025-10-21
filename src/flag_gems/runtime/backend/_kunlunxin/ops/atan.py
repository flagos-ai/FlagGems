import logging

import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

_atan = tl_extra_shim.atan
logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def atan_kernel(x):
    return _atan(x.to(tl.float32))


def atan(A):
    logger.debug("GEMS ATAN")
    out = atan_kernel(A)
    return out


def atan_(A):
    logger.debug("GEMS ATAN_")
    atan_kernel(A, out0=A)
    return A

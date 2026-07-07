import logging

import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

_atan = tl_extra_shim.atan
logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def atan_kernel(x, inplace):
    return _atan(x.to(tl.float32))


def atan(A):
    logger.debug("GEMS_CAMBRICON ATAN")
    out = atan_kernel(A, False)
    return out


def atan_(A):
    logger.debug("GEMS_CAMBRICON ATAN_")
    atan_kernel(A, True, out0=A)
    return A

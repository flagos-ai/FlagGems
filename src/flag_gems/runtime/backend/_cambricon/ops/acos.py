import logging

import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

_acos = tl_extra_shim.acos
logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit()
def acos_kernel(x):
    # TODO: use flag_gems.utils.tl_extra_shim help apis
    return _acos(x.to(tl.float32))


def acos(x):
    logger.debug("GEMS_CAMBRICON ACOS FORWARD")
    y = acos_kernel(x)
    return y

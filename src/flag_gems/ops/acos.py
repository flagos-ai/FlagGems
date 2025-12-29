import logging

import triton
import triton.language as tl
from flag_gems.utils import pointwise_dynamic, tl_extra_shim

_acos = tl_extra_shim.acos
logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit()
def acos_kernel(x):
    # TODO: use flag_gems.utils.tl_extra_shim help apis
    return _acos(x.to(tl.float32))


def acos(x):
    logger.debug("GEMS ACOS FORWARD")
    y = acos_kernel(x)
    return y

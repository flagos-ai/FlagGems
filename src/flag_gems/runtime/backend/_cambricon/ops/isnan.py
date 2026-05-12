import logging

import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))
_isnan = tl_extra_shim.isnan


@pointwise_dynamic(promotion_methods=[(0, "ALWAYS_BOOL")])
@triton.jit
def isnan_func(x):
    return _isnan(x.to(tl.float32))


def isnan(A):
    logger.debug("GEMS_CAMBRICON ISNAN")
    return isnan_func(A)

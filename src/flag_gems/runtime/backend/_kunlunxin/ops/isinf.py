import logging

import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
_isinf = tl_extra_shim.isinf


@pointwise_dynamic(promotion_methods=[(0, "ALWAYS_BOOL")])
@triton.jit
def isinf_func(x):
    return _isinf(x.to(tl.float32))


def isinf(A):
    logger.debug("GEMS ISINF")
    return isinf_func(A)

import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

_signbit = tl_extra_shim.signbit

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "ALWAYS_BOOL")])
@triton.jit
def signbit_func(x):
    return _signbit(x.to(tl.float32))


def signbit(A):
    logger.debug("GEMS SIGNBIT")
    return signbit_func(A)

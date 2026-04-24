import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

_isnan = tl_extra_shim.isnan


@pointwise_dynamic(promotion_methods=[(0, "ALWAYS_BOOL")])
@triton.jit
def isnan_func(x):
    # Convert input to float32
    x_float = x.to(tl.float32)
    # Detect NaN using x != x pattern
    return x_float != x_float


def isnan(A):
    logging.debug("GEMS ISNAN")
    return isnan_func(A)

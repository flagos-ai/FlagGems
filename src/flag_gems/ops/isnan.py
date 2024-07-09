import logging

import triton
import triton.language as tl

from ..utils import pointwise_dynamic

try:
    from triton.language.extra.cuda.libdevice import isnan as _isnan
except ImportError:
    try:
        from triton.language.math import isnan as _isnan
    except ImportError:
        from triton.language.libdevice import isnan as _isnan


@pointwise_dynamic(promotion_methods=[(0, "ALWAYS_BOOL")])
@triton.jit
def isnan_func(x):
    return _isnan(x.to(tl.float32))


def isnan(A):
    logging.debug("GEMS ISNAN")
    return isnan_func(A)

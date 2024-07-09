import logging

import triton
import triton.language as tl

from ..utils import pointwise_dynamic

try:
    from triton.language.extra.cuda.libdevice import isinf as _isinf
except ImportError:
    try:
        from triton.language.math import isinf as _isinf
    except ImportError:
        from triton.language.libdevice import isinf as _isinf


@pointwise_dynamic(promotion_methods=[(0, "ALWAYS_BOOL")])
@triton.jit
def isinf_func(x):
    return _isinf(x.to(tl.float32))


def isinf(A):
    logging.debug("GEMS ISINF")
    return isinf_func(A)
